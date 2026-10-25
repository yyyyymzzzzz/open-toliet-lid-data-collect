"""Passive episode recording; operator uses the existing robot's teaching controls."""
import argparse
import json
import queue
import select
import shutil
import signal
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path
import numpy as np
from observations import Observations, load_config
from gripper_contract import verify_audit


def quality(rows, fps):
    errors = []
    if len(rows) < 2*fps:
        errors.append('Need at least 2 seconds of samples')
    timestamps = np.asarray([r['timing']['source_ns']['cam_bottom'] for r in rows], dtype=np.int64)
    dt = np.diff(timestamps)/1e9
    measured = float(1/dt.mean()) if len(dt) and dt.mean()>0 else 0
    if len(dt) and (np.any(dt<=0) or dt.max()>.1):
        errors.append('Non-monotonic source timestamps or gap >100ms')
    if measured < fps*.9:
        errors.append(f'Actual source sampling {measured:.2f}Hz is below {fps*.9:g}Hz')
    for cam in ('cam_bottom','cam_wrist'):
        values = [r['timing']['source_ns'][cam] for r in rows]
        if len(values) != len(set(values)):
            errors.append('Repeated camera source frame: '+cam)
    return {'status':'FAIL' if errors else 'PASS', 'errors':errors,
            'frames':len(rows), 'actual_source_hz':measured,
            'max_gap_s':float(dt.max()) if len(dt) else None}


class Writer:
    def __init__(self, root):
        self.root, self.error = root, None
        root.mkdir(parents=True, exist_ok=False)
        for cam in ('cam_bottom','cam_wrist'):
            (root/cam).mkdir()
        self.queue = queue.Queue(maxsize=60)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            with (self.root/'frames.jsonl').open('x') as f:
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    row, images = item
                    for cam, data in images.items():
                        (self.root/cam/f"{row['frame_index']:06d}.jpg").write_bytes(data)
                    f.write(json.dumps(row, allow_nan=False)+'\n')
                f.flush()
        except BaseException as exc:
            self.error = exc

    def add(self, row, images):
        if self.error:
            raise RuntimeError('Disk writer failed') from self.error
        try:
            self.queue.put_nowait((row, images))
        except queue.Full:
            raise RuntimeError('Disk queue full; episode rejected')

    def close(self):
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=.1)
                break
            except queue.Full:
                if self.error:
                    break
        self.thread.join(timeout=15)
        if self.thread.is_alive() or self.error:
            raise RuntimeError('Writer did not finish cleanly') from self.error


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic-seconds', type=float, help='Capture passive diagnostic only; never accepted as demonstration')
    args = p.parse_args()
    if args.diagnostic_seconds is not None and (not np.isfinite(args.diagnostic_seconds) or args.diagnostic_seconds<=0):
        p.error('diagnostic-seconds must be finite and positive')
    if args.diagnostic_seconds is None and not sys.stdin.isatty():
        p.error('Interactive demonstration recording requires a terminal')
    cfg = load_config()
    root = Path(cfg['data_root']); root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root).free < cfg['minimum_free_gib']*1024**3:
        raise RuntimeError('Insufficient disk space')
    if cfg.get('gripper_policy',{}).get('verification')=='mechanical_fixture':
        raise RuntimeError('机械固定模式请通过面板现场确认后采集，CLI不会替你确认物理固定')
    source = Observations(cfg)
    if cfg.get('gripper_policy',{}).get('verification')=='user_configured_fixed':
        from fixed_gripper import ConfiguredFixedGuard
        gripper=ConfiguredFixedGuard()
    else:
        from gripper_guard import GripperGuard
        gripper=GripperGuard(source)
    writer = None; rows = []; terminal = None; stopped = False; failure = None
    def stop(*_):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
        signal.signal(sig,stop)
    try:
        deadline = time.monotonic()+10
        while True:
            try:
                source.snapshot(decode=False);gripper.require_open(); break
            except ValueError:
                if time.monotonic()>deadline:
                    raise RuntimeError('Preflight failed; run preflight.py for details')
                time.sleep(.03)
        if args.diagnostic_seconds is None:
            terminal = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            print('IDLE: Enter 开始采集；采集中 Enter 结束；q 退出。示教/回放由现场现有控制器操作。',flush=True)
            while not stopped:
                if select.select([sys.stdin],[],[],.1)[0]:
                    key=sys.stdin.read(1)
                    if key=='q':return
                    if key in ('\r','\n'):break
        if stopped:return
        destination = root/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        writer = Writer(destination)
        (destination/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        start = time.monotonic(); previous = {}; last_good = start
        print('RECORDING '+str(destination),flush=True)
        while not stopped:
            if args.diagnostic_seconds is not None and time.monotonic()-start>=args.diagnostic_seconds:break
            if terminal is not None and select.select([sys.stdin],[],[],0)[0]:
                key=sys.stdin.read(1)
                if key=='q':stopped=True;break
                if key in ('\r','\n'):break
            grip=verify_audit(gripper.require_open())
            try:
                obs,timing = source.snapshot(decode=False)
                stamps = timing['source_ns']
                if any(stamps[c] <= previous.get(c,0) for c in cfg['cameras']):
                    if time.monotonic()-last_good>.25:raise RuntimeError('Camera source frame stopped advancing')
                    time.sleep(.002);continue
                row={'frame_index':len(rows), 'state':obs['observation.state'].tolist(), 'timing':timing,'gripper':grip}
                writer.add(row,{k:obs['observation.images.'+k] for k in cfg['cameras']})
                rows.append(row); previous=stamps; last_good=time.monotonic()
                if len(rows)%150==0 and shutil.disk_usage(root).free < cfg['minimum_free_gib']*1024**3:
                    raise RuntimeError('Disk free-space limit reached')
            except ValueError as exc:
                if time.monotonic()-last_good>.25:raise RuntimeError('Observation unavailable: '+str(exc))
            time.sleep(.002)
        writer.close(); writer=None
        report=quality(rows,cfg['fps'])
        accepted=False
        if terminal is not None and not stopped and report['status']=='PASS':
            print('REVIEW: s 接受这条成功示教；d 丢弃；q 退出。',flush=True)
            while not stopped:
                if select.select([sys.stdin],[],[],.1)[0]:
                    key=sys.stdin.read(1)
                    if key in 'sdq':accepted=key=='s';break
        report.update(accepted=accepted, diagnostic=args.diagnostic_seconds is not None,gripper_policy_validated=True,gripper_open_verified=cfg.get('gripper_policy',{}).get('verification')!='user_configured_fixed',
                      interrupted=stopped, motion_commands_sent=0)
        (destination/'quality.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    except BaseException as exc:
        failure=exc
        if writer is not None:
            try:writer.close()
            except Exception:pass
            (writer.root/'quality.json').write_text(json.dumps({'status':'FAIL','accepted':False,'error':str(exc)},indent=2)+'\n')
        raise
    finally:
        if terminal is not None:termios.tcsetattr(sys.stdin,termios.TCSADRAIN,terminal)
        try:gripper.close()
        finally:source.close()


if __name__=='__main__':main()
