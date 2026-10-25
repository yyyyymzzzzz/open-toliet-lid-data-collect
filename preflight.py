"""Read-only ROS sampling and camera preview. Exit nonzero unless observations are usable."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from observations import Observations, load_config, camera_snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, default=10)
    p.add_argument('--output', type=Path, default=Path('evidence/preflight'))
    args = p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 3:
        p.error('seconds must be finite and >=3')
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    source = Observations(cfg)
    failures, successful, first, last = Counter(), 0, None, None
    stamps = {k:[] for k in ['joints', *cfg['cameras']]}
    try:
        start = time.monotonic()
        while time.monotonic()-start < args.seconds:
            try:
                obs, timing = source.snapshot(decode=False)
                successful += 1
                first = first or timing
                last = {'timing':timing, 'state':obs['observation.state'].tolist()}
            except ValueError as exc:
                failures[str(exc).split(':')[0]] += 1
            with source.lock:
                for key in stamps:
                    if key in source.samples:
                        stamp = source.samples[key][1]
                        if not stamps[key] or stamps[key][-1] != stamp:
                            stamps[key].append(stamp)
            time.sleep(.01)
        previews = {}
        for key in cfg['cameras']:
            try:
                frame = camera_snapshot(source, key)
                target = args.output / (key+'.jpg')
                if not cv2.imwrite(str(target), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError('Preview write failed')
                previews[key] = str(target)
            except ValueError as exc:
                previews[key] = str(exc)
        rates = {}
        for key, seq in stamps.items():
            dt = np.diff(seq)/1e9
            rates[key] = {'unique_samples':len(seq),
                          'observed_hz':float(1/dt.mean()) if len(dt) else 0,
                          'max_gap_s':float(dt.max()) if len(dt) else None}
        report = {'robot':cfg['robot_id'], 'config':cfg, 'successful_snapshots':successful,
                  'failure_counts':dict(failures), 'message_counts':dict(source.counts),
                  'subscriber_errors':dict(source.errors), 'rates':rates,
                  'last_valid':last, 'previews':previews, 'motion_commands_sent':0,
                  'status':'PASS' if successful >= 10 else 'FAIL'}
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report['status']=='PASS' else 2
    finally:
        source.close()


if __name__ == '__main__':
    raise SystemExit(main())
