import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import cv2
import numpy as np
import pyarrow.parquet as pq
from observations import state_vector,decode_image,validate_timing,load_config
from record_episode import quality
from export_v21 import resample,export
from validate_dataset import validate
from pi05_joint_core import CONTRACT,BoundedClient


def fixture_rows(n=90,hz=30):
    rows=[]
    for i in range(n):
        state=state_vector([i*.0001,0,0,0,0,0],[.1,0,.2],[0,0,0,1])
        stamp=1_000_000_000+round(i/hz*1e9)
        rows.append({'frame_index':i,'state':state.tolist(),'gripper':{'ready':True,'at_max_open':True,'position_raw':12,'open_target_raw':12,'raw_status':241,'fault_code':0,'feedback_age_s':0.01,'stable_s':1.0},
                     'timing':{'source_ns':dict(joints=stamp,tf=stamp,cam_bottom=stamp,cam_wrist=stamp)}})
    return rows


class PipelineTests(unittest.TestCase):
    def test_units_order_and_quaternion(self):
        s=state_vector([np.pi/2]*6,[1,2,3],[0,0,np.sin(.25),np.cos(.25)])
        np.testing.assert_allclose(s[:6],90)
        np.testing.assert_allclose(s[6:12],[1,2,3,0,0,.5],atol=1e-6)
        np.testing.assert_allclose(s[12:16],[np.cos(.25),0,0,np.sin(.25)],atol=1e-6)
        self.assertEqual(s[16],0)

    def test_bad_robot_feedback_rejected(self):
        for joints,q in [([np.nan]*6,[0,0,0,1]),([0]*6,[0,0,0,0]),([0]*5,[0,0,0,1])]:
            with self.assertRaises(ValueError):state_vector(joints,[0,0,0],q)

    def test_rgb_and_aspect_ratio(self):
        bgr=np.zeros((72,128,3),np.uint8);bgr[:,:,2]=255
        ok,png=cv2.imencode('.png',bgr);self.assertTrue(ok)
        rgb=decode_image(png.tobytes(),[480,640])
        self.assertEqual(rgb.shape,(480,640,3))
        self.assertEqual(rgb[240,320].tolist(),[255,0,0])
        self.assertEqual(rgb[:60].max(),0)

    def test_stale_and_skew_rejected(self):
        cfg=load_config();now=2_000_000_000
        stamps=dict(joints=now,tf=now,cam_bottom=now,cam_wrist=now)
        validate_timing(now,stamps,cfg)
        for key,delta in [('joints',200_000_000),('cam_wrist',60_000_000),('tf',60_000_000)]:
            bad=stamps.copy();bad[key]-=delta
            with self.assertRaises(ValueError):validate_timing(now,bad,cfg)

    def test_actual_16hz_is_not_labelled_30hz(self):
        self.assertEqual(quality(fixture_rows(hz=16),30)['status'],'FAIL')

    def test_duplicate_camera_rejected(self):
        rows=fixture_rows();rows[10]['timing']['source_ns']['cam_wrist']=rows[9]['timing']['source_ns']['cam_wrist']
        self.assertEqual(quality(rows,30)['status'],'FAIL')

    def test_next_state_actions_and_final_hold(self):
        s,a,t,_=resample(fixture_rows())
        np.testing.assert_allclose(a[:-1,:6],s[1:,:6])
        np.testing.assert_allclose(a[-1,:6],s[-1,:6])
        self.assertGreater(a[-1,0],0)
        self.assertTrue(np.all(a[:,6]==0))
        np.testing.assert_allclose(np.diff(t),1/30)

    def test_time_gap_not_interpolated_away(self):
        rows=fixture_rows();rows=rows[:20]+rows[30:]
        with self.assertRaises(ValueError):resample(rows)

    def test_true_parquet_video_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);episode=root/'raw';episode.mkdir()
            rows=fixture_rows();cfg=load_config();cfg['gripper_policy']['verification']='device_feedback'
            (episode/'config.json').write_text(json.dumps(cfg))
            (episode/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            rgb=np.zeros((48,64,3),np.uint8);rgb[:,:,1]=140
            ok,jpg=cv2.imencode('.jpg',rgb);self.assertTrue(ok)
            for cam in cfg['cameras']:
                (episode/cam).mkdir()
                for i in range(len(rows)):(episode/cam/f'{i:06d}.jpg').write_bytes(jpg.tobytes())
            info=export([episode],root/'out')
            table=pq.read_table(root/'out/data/chunk-000/episode_000000.parquet')
            self.assertEqual(len(table),info['total_frames'])
            states=np.array(table['observation.state'].to_pylist())
            actions=np.array(table['action'].to_pylist())
            np.testing.assert_array_equal(actions[:-1,:6],states[1:,:6])
            np.testing.assert_array_equal(actions[-1,:6],states[-1,:6])
            stats=json.loads((root/'out/meta/stats.json').read_text())
            np.testing.assert_allclose(stats['action']['min'],actions.min(0),atol=1e-6)
            self.assertEqual(info['codebase_version'],'v2.1')
            # Production CLI must never admit diagnostic/unaccepted captures.
            (episode/'quality.json').write_text(json.dumps({'status':'PASS','accepted':False,'diagnostic':True}))
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('export_v21.py')),
                '--episodes',str(root),'--output',str(root/'forbidden')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((root/'forbidden').exists())
            cfg['data_purpose']='synthetic_integration_test'
            (episode/'config.json').write_text(json.dumps(cfg))
            (episode/'quality.json').write_text(json.dumps({'status':'PASS','accepted':True,'diagnostic':False,'fixture_only':True}))
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('export_v21.py')),
                '--episodes',str(root),'--output',str(root/'cli_out')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=validate(root/'cli_out')
            self.assertEqual(report['status'],'PASS')
            with self.assertRaisesRegex(ValueError,'cannot be used for training'):
                validate(root/'cli_out',for_training=True)

    def test_websocket_probe_without_ros(self):
        from websockets.sync.server import serve
        from openpi_client import msgpack_numpy
        from pi05_joint_core import validate_request
        errors=[]
        def handler(ws):
            try:
                ws.send(msgpack_numpy.packb(CONTRACT))
                request=msgpack_numpy.unpackb(ws.recv(timeout=5));validate_request(request)
                ws.send(msgpack_numpy.packb({'actions':np.zeros((16,7),np.float32)}))
            except Exception as exc:errors.append(exc)
        with serve(handler,'127.0.0.1',0,max_size=16*1024*1024) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            port=server.socket.getsockname()[1]
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('infer_shadow.py')),
                '--probe-only','--port',str(port)],capture_output=True,text=True,timeout=15)
            server.shutdown();thread.join(timeout=3)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(errors)
        self.assertIn('"motion_commands_sent": 0',result.stdout)

    def test_cartesian_server_contract_rejected(self):
        from websockets.sync.server import serve
        from openpi_client import msgpack_numpy
        metadata=copy.deepcopy(CONTRACT);metadata['action_semantics']='absolute_tcp_pose'
        def handler(ws):ws.send(msgpack_numpy.packb(metadata))
        with serve(handler,'127.0.0.1',0) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                with self.assertRaisesRegex(ValueError,'action_semantics'):
                    BoundedClient(port=server.socket.getsockname()[1])
            finally:server.shutdown();thread.join(timeout=3)


if __name__=='__main__':unittest.main(verbosity=2)
