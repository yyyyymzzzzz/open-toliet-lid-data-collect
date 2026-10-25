import copy
from collections import deque
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer
import cv2
import numpy as np
from observations import load_config,state_vector
from dashboard_server import Workbench,Conflict,make_handler
from record_episode import Writer
from fixed_gripper import MechanicalFixedGuard


class FakeGripper:
    def __init__(self):self.failed=False
    def status(self):
        return dict(ready=not self.failed,at_max_open=not self.failed,position_raw=12,open_target_raw=12,
                    target_raw=12,raw_status=241,fault_code=0,feedback_age_s=.01,stable_s=1.0,mode='IDLE')
    def require_open(self):
        if self.failed:raise ValueError('injected gripper failure')
        return self.status()
    def close(self):pass


class FakeMotion:
    def __init__(self):self.stopped=threading.Event();self.mode='REPLAYING'
    def status(self):return {'mode':self.mode}
    def stop(self):self.stopped.set();return {}
    def replay(self,client):raise ValueError('replay rejected')
    def close(self):pass


class FakeSource:
    """Test-only data; never imports ROS or the robot SDK."""
    def __init__(self):
        self.lock=threading.Lock();self.samples={};self.history={c:deque(maxlen=80) for c in ('joints','cam_bottom','cam_wrist')}
        self.stopping=threading.Event();self.outage=False;self.started=time.monotonic();self.counter=0
        image=np.zeros((48,64,3),np.uint8);image[:,:,1]=100
        _,jpeg=cv2.imencode('.jpg',image);self.jpeg=jpeg.tobytes()
        self._tick();self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
    def _tick(self):
        stamp=time.time_ns();received=time.monotonic_ns();self.counter+=1
        state=state_vector([self.counter*.0001,0,0,0,0,0],[.1,0,.2],[0,0,0,1])
        with self.lock:
            self.state=state
            for c in ('joints','cam_bottom','cam_wrist'):
                sample=(state if c=='joints' else self.jpeg,stamp,received)
                self.samples[c]=sample;self.history[c].append(sample)
    def _run(self):
        tick=time.monotonic()
        while not self.stopping.is_set():
            tick+=1/30;self._tick();self.stopping.wait(max(0,tick-time.monotonic()))
    def snapshot(self,decode=False):
        if self.outage:raise ValueError('injected source outage')
        with self.lock:
            obs={'observation.state':self.state.copy(),'observation.images.cam_bottom':self.jpeg,'observation.images.cam_wrist':self.jpeg}
            stamps={k:v[1] for k,v in self.samples.items()};stamps['tf']=stamps['joints']
        return obs,{'source_ns':stamps,'snapshot_mono_ns':time.monotonic_ns(),'snapshot_unix_ns':time.time_ns()}
    def close(self):self.stopping.set();self.thread.join(timeout=2)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();cfg=copy.deepcopy(load_config());cfg['data_root']=str(Path(self.tmp.name)/'episodes');cfg['minimum_free_gib']=0;cfg['gripper_policy']['verification']='device_feedback'
        self.source=FakeSource();self.service=Workbench(cfg,self.source)
        self.service.gripper=FakeGripper()
    def tearDown(self):self.service.close();self.tmp.cleanup()
    def test_custom_zero_dispatch_and_recording_interlock(self):
        self.service.motion=FakeMotion();calls=[]
        self.service.motion.set_reset_pose=lambda client,source,q:calls.append((client,source,q)) or {'saved':True}
        body={'client_id':'browser','source':'manual','joints_deg':[1,2,3,4,5,6]}
        self.assertEqual(self.service.motion_command('set_reset_pose',body),{'saved':True})
        self.assertEqual(calls,[('browser','manual',[1,2,3,4,5,6])])
        self.service.mode='RECORDING'
        with self.assertRaises(Conflict):self.service.motion_command('set_reset_pose',body)
        self.service.mode='IDLE'
        self.assertEqual(len(calls),1)

    def test_parameter_endpoint_rejects_recording_and_forwards_values(self):
        self.service.motion=FakeMotion();calls=[]
        self.service.motion.set_parameters=lambda c,p:calls.append((c,p)) or {'saved':True}
        body={'client_id':'browser','parameters':{'teach_force_n':40}}
        self.assertEqual(self.service.motion_command('set_parameters',body),{'saved':True})
        self.service.mode='RECORDING'
        with self.assertRaises(Conflict):self.service.motion_command('set_parameters',body)
        self.service.mode='IDLE';self.assertEqual(len(calls),1)

    def test_changed_motion_semantics_require_new_endpoints(self):
        self.service.motion=FakeMotion();calls=[]
        self.service.motion.home=lambda client:calls.append(('home_movej',client)) or {}
        self.service.motion.replay=lambda client:calls.append(('replay_recorded',client)) or {}
        for old in ('home','replay'):
            with self.assertRaises(Conflict):self.service.motion_command(old,{'client_id':'browser'})
        self.assertEqual(calls,[])
        self.service.motion_command('home_movej',{'client_id':'browser'})
        self.service.motion_command('replay_recorded',{'client_id':'browser'})
        self.assertEqual(calls,[('home_movej','browser'),('replay_recorded','browser')])

    def capture(self,diagnostic=False):
        result=self.service.start(diagnostic);time.sleep(2.25);self.service.stop();return result['id']
    def test_record_review_export_validate_complete_path(self):
        episode=self.capture();detail=self.service.detail(episode)
        self.assertEqual(detail['quality']['status'],'PASS')
        self.assertGreaterEqual(detail['frame_count'],60)
        self.assertEqual(self.service.review(episode,'accept','operator test fixture')['status'],'ACCEPTED')
        job=self.service.start_export('integration_fixture')
        self.service.export_thread.join(timeout=20)
        self.assertFalse(self.service.export_thread.is_alive())
        finished=self.service.exports()[0]
        self.assertEqual(finished['status'],'DONE',finished.get('error'))
        self.assertEqual(finished['validation']['status'],'PASS')
        self.assertEqual(len(finished['validation']['videos']),2)
        self.assertTrue(Path(job['path'],'meta/validation.json').is_file())
    def test_diagnostic_cannot_be_accepted_or_exported(self):
        episode=self.capture(diagnostic=True)
        with self.assertRaises(Conflict):self.service.review(episode,'accept')
        with self.assertRaises(Conflict):self.service.start_export()
        self.assertEqual(self.service.episodes()[0]['status'],'DIAGNOSTIC')
    def test_mechanical_fixed_capture_exports_explicit_attestation(self):
        self.service.cfg['gripper_policy']['verification']='mechanical_fixture'
        self.service.gripper=MechanicalFixedGuard(self.service.storage)
        self.service.confirm_fixed_gripper({'confirmed':True,'fixture_locked':True})
        episode=self.capture();self.service.review(episode,'accept','test fixture only')
        self.service.start_export('mechanical_fixture_test');self.service.export_thread.join(timeout=20)
        finished=self.service.exports()[0];self.assertEqual(finished['status'],'DONE',finished.get('error'))
        info=json.loads(Path(finished['path'],'meta/info.json').read_text())
        self.assertFalse(info['gripper_device_feedback_verified'])
        self.assertEqual(info['gripper_verification_sources'],['mechanical_fixture'])
    def test_double_start_and_source_outage_fail_closed(self):
        episode=self.service.start()['id']
        with self.assertRaises(Conflict):self.service.start()
        self.source.outage=True;self.service.capture_thread.join(timeout=2)
        self.assertFalse(self.service.capture_thread.is_alive())
        q=self.service.detail(episode)['quality']
        self.assertEqual(q['status'],'FAIL');self.assertFalse(q['accepted'])
        with self.assertRaises(Conflict):self.service.review(episode,'accept')
    def test_writer_failure_is_not_success(self):
        with patch('dashboard_server.Writer.add',side_effect=OSError('injected disk failure')):
            episode=self.service.start()['id'];self.service.capture_thread.join(timeout=2)
        q=self.service.detail(episode)['quality'];self.assertEqual(q['status'],'FAIL');self.assertFalse(q['accepted'])
    def test_replay_capture_failure_cancels_before_blocking_close(self):
        self.service.motion=FakeMotion();entered=threading.Event();release=threading.Event()
        original=Writer.close
        def blocked_close(writer):entered.set();release.wait(3);return original(writer)
        try:
            with patch.object(Writer,'add',side_effect=OSError('disk failure')),patch.object(Writer,'close',blocked_close):
                episode=self.service.replay_start()
                self.assertTrue(entered.wait(2))
                self.assertEqual(self.service.replay_episode_id,episode)
                self.assertTrue(self.service.motion.stopped.is_set(),'Stop must precede blocked disk close')
                release.set();self.service.capture_thread.join(timeout=3)
        finally:release.set()
    def test_rejected_replay_does_not_clear_episode_ownership(self):
        episode=self.capture();self.service.replay_episode_id=episode;self.service.motion=FakeMotion();self.service.motion.mode='STOPPING'
        with self.assertRaises(ValueError):self.service.motion_command('replay',{'client_id':'test'})
        self.assertEqual(self.service.replay_episode_id,episode)
        with self.assertRaises(Conflict):self.service.stop()
        with self.assertRaises(Conflict):self.service.review(episode,'accept')
        self.service.replay_finish(False,'cancelled')
        self.assertEqual(self.service.detail(episode)['quality']['status'],'FAIL')
    def test_unverified_gripper_blocks_capture_and_review(self):
        self.service.gripper.failed=True
        with self.assertRaises(ValueError):self.service.start()
        self.assertFalse(self.service.status()['observation']['ready'])
    def test_import_validation_and_no_empty_fake_runs(self):
        for data in [dict(name='empty'),dict(name='bad',training=[{'step':1,'loss':float('nan')}]),
                     dict(name='bad',evaluation={'trials':1,'successes':2}),
                     dict(name='bad',training=[{'step':2,'loss':1},{'step':1,'loss':1}])]:
            with self.assertRaises(ValueError):self.service.import_evaluation(data)
        result=self.service.import_evaluation({'name':'test fixture','training':[{'step':0,'loss':1},{'step':1,'loss':.5}],
                 'evaluation':{'trials':2,'successes':0,'mean_joint_mae_deg':0},'checkpoint':'fixture','dataset':'fixture'})
        self.assertEqual(result['evaluation']['successes'],0);self.assertEqual(len(self.service.evaluations()),1)
        self.service.import_evaluation({'name':'large fixture','training':[{'step':i,'loss':100 if i==1234 else .5} for i in range(5000)]})
        sampled=self.service.evaluations()[0]['training']
        self.assertLessEqual(len(sampled),1000);self.assertEqual(sampled[-1]['step'],4999)
        self.assertEqual(max(v['loss'] for v in sampled),100)
    def test_http_read_and_cross_site_write_guards(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(self.service));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            c=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
            c.request('GET','/api/status');r=c.getresponse();self.assertEqual(r.status,200);self.assertIn('observation',json.loads(r.read()))
            for headers in [{'Host':'untrusted.example'},{'X-R5-Panel':'1','Origin':'http://untrusted.example'}]:
                c.request('POST','/api/record/start','{}',headers);r=c.getresponse();self.assertEqual(r.status,403);r.read()
            c.request('GET','/api/episodes/..');r=c.getresponse();self.assertEqual(r.status,400);r.read()
            c.request('GET','/api/live/frame?camera=cam_bottom');r=c.getresponse();self.assertEqual(r.status,200);self.assertTrue(r.read().startswith(b'\xff\xd8'))
            c.close()
        finally:server.shutdown();server.server_close();thread.join(timeout=2)


if __name__=='__main__':unittest.main(verbosity=2)
