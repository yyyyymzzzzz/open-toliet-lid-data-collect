"""Fake-only motion tests. No ROS initialization, SDK import, or hardware calls."""
import math
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

from manual_motion import ManualMotion, MotionError, MAX_TEACH_S


class FakeSource:
    def __init__(self):
        self.q = [10.0]*6
        self.stale = False

    def snapshot(self, decode=False):
        assert decode is False
        return {'observation.state': self.q+[0.0]*11}, {
            'source_ns': {'joints': time.time_ns()},
            'receive_mono_ns': {'joints': time.monotonic_ns()-int(.2e9 if self.stale else 0)}}


class FakeMonitor:
    def __init__(self):
        self.mode = 'idle'
        self.seq = 4
        self.generation = 0
        self.force = [0.0]*3
        self.torque = [0.0]*3
        self.age = 0

    def snapshot(self):
        return dict(mode=self.mode, dispatch_seq=self.seq, generation=self.generation,
                    idle_s=1.0, control_age_s=self.age, force_age_s=self.age,
                    force=self.force, torque=self.torque)

    def joint_limits(self):
        return [(-90.0, 90.0)]*6

    def close(self):
        pass


class FakeSDK:
    arm_dof = 6

    def __init__(self, source):
        self.source = source
        self.calls = []
        self.stop_code = 0
        self.drag_code = 0
        self.alarm = [0]
        self.query_block = None

    def rm_create_robot_arm(self, ip, port):
        self.calls.append(('connect', ip, port))
        return types.SimpleNamespace(id=1)

    def rm_get_joint_max_speed(self):
        return 0,[180.0]*6

    def rm_get_joint_en_state(self):
        return 0,[1]*6

    def rm_get_joint_min_pos(self):
        return 0, [-80.0]*6

    def rm_get_joint_max_pos(self):
        return 0, [80.0]*6

    def rm_get_current_arm_state(self):
        if self.query_block:
            self.query_block.wait(2)
        return 0, dict(joint=list(self.source.q), err={'err': self.alarm})

    def rm_set_force_drag_mode(self, mode):
        self.calls.append(('force_mode', mode))
        return 0

    def rm_set_drag_teach_sensitivity(self, grade):
        self.calls.append(('sensitivity', grade))
        return 0

    def rm_start_multi_drag_teach(self, **kw):
        self.calls.append(('drag', kw))
        return self.drag_code

    def rm_stop_drag_teach(self):
        self.calls.append(('stop_drag',))
        return 0

    def rm_set_arm_slow_stop(self):
        self.calls.append(('slow_stop',))
        return self.stop_code

    def rm_movej(self, joints, **kwargs):
        self.calls.append(('reset',list(joints),kwargs))
        self.source.q=list(joints)
        return 0

    def rm_movej_canfd(self, joints, follow):
        self.calls.append(('send', list(joints), follow))
        self.source.q = list(joints)
        return 0

    def rm_delete_robot_arm(self):
        self.calls.append(('delete',))
        return 0


def trajectory(moving=False):
    frames = []
    for i in range(61):
        u = i/60
        delta = .1*(10*u**3-15*u**4+6*u**5) if moving else 0
        frames.append({'t': i/30, 'joints_deg': [10.0+delta]*6})
    return dict(id='fake-trajectory', unit='degree', complete=True, samples=frames)


def wait_until(predicate, seconds=3):
    end = time.monotonic()+seconds
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('Timed out waiting for fake motion state')


class ManualTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.source = FakeSource()
        self.monitor = FakeMonitor()
        self.sdk = FakeSDK(self.source)
        self.telemetry = FakeSDK(self.source)
        self.created = 0
        self.finished = []
        def factory():
            self.created += 1
            return self.sdk
        self.motion = ManualMotion(self.source, self.tmp.name, sdk_factory=factory, monitor=self.monitor,telemetry_factory=lambda:self.telemetry,
                                   on_replay_start=lambda:'fake-new-episode',
                                   on_replay_finish=lambda success,error:self.finished.append((success,error)))

    def tearDown(self):
        if self.sdk.query_block:
            self.sdk.query_block.set()
        if self.telemetry.query_block:
            self.telemetry.query_block.set()
        try:
            self.motion.close()
        except MotionError:
            pass
        self.tmp.cleanup()

    def arm(self):
        self.motion.arm('owner', True)

    def test_lazy_sdk_and_arming_has_no_motion_writes(self):
        self.assertEqual(self.created, 0)
        self.assertEqual(self.motion.status()['mode'], 'DISARMED')
        self.arm()
        self.assertEqual(self.created, 1)
        self.assertEqual(self.sdk.calls, [('connect', '192.168.1.18', 8080)])
        self.assertEqual(self.motion.limits, [(-80.0, 80.0)]*6)
        self.assertFalse(self.motion.status()['validated_on_hardware'])

    def test_explicit_confirmation_and_other_controller_interlock(self):
        with self.assertRaises(MotionError):
            self.motion.arm('owner', False)
        self.monitor.mode = 'track'
        with self.assertRaises(MotionError):
            self.arm()
        self.assertEqual(self.created, 0)

    def test_lease_expires_even_without_browser_calls(self):
        self.arm()
        self.motion.deadline = time.monotonic()+.08
        wait_until(lambda:self.motion.mode == 'DISARMED')
        self.assertIn(('slow_stop',), self.sdk.calls)
        self.assertIsNone(self.motion.owner)
        with self.assertRaises(MotionError):
            self.motion.heartbeat('owner')

    def test_stop_failure_latches_fault(self):
        self.arm()
        self.sdk.stop_code = -2
        self.motion.stop()
        wait_until(lambda:self.motion.mode == 'FAULT')
        self.assertFalse(self.motion.status()['ready'])
        with self.assertRaises(MotionError):
            self.motion.drag_start('owner')
        with self.assertRaises(MotionError):
            self.motion.arm('owner', True)
        self.assertIn('停止未确认', self.motion.error)

    def test_start_failure_attempts_both_stops(self):
        self.arm()
        self.sdk.drag_code = -2
        with self.assertRaises(MotionError):
            self.motion.drag_start('owner')
        wait_until(lambda:self.motion.mode == 'FAULT')
        self.assertIn(('stop_drag',), self.sdk.calls)
        self.assertIn(('slow_stop',), self.sdk.calls)
        self.assertIn(('sensitivity', 100), self.sdk.calls)
        self.assertIn(('drag', {'mode':0, 'singular_wall':1}), self.sdk.calls)

    def test_degree_units_and_replay_cancel_marks_only_new_episode_failed(self):
        self.arm()
        self.motion.trajectory = trajectory()
        self.motion.replay('owner')
        wait_until(lambda:any(c[0]=='send' for c in self.sdk.calls))
        self.motion.stop()
        wait_until(lambda:self.motion.mode in ('ARMED','FAULT'))
        sends = [c for c in self.sdk.calls if c[0]=='send']
        self.assertTrue(sends)
        self.assertEqual(sends[0], ('send', [10.0]*6, True))
        self.assertEqual(len(self.finished), 1)
        self.assertFalse(self.finished[0][0])
        self.assertFalse(self.motion.fault_latched, self.motion.error)

    def test_replay_start_callback_failure_never_finishes_old_episode(self):
        self.arm()
        self.motion.trajectory = trajectory()
        def fail():
            raise RuntimeError('existing passive capture busy')
        self.motion.on_replay_start = fail
        self.motion.replay('owner')
        wait_until(lambda:self.motion.mode == 'FAULT')
        self.assertFalse(any(c[0]=='send' for c in self.sdk.calls))
        self.assertEqual(self.finished, [])

    def test_replay_starts_from_current_pose_without_alignment_segment(self):
        self.arm()
        self.motion.trajectory = trajectory(moving=True)
        self.source.q = [14.0]*6
        self.motion.replay('owner')
        wait_until(lambda:any(c[0]=='send' for c in self.sdk.calls))
        self.motion.stop()
        wait_until(lambda:self.motion.mode in ('ARMED','FAULT'))
        self.assertFalse(self.motion.fault_latched, self.motion.error)
        self.assertEqual(list((Path(self.tmp.name)/'manual_motion').glob('*.alignment.json')), [])
        self.assertEqual([c for c in self.sdk.calls if c[0]=='send'][0][2], True)

    def test_controller_dispatch_change_stops_and_latches(self):
        self.arm()
        self.monitor.seq += 1
        wait_until(lambda:self.motion.mode == 'FAULT')
        self.assertIn('dispatch', self.motion.error)
        self.assertIn(('slow_stop',), self.sdk.calls)

    def test_force_threshold_stops(self):
        self.arm()
        self.monitor.force = [20.01, 0, 0]
        wait_until(lambda:self.motion.mode == 'FAULT')
        self.assertIn('20 N', self.motion.error)

    def test_stale_feedback_cannot_claim_stationary(self):
        self.arm()
        self.source.stale = True
        self.motion.stop()
        wait_until(lambda:self.motion.mode == 'FAULT')
        self.assertIn('停止未确认', self.motion.error)

    def test_lease_watchdog_not_blocked_by_native_query(self):
        self.arm()
        self.telemetry.query_block = threading.Event()
        self.motion.controller_checked = time.monotonic()-.1
        time.sleep(.04)
        self.motion.deadline = time.monotonic()+.05
        start = time.monotonic()
        wait_until(lambda:self.motion.cancel.is_set(), seconds=.3)
        self.assertLess(time.monotonic()-start, .3)
        self.telemetry.query_block.set()

    def test_old_worker_cannot_revive_after_stop_and_new_action(self):
        self.arm()
        self.motion.trajectory = trajectory()
        old_ticket = self.motion.operation_generation
        entered, release = threading.Event(), threading.Event()
        real_thread = threading.Thread
        paused = [False]
        class PauseStart(real_thread):
            def start(thread):
                if getattr(thread._target, '__name__', '') == '_play' and not paused[0]:
                    paused[0] = True
                    entered.set()
                    release.wait(2)
                return super().start()
        errors = []
        def launch():
            try:
                self.motion.replay('owner')
            except Exception as exc:
                errors.append(str(exc))
        with patch('manual_motion.threading.Thread', PauseStart):
            caller = real_thread(target=launch)
            caller.start()
            self.assertTrue(entered.wait(3))
            stopper = real_thread(target=self.motion.stop)
            stopper.start()
            wait_until(self.motion.cancel.is_set)
            release.set()
            caller.join(2)
            stopper.join(2)
            wait_until(lambda:self.motion.mode in ('ARMED','FAULT'))
            self.assertFalse(self.motion.fault_latched, self.motion.error)
            self.assertFalse(any(c[0]=='send' for c in self.sdk.calls))
            self.motion.heartbeat('owner')
            self.motion.replay('owner')
            wait_until(lambda:any(c[0]=='send' for c in self.sdk.calls))
            before = len([c for c in self.sdk.calls if c[0]=='send'])
            with self.assertRaisesRegex(MotionError,'generation'):
                self.motion._sdk('rm_movej_canfd',[50.0]*6,follow=False,ticket=old_ticket)
            self.assertFalse(any(c[0]=='send' and c[1]==[50.0]*6 for c in self.sdk.calls))
            self.motion.stop()

    def test_complete_drag_saves_only_valid_degree_trajectory(self):
        self.arm()
        self.motion.drag_start('owner')
        for i in range(70):
            u=min(1,i/60);self.source.q=[10+.5*(10*u**3-15*u**4+6*u**5)]*6
            time.sleep(1/30)
        self.motion.drag_stop('owner')
        wait_until(lambda:self.motion.mode in ('ARMED','FAULT'))
        self.assertFalse(self.motion.fault_latched, self.motion.error)
        self.assertIsNotNone(self.motion.trajectory, self.motion.error)
        self.assertEqual(self.motion.trajectory['unit'], 'degree')
        self.assertTrue(self.motion.trajectory['complete'])
        self.assertGreaterEqual(len(self.motion.trajectory['samples']), 40)
        self.assertLess(self.motion.trajectory['samples'][0]['joints_deg'][0],10.02)
        self.assertFalse(any(c[0]=='send' for c in self.sdk.calls))

    def test_reset_native_speed_and_target_no_teaching_required(self):
        self.motion.reset_pose=[0.0]*6
        self.arm()
        self.assertTrue(self.motion.status()['reset_ready'])
        self.motion.reset('owner')
        wait_until(lambda:self.motion.mode=='ARMED')
        self.assertIn(('reset',[0.0]*6,dict(v=100,r=0,connect=0,block=0)),self.sdk.calls)
        self.assertFalse(any(c[0]=='send' for c in self.sdk.calls))
        self.assertFalse(self.motion.fault_latched,self.motion.error)

    def test_reset_rejects_target_outside_limits(self):
        self.arm()
        self.assertFalse(self.motion.status()['reset_ready'])
        with self.assertRaisesRegex(MotionError,'range'):self.motion.reset('owner')
        self.assertFalse(any(c[0]=='reset' for c in self.sdk.calls))

    def test_reset_cancelled_ticket_never_sends(self):
        self.arm();self.motion.reset_pose=[0.0]*6
        self.motion.cancel.set()
        self.motion._reset([0.0]*6,self.motion.operation_generation-1)
        self.assertFalse(any(c[0]=='reset' for c in self.sdk.calls))

    def test_control_feedback_uses_latest_joint_sample_not_camera_alignment(self):
        self.source.joint_feedback=lambda:([12.0]*6,12345)
        self.assertEqual(self.motion._feedback(),([12.0]*6,12345))

    def test_save_current_pose_has_no_motion_and_survives_restart(self):
        self.arm();self.source.q=[12.0]*6
        before=list(self.sdk.calls)
        status=self.motion.set_reset_pose('owner','current')
        self.assertEqual(status['default_initial_pose'],[12.0]*6)
        self.assertEqual(status['reset_target_source'],'user_current_pose')
        self.assertEqual(self.sdk.calls,before)
        self.motion.close()
        self.motion=ManualMotion(self.source,self.tmp.name,sdk_factory=lambda:self.sdk,monitor=self.monitor,telemetry_factory=lambda:self.telemetry)
        self.assertEqual(self.motion.status()['default_initial_pose'],[12.0]*6)
        self.assertEqual(self.motion.mode,'DISARMED')

    def test_save_manual_validates_and_does_not_move(self):
        self.arm();before=list(self.sdk.calls)
        status=self.motion.set_reset_pose('owner','manual',[1,2,3,4,5,6])
        self.assertEqual(status['default_initial_pose'],[1,2,3,4,5,6])
        self.assertEqual(self.sdk.calls,before)
        for q in ([81]*6,[0]*5,[math.nan]*6,['1']*6,[True]*6):
            with self.assertRaises(MotionError):self.motion.set_reset_pose('owner','manual',q)
        self.assertEqual(self.motion.reset_pose,[1,2,3,4,5,6])

    def test_save_zero_wrong_owner_busy_and_stale_rejected(self):
        with self.assertRaises(MotionError):self.motion.set_reset_pose('owner','current')
        self.arm()
        with self.assertRaises(MotionError):self.motion.set_reset_pose('other','current')
        self.motion.mode='TEACHING'
        with self.assertRaises(MotionError):self.motion.set_reset_pose('owner','current')
        self.motion.mode='ARMED';self.source.stale=True
        with self.assertRaises(MotionError):self.motion.set_reset_pose('owner','current')
        self.source.stale=False
        self.assertFalse(self.motion.reset_file.exists())

    def test_save_zero_write_failure_preserves_target(self):
        self.arm();previous=list(self.motion.reset_pose)
        with patch('manual_motion.os.replace',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):self.motion.set_reset_pose('owner','manual',[1]*6)
        self.assertEqual(self.motion.reset_pose,previous)
        self.assertFalse(list(self.motion.storage.glob('reset_pose.*.tmp')))

    def test_corrupt_saved_zero_blocks_reset_until_repaired(self):
        self.motion.close()
        (Path(self.tmp.name)/'manual_motion/reset_pose.json').write_text('{broken')
        self.motion=ManualMotion(self.source,self.tmp.name,sdk_factory=lambda:self.sdk,monitor=self.monitor,telemetry_factory=lambda:self.telemetry)
        self.arm();self.assertFalse(self.motion.status()['reset_ready'])
        with self.assertRaisesRegex(MotionError,'读取失败'):self.motion.reset('owner')
        self.motion.set_reset_pose('owner','manual',[0]*6)
        self.assertTrue(self.motion.status()['reset_ready'])
        self.assertIsNone(self.motion.reset_load_error)

    def test_slow_drag_ack_does_not_starve_telemetry(self):
        self.arm()
        original=self.sdk.rm_start_multi_drag_teach
        def slow(**kw):time.sleep(.35);return original(**kw)
        self.sdk.rm_start_multi_drag_teach=slow
        self.motion.drag_start('owner')
        self.assertEqual(self.motion.mode,'TEACHING',self.motion.error)
        self.assertLess(time.monotonic()-self.motion.controller_checked,.15)
        self.motion.stop()
        wait_until(lambda:self.motion.mode=='ARMED')

    def test_drag_cancel_preserves_first_fault_and_no_late_start(self):
        self.arm()
        entered=threading.Event()
        def slow(grade):entered.set();time.sleep(.35);return 0
        self.sdk.rm_set_drag_teach_sensitivity=slow
        errors=[]
        def start():
            try:self.motion.drag_start('owner')
            except MotionError as e:errors.append(str(e))
        caller=threading.Thread(target=start);caller.start();self.assertTrue(entered.wait(1))
        self.monitor.seq+=1
        caller.join(2)
        wait_until(lambda:self.motion.mode=='FAULT')
        self.assertTrue(errors)
        self.assertIn('control ownership lost',errors[0])
        self.assertNotEqual(self.motion.error,'Operation generation cancelled')
        self.assertFalse(any(c[0]=='drag' for c in self.sdk.calls))

    def test_alarm_during_drag_start_still_stops(self):
        self.arm()
        original=self.sdk.rm_start_multi_drag_teach
        def slow(**kw):
            self.telemetry.alarm=[123]
            time.sleep(.25)
            return original(**kw)
        self.sdk.rm_start_multi_drag_teach=slow
        with self.assertRaisesRegex(MotionError,'Controller alarm'):self.motion.drag_start('owner')
        wait_until(lambda:self.motion.mode=='FAULT')
        self.assertTrue(self.motion.cancel.is_set())

    def force_fault(self):
        self.arm();self.monitor.force=[21,0,0]
        wait_until(lambda:self.motion.mode=='FAULT' and self.motion.stop_confirmed and not self.motion.stop_thread.is_alive())

    def test_recovery_after_force_returns_to_normal_sends_no_motion(self):
        self.force_fault();self.monitor.force=[0,0,0]
        before=list(self.sdk.calls)
        status=self.motion.recover('owner')
        self.assertEqual(status['mode'],'ARMED')
        self.assertEqual(self.sdk.calls,before)
        self.assertFalse(self.motion.fault_latched)
        self.assertEqual(self.motion.owner,'owner')
        self.assertFalse(self.motion.cancel.is_set())
        self.assertTrue(list(self.motion.storage.glob('*.recovery.json')))

    def test_recovery_rejects_active_force_controller_alarm_stale_and_disabled(self):
        self.force_fault()
        with self.assertRaisesRegex(MotionError,'20 N'):self.motion.recover('owner')
        self.monitor.force=[0,0,0];self.telemetry.alarm=[55]
        with self.assertRaisesRegex(MotionError,'alarm'):self.motion.recover('owner')
        self.telemetry.alarm=[0];self.source.stale=True
        with self.assertRaises(MotionError):self.motion.recover('owner')
        self.source.stale=False;self.sdk.rm_get_joint_en_state=lambda:(0,[0]*6)
        with self.assertRaisesRegex(MotionError,'使能'):self.motion.recover('owner')
        self.assertEqual(self.motion.mode,'FAULT');self.assertTrue(self.motion.cancel.is_set())

    def test_recovery_rejects_failed_stop(self):
        self.arm();self.sdk.stop_code=-2;self.motion.stop()
        wait_until(lambda:self.motion.mode=='FAULT')
        self.assertFalse(self.motion.status()['recovery_available'])
        with self.assertRaises(MotionError):self.motion.recover('owner')

    def test_stop_during_recovery_cancels_recovery(self):
        self.force_fault();self.monitor.force=[0,0,0]
        errors=[]
        def recover():
            try:self.motion.recover('owner')
            except MotionError as exc:errors.append(str(exc))
        t=threading.Thread(target=recover);t.start()
        wait_until(lambda:self.motion.recovering)
        self.motion.stop();t.join(2)
        self.assertTrue(errors);self.assertTrue(self.motion.fault_latched)
        wait_until(lambda:self.motion.mode=='FAULT')

    def test_native_drag_contact_does_not_use_software_force_torque_cutoff(self):
        self.arm()
        with self.motion.lock:
            self.monitor.force=[80,0,0];self.monitor.torque=[8,0,0];self.motion.mode='TEACHING'
            self.motion._monitor()
            self.motion.mode='ARMED'
            with self.assertRaisesRegex(MotionError,'20 N'):self.motion._monitor()
            self.monitor.force=[0,0,0];self.monitor.torque=[0,0,0]
        self.assertEqual(self.motion.parameters['teach_force_n'],40)

    def test_parameters_bounded_persist_and_never_move(self):
        original=list(self.sdk.calls)
        self.motion.set_parameters('owner',{'teach_force_n':50,'reset_speed_percent':60,'replay_speed':.5})
        self.assertEqual(self.sdk.calls,original)
        self.assertEqual(self.motion.parameters['force_n'],20)
        from manual_motion import validate_parameters
        for value in ({'teach_force_n':61},{'force_n':21},{'torque_nm':4},{'reset_speed_percent':10.5},{'replay_speed':0},{'max_velocity_deg_s':11},{'tracking_deg':11},{'unknown':1}):
            with self.assertRaises(MotionError):validate_parameters(value)
        self.motion.close()
        self.motion=ManualMotion(self.source,self.tmp.name,sdk_factory=lambda:self.sdk,monitor=self.monitor,telemetry_factory=lambda:self.telemetry)
        self.assertEqual(self.motion.parameters['teach_force_n'],50)
        self.assertEqual(self.motion.parameters['reset_speed_percent'],60)

    def test_changed_native_reset_speed_applied(self):
        self.motion.set_parameters('owner',{'reset_speed_percent':40})
        self.motion.reset_pose=[0]*6;self.arm();self.motion.reset('owner')
        wait_until(lambda:self.motion.mode=='ARMED')
        self.assertIn(('reset',[0.0]*6,dict(v=40,r=0,connect=0,block=0)),self.sdk.calls)

    def test_settings_cannot_change_in_fault_or_other_owners_session(self):
        self.arm()
        with self.assertRaises(MotionError):self.motion.set_parameters('other',{'teach_force_n':50})
        with self.motion.lock:
            self.motion.mode='TEACHING'
            with self.assertRaises(MotionError):self.motion.set_parameters('owner',{'teach_force_n':50})
            self.motion.mode='ARMED'
        self.monitor.force=[21,0,0]
        wait_until(lambda:self.motion.mode=='FAULT')
        with self.assertRaises(MotionError):self.motion.set_parameters('owner',{'teach_force_n':50})

    def test_lease_expiry_racing_guard_is_normal_disarm(self):
        self.arm()
        self.motion._request_stop('5 second owner lease expired',fault=True)
        wait_until(lambda:self.motion.mode=='DISARMED')
        self.assertFalse(self.motion.fault_latched)
        self.assertIn(('slow_stop',),self.sdk.calls)

    def test_raw_teaching_saved_even_if_not_replayable(self):
        self.arm();self.motion.samples=trajectory()['samples']
        self.motion.samples[10]['t']=self.motion.samples[9]['t']
        self.motion.teaching_window=types.SimpleNamespace(moved=True,waiting=[],trimmed_s=0,detected_at=1,origin=0)
        self.motion._save_teaching()
        self.assertTrue(self.motion.trajectory['raw_saved'])
        self.assertFalse(self.motion.trajectory['replay_ready'])
        saved=self.motion.storage/(self.motion.trajectory['id']+'.json')
        self.assertTrue(saved.exists())
        import json
        self.assertEqual(json.loads(saved.read_text())['samples'],self.motion.samples)

    def test_raw_sharp_path_saved_and_retimed_then_restored(self):
        self.arm();self.motion.samples=trajectory()['samples']
        for f in self.motion.samples[30:]:f['joints_deg']=[11.0]*6
        self.motion.teaching_window=types.SimpleNamespace(moved=True,waiting=[],trimmed_s=0,detected_at=1,origin=0)
        self.motion._save_teaching()
        self.assertTrue(self.motion.trajectory['replay_ready'])
        ident=self.motion.trajectory['id'];self.motion.close()
        self.motion=ManualMotion(self.source,self.tmp.name,sdk_factory=lambda:self.sdk,monitor=self.monitor,telemetry_factory=lambda:self.telemetry)
        self.assertEqual(self.motion.trajectory['id'],ident)
        self.assertFalse(self.motion.trajectory['replay_ready'])
        self.arm();self.assertTrue(self.motion.trajectory['replay_ready'])
        self.assertFalse(any(c[0]=='send' for c in self.sdk.calls))

    def test_replay_starts_without_offline_planner(self):
        self.arm();self.motion.trajectory=trajectory()
        self.motion.replay('owner')
        wait_until(lambda:any(c[0]=='send' for c in self.sdk.calls))
        self.motion.stop();wait_until(lambda:self.motion.mode in ('ARMED','FAULT'))
        self.assertFalse(self.motion.fault_latched,self.motion.error)
        self.assertFalse(self.motion.planning)

    def test_controller_limits_override_model_only_lower_bound(self):
        model_limits=[(-90,90)]*6;model_limits[4]=(30,90)
        self.monitor.joint_limits=lambda:model_limits
        self.source.q=[10,10,10,10,29,10]
        self.arm()
        self.assertEqual(self.motion.limits,[(-80,80)]*6)
        self.assertEqual(self.motion.status()['joint_limit_source'],'controller')
        self.motion._guard()
        self.motion.stop();wait_until(lambda:self.motion.mode=='ARMED')
        self.assertTrue(self.motion.stop_confirmed)
        self.assertFalse(self.motion.fault_latched)

    def test_stop_can_confirm_stationary_outside_hardware_range_but_cannot_recover(self):
        self.arm();self.source.q=[81]*6
        wait_until(lambda:self.motion.mode=='FAULT' and self.motion.stop_confirmed)
        self.assertIn('exceeds validated range',self.motion.error)
        self.assertNotIn('停止未确认',self.motion.error)
        with self.assertRaisesRegex(MotionError,'range'):self.motion.recover('owner')
        self.assertTrue(self.motion.fault_latched)

    def test_repeated_stop_fault_text_does_not_grow_recursively(self):
        self.motion.primary_fault='Joint 5 exceeds validated range'
        self.motion.error=self.motion.primary_fault
        for _ in range(10):self.motion._fault('停止未确认：Joint 5 exceeds validated range')
        self.assertEqual(self.motion.error.count('最初触发'),1)
        self.assertEqual(self.motion.primary_fault,'Joint 5 exceeds validated range')

    def test_short_sdk_delay_extends_schedule_without_catchup_or_dropped_points(self):
        self.arm();self.assertEqual(self.motion.parameters['replay_speed'],1)
        points=[[10+i*.0001]*6 for i in range(6)];times=[];ends=[]
        original=self.sdk.rm_movej_canfd
        def delayed(q,follow):
            times.append(time.monotonic())
            if len(times)==2:time.sleep(.07)
            ret=original(q,follow);ends.append(time.monotonic());return ret
        self.sdk.rm_movej_canfd=delayed
        self.motion.mode='HOMING';self.motion.command=[10]*6
        self.motion.worker=threading.Thread(target=self.motion._play,args=(points,False,self.motion.operation_generation))
        self.motion.worker.start();wait_until(lambda:self.motion.mode=='ARMED')
        self.assertFalse(self.motion.fault_latched,self.motion.error)
        sent=[c[1] for c in self.sdk.calls if c[0]=='send']
        self.assertEqual(sent[:len(points)],points)
        self.assertEqual(sent[-1],points[-1])
        self.assertLess(times[2]-ends[1],.02)
        self.assertTrue(all(c[2] is True for c in self.sdk.calls if c[0]=='send'))

    def test_slow_canfd_call_does_not_abort_playback(self):
        self.arm();original=self.sdk.rm_movej_canfd
        def stalled(q,follow):time.sleep(.05);return original(q,follow)
        self.sdk.rm_movej_canfd=stalled
        self.motion.mode='HOMING';self.motion.command=[10]*6
        points=[[10]*6]*4
        self.motion.worker=threading.Thread(target=self.motion._play,args=(points,False,self.motion.operation_generation))
        self.motion.worker.start();wait_until(lambda:self.motion.mode=='ARMED',seconds=4)
        self.assertFalse(self.motion.fault_latched,self.motion.error)
        self.assertGreaterEqual(len([c for c in self.sdk.calls if c[0]=='send']),len(points))

    def test_home_is_one_native_movej_to_recorded_start(self):
        self.arm();self.motion.trajectory=trajectory();self.source.q=[14]*6
        self.motion.home('owner');wait_until(lambda:self.motion.mode=='ARMED')
        self.assertEqual([c for c in self.sdk.calls if c[0]=='reset'],[('reset',[10.0]*6,dict(v=80,r=0,connect=0,block=0))])
        self.assertFalse(any(c[0]=='send' for c in self.sdk.calls))

    def test_tracking_matches_feedback_timestamp_not_newer_target(self):
        self.arm();self.motion.command=[15]*6
        self.motion._remember_command([0]*6,0)
        self.motion._remember_command([15]*6,200)
        self.motion._check_tracking([0]*6,100)
        self.assertEqual(self.motion.tracking_status['error_deg'],0)
        self.assertEqual(self.motion.tracking_status['raw_latest_error_deg'],15)
        with self.assertRaisesRegex(MotionError,'exceeds 10'):self.motion._check_tracking([0]*6,201)

    def test_tracking_ten_degree_limit_is_finite_and_not_disabled(self):
        self.arm();self.motion.command=[10]*6;self.motion._remember_command([10]*6,0)
        self.motion._check_tracking([5]*6,1)
        with self.assertRaisesRegex(MotionError,'exceeds 10'):self.motion._check_tracking([-1]*6,2)
        self.assertEqual(self.motion.tracking_status['joint'],1)

    def test_wrong_dof_rejected(self):
        self.sdk.arm_dof = 7
        with self.assertRaisesRegex(MotionError,'six-axis'):
            self.arm()
        self.assertFalse(any(c[0]=='send' for c in self.sdk.calls))


if __name__=='__main__':
    unittest.main(verbosity=2)
