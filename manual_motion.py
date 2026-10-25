"""Explicit, leased arm-only manual controls. No SDK connection until arm().

Not hardware validated. This is NOT a hardware emergency stop or a global robot
control lock. An operator must release other outputs and park the base. Native
SDK calls are serialized; a stuck native call cannot be killed safely by Python.
A stop timeout latches FAULT and never reports that the arm is stationary.

Injected monitor contract: snapshot() -> mode, dispatch_seq, generation,
control_age_s, force_age_s, idle_s, force[3], torque[3]; joint_limits() -> six
(lower, upper) DEGREE pairs; close(). Test SDKs expose the verified SDK 1.1.6 API.
"""
from __future__ import annotations
import bisect
from collections import deque
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid
import xml.etree.ElementTree as ET

LEASE_S = 5.0
FRESH_S = .15
FORCE_N = 20.0
TORQUE_NM = 3.0
STREAM_HZ = 125
TEACH_HZ = 30
SPEED = 1.0
MAX_VEL = 10.0
MAX_ACC = 30.0
MAX_STEP = 1.0
TRACKING = 10.0
START_TOL = 2.0
LATE_S = .05
STREAM_STALL_S = .15
STOP_TIMEOUT = 3.0
STABLE_S = .5
STABLE_TOL = .1
MAX_TEACH_S = 140.0
# Motion calls below follow old_code/collect_open_toilet_lid.py and drag_teach_utils.py.
DRAG_MODE = 0
DRAG_SENSITIVITY = 100
LEGACY_HOME_SPEED = 80
REPLAY_DT = 0.010
REPLAY_SMOOTH = 11
PASSIVE_MODES = ('TEACHING', 'HOMING', 'RESETTING', 'REPLAYING')


DEFAULT_PARAMETERS=dict(teach_force_n=40.0,force_n=20.0,torque_nm=3.0,
    reset_speed_percent=100,replay_speed=1.0,max_velocity_deg_s=10.0,
    max_acceleration_deg_s2=30.0,tracking_deg=10.0,endpoint_tolerance_deg=2.0,max_step_deg=1.0)
PARAMETER_BOUNDS=dict(teach_force_n=(5,60),force_n=(5,20),torque_nm=(.5,3),
    reset_speed_percent=(1,100),replay_speed=(.1,1),max_velocity_deg_s=(1,10),
    max_acceleration_deg_s2=(5,30),tracking_deg=(.5,10),endpoint_tolerance_deg=(.1,2),max_step_deg=(.1,1))

def validate_parameters(value,base=None):
    if not isinstance(value,dict) or not value or set(value)-set(DEFAULT_PARAMETERS):
        raise MotionError('Invalid or unknown motion parameters')
    result=dict(DEFAULT_PARAMETERS if base is None else base)
    for key,v in value.items():
        lo,hi=PARAMETER_BOUNDS[key]
        if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi:
            raise MotionError(key+' must be within '+str(lo)+'–'+str(hi))
        if key=='reset_speed_percent' and int(v)!=v:raise MotionError('Reset speed must be an integer percent')
        result[key]=int(v) if key=='reset_speed_percent' else float(v)
    return result


class MotionError(ValueError):
    pass


def vector(value, n=6):
    try:
        result = [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise MotionError('Invalid numeric feedback') from exc
    if len(result) != n or not all(math.isfinite(v) for v in result):
        raise MotionError('Invalid or non-finite feedback dimensions')
    return result


def maxdiff(a, b):
    return max(abs(x-y) for x, y in zip(a, b))


def read_limits():
    """Resolve the deployed arm configuration; no guessed model or limits."""
    import yaml
    from ament_index_python.packages import get_package_share_directory
    config_path = Path(get_package_share_directory('arm_control'))/'config/robot_config.yaml'
    arm = yaml.safe_load(config_path.read_text())['arm']
    path = Path(get_package_share_directory(arm['description_package']))/arm['urdf_file']
    root = ET.parse(path).getroot()
    names = arm.get('rm_joint_names')
    if names is None:
        # Project observation contract explicitly orders joint1..joint6.
        names = ['joint'+str(i) for i in range(1, 7)]
    if names != ['joint'+str(i) for i in range(1, 7)]:
        raise MotionError('Deployed URDF joint ordering does not match observation contract')
    result = []
    for name in names:
        joint = root.find("./joint[@name='"+name+"']")
        limit = joint.find('limit') if joint is not None else None
        if joint is None or joint.get('type') != 'revolute' or limit is None:
            raise MotionError('Missing finite URDF revolute joint limit: '+name)
        lo, hi = (math.degrees(float(limit.attrib[k])) for k in ('lower', 'upper'))
        if not math.isfinite(lo+hi) or lo >= hi:
            raise MotionError('Invalid URDF limits: '+name)
        result.append((lo, hi))
    return result


class ROSMonitor:
    """Independent ROS executor, read-only subscriptions; transient conflicts latch."""
    def __init__(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from robot_interfaces.msg import ControlLog, RobotTrajectoryPoint
        from rm_ros_interfaces.msg import Rmerr,Jointerrorcode,Jointenflag
        if not rclpy.ok():
            raise MotionError('Observation ROS context must already be running')
        self.lock = threading.Lock()
        self.dead = threading.Event()
        self.error = None
        self.ctrl = self.force = None
        self.controller_packets={}
        self.generation = 0
        self.change_reason = None
        self.idle_since = None
        self.limits = read_limits()
        self.node = rclpy.create_node('r5_manual_guard_'+str(os.getpid()))
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.subs = [self.node.create_subscription(ControlLog, '/arm_control/log/ctrl', self._control, qos_profile_sensor_data),
                     self.node.create_subscription(RobotTrajectoryPoint, '/arm_control/robot_state', self._force, qos_profile_sensor_data),
                     self.node.create_subscription(Rmerr,'/rm_driver/udp_rm_err',self._system_alarm,qos_profile_sensor_data),
                     self.node.create_subscription(Jointerrorcode,'/rm_driver/udp_joint_error_code',self._joint_alarm,qos_profile_sensor_data),
                     self.node.create_subscription(Jointenflag,'/rm_driver/udp_joint_en_flag',self._joint_enable,qos_profile_sensor_data)]
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _system_alarm(self,msg):
        with self.lock:self.controller_packets['system']=(time.monotonic(),dict(count=int(msg.err_len),codes=list(msg.err)))
    def _joint_alarm(self,msg):
        with self.lock:self.controller_packets['joints']=(time.monotonic(),dict(dof=int(msg.dof),codes=list(msg.joint_error)))
    def _joint_enable(self,msg):
        with self.lock:self.controller_packets['enabled']=(time.monotonic(),list(msg.joint_en_flag))
    def controller_snapshot(self):
        from ros_controller_telemetry import validate_packets
        with self.lock:return validate_packets(dict(self.controller_packets))

    def _control(self, msg):
        now = time.monotonic()
        with self.lock:
            changed = self.ctrl is not None and int(msg.dispatch_seq) != self.ctrl[1]
            # A late but still-idle sample is not another controller taking over.
            if msg.mode != 'idle' or changed:
                self.change_reason=('Other controller mode changed to '+str(msg.mode) if msg.mode!='idle'
                                    else 'Other controller dispatch sequence changed')
                self.generation += 1
                self.idle_since = None
            elif self.idle_since is None:
                self.idle_since = now
            self.ctrl = (str(msg.mode), int(msg.dispatch_seq), now)

    def _force(self, msg):
        w = msg.wrench
        with self.lock:
            self.force = ([w.force.x, w.force.y, w.force.z], [w.torque.x, w.torque.y, w.torque.z], time.monotonic())

    def _spin(self):
        try:
            while not self.dead.is_set():
                self.executor.spin_once(timeout_sec=.02)
        except Exception as exc:
            self.error = str(exc)

    def joint_limits(self):
        return self.limits

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            if self.error or self.ctrl is None or self.force is None:
                raise MotionError('Waiting for fresh control log and wrench: '+str(self.error or 'missing topics'))
            return dict(mode=self.ctrl[0], dispatch_seq=self.ctrl[1], generation=self.generation,change_reason=self.change_reason,
                        control_age_s=now-self.ctrl[2], force_age_s=now-self.force[2],
                        idle_s=now-self.idle_since if self.idle_since else 0,
                        force=list(self.force[0]), torque=list(self.force[1]))

    def close(self):
        self.dead.set()
        self.thread.join(1)
        if self.thread.is_alive():
            raise MotionError('Read-only monitor did not exit')
        self.executor.remove_node(self.node)
        self.node.destroy_node()
        self.executor.shutdown()


def sdk_factory():
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
    return RoboticArm(rm_thread_mode_e.RM_DUAL_MODE_E)


def telemetry_factory(monitor,source):
    from ros_controller_telemetry import ROSControllerTelemetry
    return ROSControllerTelemetry(monitor,source)


def checked_query(result, name):
    if not isinstance(result, (tuple, list)) or len(result) != 2 or result[0] != 0:
        raise MotionError(name+' failed: '+repr(result))
    return result[1]


def controller_state(value):
    if not isinstance(value, dict) or 'err' not in value:
        raise MotionError('Controller alarm telemetry is missing')
    err = value['err']
    codes = err.get('err') if isinstance(err, dict) else err
    if not isinstance(codes, (tuple, list)):
        codes = [codes]
    if any(str(v) not in ('0', '0x0', '0x0000', '') for v in codes):
        raise MotionError('Controller alarm: '+repr(err))
    return vector(value.get('joint'))


def in_bounds(joints, limits):
    if len(limits) != 6:
        raise MotionError('Six validated joint limits are required')
    for i, (q, bounds) in enumerate(zip(vector(joints), limits)):
        lo, hi = bounds
        if not math.isfinite(lo+hi) or lo >= hi or not lo <= q <= hi:
            raise MotionError('Joint '+str(i+1)+' exceeds validated range')


def smooth_and_resample(joints, grippers, t_rels, dt=0.010, smooth_window=11):
    """Uniform 10 ms resample and moving average, as in old_code/drag_teach_utils.py."""
    n=len(joints)
    if n<2:
        return joints, grippers, t_rels
    n_joints=len(joints[0])
    new_t=[]
    t=t_rels[0]
    while t<=t_rels[-1]+1e-9:
        new_t.append(t)
        t+=dt
    resampled=[];new_grippers=[]
    for nt in new_t:
        pos=bisect.bisect_right(t_rels, nt)
        i0=max(0, pos-1);i1=min(pos, n-1)
        span=t_rels[i1]-t_rels[i0]
        alpha=max(0.0, min(1.0, (nt-t_rels[i0])/span if span>1e-9 else 0.0))
        resampled.append([joints[i0][j]+(joints[i1][j]-joints[i0][j])*alpha for j in range(n_joints)])
        new_grippers.append(grippers[i0])
    half=smooth_window//2
    n2=len(resampled)
    smoothed=[]
    for i in range(n2):
        lo,hi=max(0,i-half),min(n2,i+half+1)
        count=hi-lo
        smoothed.append([sum(resampled[k][j] for k in range(lo,hi))/count for j in range(n_joints)])
    return smoothed, new_grippers, [i*dt for i in range(n2)]


def pre_roll_seconds(value):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=10:
        raise MotionError('移动前保留时间必须是0至10秒')
    return float(value)


class TeachingWindow:
    """Keep a bounded pre-roll until real joint displacement crosses 0.3 degree."""
    def __init__(self,pre_roll_s):
        self.pre_roll_s=pre_roll_seconds(pre_roll_s)
        self.waiting=deque();self.samples=[];self.moved=False;self.anchor=None
        self.origin=None;self.detected_at=None;self.trimmed_s=0
    def add(self,t,q):
        q=vector(q)
        if self.anchor is None:self.anchor=list(q)
        if not self.moved:
            self.waiting.append((t,q))
            # At most one sample period beyond the requested pre-roll.
            while len(self.waiting)>1 and self.waiting[1][0]<=t-self.pre_roll_s:
                self.waiting.popleft()
            if maxdiff(q,self.anchor)<.3:return
            self.moved=True;self.detected_at=t
            self.origin=self.waiting[0][0];self.trimmed_s=self.origin
            self.samples=[dict(t=a-self.origin,joints_deg=b) for a,b in self.waiting]
            self.waiting.clear()
        else:
            self.samples.append(dict(t=t-self.origin,joints_deg=q))
        if self.samples[-1]['t']>MAX_TEACH_S:
            raise MotionError('Teaching reached 140 second maximum after trimming')
    def trim_tail(self, post_roll_s=1.0):
        """Keep one second of stillness after the last real motion; drop a longer hold."""
        post_roll_s=pre_roll_seconds(post_roll_s)
        if len(self.samples)<2:
            return
        final=self.samples[-1]['joints_deg']
        last_move=0
        for i,frame in enumerate(self.samples):
            if maxdiff(frame['joints_deg'], final)>=.3:
                last_move=i
        keep_until=self.samples[last_move]['t']+post_roll_s
        kept=[frame for frame in self.samples if frame['t']<=keep_until+1e-9]
        if len(kept)>=2:
            self.trimmed_trailing_s=self.samples[-1]['t']-kept[-1]['t']
            self.samples=kept


class ManualMotion:
    def __init__(self, source, storage, *, sdk_factory=None, monitor=None,
                 on_replay_start=None, on_replay_finish=None, external_guard=None, telemetry_factory=None):
        self.source = source
        self.storage = Path(storage)/'manual_motion'
        self.storage.mkdir(parents=True, exist_ok=True)
        self.factory = sdk_factory or globals()['sdk_factory']
        self.telemetry_factory = telemetry_factory or (self.factory if sdk_factory else lambda: globals()['telemetry_factory'](self.monitor,self.source))
        self.telemetry_sdk = None
        self.telemetry_lock = threading.Lock()
        self.monitor = monitor
        self.monitor_error = None
        if monitor is None:
            try:
                self.monitor = ROSMonitor()
            except Exception as exc:
                self.monitor_error = str(exc)
        self.on_replay_start = on_replay_start
        self.on_replay_finish = on_replay_finish
        self.external_guard = external_guard
        self.lock = threading.RLock()
        self.io_lock = threading.Lock()
        self.cancel = threading.Event()
        self.shutdown = threading.Event()
        self.sdk = None
        self.mode = 'DISARMED'
        self.owner = None
        self.deadline = 0
        self.error = None
        self.reason = '现场确认释放其它控制输出并停稳底盘后，才能启用手动会话'
        self.progress = 0.0
        self.playback_timing={'deferred_cycles':0,'max_lateness_s':0.0,'max_sdk_call_s':0.0}
        self.trajectory = None
        self.trajectory_load_error=None
        pointer=self.storage/'latest_teaching.json'
        if pointer.exists():
            try:
                name=json.loads(pointer.read_text())['file']
                if Path(name).name!=name or not name.endswith('.json'):raise ValueError('Invalid saved trajectory name')
                saved=json.loads((self.storage/name).read_text())
                if saved.get('unit')!='degree' or not saved.get('samples'):raise ValueError('Empty saved trajectory')
                for frame in saved['samples']:
                    vector(frame['joints_deg'])
                    if not math.isfinite(float(frame['t'])):raise ValueError('Bad timestamp')
                saved['restored']=True;saved['replay_ready']=False
                saved['replay_reason']='启用会话后按当前限位与参数检查回放'
                self.trajectory=saved
            except Exception as exc:self.trajectory_load_error=str(exc)
        self.samples = []
        self.episode_home = None
        self.teach_pre_roll_s=1.0
        self.teach_settings_file=self.storage/'teach_settings.json'
        if self.teach_settings_file.exists():
            try:self.teach_pre_roll_s=pre_roll_seconds(json.loads(self.teach_settings_file.read_text())['pre_roll_s'])
            except (ValueError,KeyError,TypeError):pass
        self.teaching_window=None
        self.limits = None
        self.native_speed_limits = None
        self.urdf_limits = None
        self.baseline = None
        self.drag_maybe_active = False
        self.worker = self.stop_thread = None
        self.opening = False
        self.planning = False
        self.operation_generation = 0
        self.start_pending = False
        self.fault_latched = False
        self.primary_fault = None
        self.stop_confirmed = False
        self.recovering = False
        self.recovery_error = None
        self.stop_started = None
        self.controller_checked = 0.0
        self.command = None
        self.command_history=deque(maxlen=500)
        self.tracking_status=None
        self.tracking_violation=None
        self.parameters=dict(DEFAULT_PARAMETERS)
        self.parameter_error=None
        self.parameters_file=self.storage/'parameters.json'
        if self.parameters_file.exists():
            try:self.parameters=validate_parameters(json.loads(self.parameters_file.read_text()))
            except Exception as exc:self.parameter_error='参数文件无效，请在未开启会话时重新保存：'+str(exc)
        self.reset_pose = vector(getattr(source,'cfg',{}).get('manual_reset_pose_deg',[0,0,0,0,90,0]))
        self.reset_reason = '启用会话后检查控制器限位'
        self.reset_ready = False
        self.reset_target_source = 'user_selected_standby'
        self.reset_load_error = None
        self.reset_file = self.storage/'reset_pose.json'
        if self.reset_file.exists():
            try:
                saved=json.loads(self.reset_file.read_text())
                if saved.get('unit')!='degree' or saved.get('source') not in ('user_current_pose','user_manual_pose'):
                    raise MotionError('Invalid saved reset pose contract')
                self.reset_pose=vector(saved['joints_deg'])
                self.reset_target_source=saved['source']
            except Exception as exc:
                self.reset_load_error='已存零位读取失败，请重新设置：'+str(exc)
                self.reset_reason=self.reset_load_error
        self.replay_recording = False
        self.teach_complete = False
        self.disarm_pending = False
        self.watch = threading.Thread(target=self._watch, daemon=True)
        self.watch.start()
        self.alarm_watch = threading.Thread(target=self._alarm_watch, daemon=True)
        self.alarm_watch.start()

    def _feedback(self):
        if hasattr(self.source,'joint_feedback'):
            q,stamp=self.source.joint_feedback()
            return vector(q),int(stamp)
        obs, timing = self.source.snapshot(decode=False)
        q = vector(obs['observation.state'][:6])
        receipt = timing['receive_mono_ns']['joints']/1e9
        if not 0 <= time.monotonic()-receipt <= FRESH_S:
            raise MotionError('Joint feedback older than 150 ms')
        return q, int(timing['source_ns']['joints'])

    def _monitor(self, baseline=True, check_fresh=True, check_force=True):
        if self.monitor is None:
            raise MotionError('Manual guard unavailable: '+str(self.monitor_error))
        m = self.monitor.snapshot()
        if check_fresh and any(not isinstance(m[k], (float, int)) or not math.isfinite(m[k]) or not 0 <= m[k] <= FRESH_S for k in ('control_age_s', 'force_age_s')):
            raise MotionError('Control or wrench feedback older than 150 ms')
        if m['mode'] != 'idle':
            raise MotionError('Other controller is not idle')
        if baseline and self.baseline is not None and (m['dispatch_seq'], m['generation']) != self.baseline:
            raise MotionError(m.get('change_reason') or 'Other controller dispatched or changed mode; control ownership lost')
        force=math.sqrt(sum(x*x for x in vector(m['force'],3)))
        torque=math.sqrt(sum(x*x for x in vector(m['torque'],3)))
        # Drag, MoveJ and high-follow replay use the old collector: contact force is expected.
        # Idle sessions and recovery still use the configured thresholds.
        if check_force and self.mode not in PASSIVE_MODES:
            if force>self.parameters['force_n']:
                raise MotionError('Force exceeds '+format(self.parameters['force_n'],'g')+' N')
            if torque>self.parameters['torque_nm']:
                raise MotionError('Torque exceeds '+format(self.parameters['torque_nm'],'g')+' Nm')

        return m

    def _ticket(self, ticket):
        if ticket != self.operation_generation or self.cancel.is_set():
            raise MotionError('Operation generation cancelled')

    def _guard(self, ticket=None):
        if self.external_guard:self.external_guard()
        if ticket is not None:
            self._ticket(ticket)
        if self.start_pending or self.mode not in PASSIVE_MODES:
            self._monitor(check_force=not self.start_pending)
            q, stamp = self._feedback()
            if not self.start_pending:
                in_bounds(q, self.limits)
                if self.command is not None:self._check_tracking(q,stamp)
        else:
            # Old collector keeps sampling through brief feedback gaps instead of stopping the arm.
            try:
                q, stamp = self._feedback()
            except Exception:
                q, stamp = (list(self.command) if self.command is not None else [0.0]*6), 0
        if self.cancel.is_set():
            raise MotionError('Operation cancelled')
        if time.monotonic() >= self.deadline:
            raise MotionError('5 second owner lease expired')
        return q, stamp

    def _command_clock_ns(self):
        if hasattr(self.source,'node'):return self.source.node.get_clock().now().nanoseconds
        return time.time_ns()

    def _remember_command(self,target,stamp):
        with self.lock:
            if self.command_history and stamp<self.command_history[-1][0]:raise MotionError('Command source clock moved backwards')
            self.command_history.append((int(stamp),vector(target)))

    def _check_tracking(self,q,feedback_stamp):
        with self.lock:
            latest=vector(self.command)
            reference=latest;command_stamp=None
            for stamp,target in reversed(self.command_history):
                if stamp<=feedback_stamp:reference=target;command_stamp=stamp;break
            errors=[abs(a-b) for a,b in zip(q,reference)]
            axis=max(range(6),key=lambda i:errors[i]);limit=self.parameters['tracking_deg']
            self.tracking_status=dict(joint=axis+1,error_deg=errors[axis],limit_deg=limit,
                reference='feedback_timestamp',raw_latest_error_deg=maxdiff(q,latest),
                actual_deg=list(q),reference_deg=list(reference),latest_sent_deg=latest,
                feedback_source_ns=int(feedback_stamp),command_source_ns=command_stamp,
                feedback_age_s=max(0,(self._command_clock_ns()-feedback_stamp)/1e9))
            if errors[axis]>limit:
                self.tracking_violation=dict(self.tracking_status)
                raise MotionError('J'+str(axis+1)+' timestamp-aligned tracking error '+format(errors[axis],'.2f')+' exceeds '+format(limit,'g')+' degrees')

    def _sdk(self, method, *args, stopping=False, query=False, ticket=None, **kwargs):
        # No lock spans sleeps or sampling. Stop sets cancel before waiting here;
        # regular calls recheck it under the same SDK serialization lock.
        if not self.io_lock.acquire(timeout=.15):
            raise MotionError('SDK call is blocked; stop cannot yet be confirmed')
        try:
            if self.cancel.is_set() and not stopping:
                raise MotionError('Operation cancelled before SDK call')
            if ticket is not None:
                self._ticket(ticket)
            if self.sdk is None:
                raise MotionError('No SDK session')
            result = getattr(self.sdk, method)(*args, **kwargs)
            if query:
                return checked_query(result, method)
            if result != 0:
                raise MotionError(method+' failed: '+repr(result))
            return result
        finally:
            self.io_lock.release()

    def _check_controller(self,validate_range=True):
        # Existing rm_driver UDP feedback: no additional TCP query connection.
        if not self.telemetry_lock.acquire(timeout=.075):
            raise MotionError('Controller telemetry query is blocked')
        try:
            if self.telemetry_sdk is None:raise MotionError('Controller telemetry connection missing')
            q=controller_state(checked_query(self.telemetry_sdk.rm_get_current_arm_state(),'controller telemetry'))
            if validate_range:in_bounds(q,self.limits)
            self.controller_checked=time.monotonic()
            return q
        finally:
            self.telemetry_lock.release()

    def _owned(self, client_id, expected='ARMED'):
        with self.lock:
            if self.fault_latched:
                raise MotionError('FAULT is latched; on-site inspection and service restart required')
            if not isinstance(client_id, str) or not client_id or client_id != self.owner:
                raise MotionError('This browser does not own the manual session')
            if self.mode != expected or self.opening or self.planning:
                raise MotionError('Operation requires '+expected+', current '+self.mode)
            if time.monotonic() >= self.deadline:
                raise MotionError('Owner lease expired')

    def status(self):
        with self.lock:
            trajectory = self.trajectory
            out = dict(enabled=self.sdk is not None and self.mode not in ('DISARMED', 'FAULT'),
                       mode=self.mode, ready=False, reason=self.reason, owner=self.owner,session_generation=self.operation_generation,teach_contact_mode='native_drag',playback_mode='recorded_timing',native_joint_speed_deg_s=self.native_speed_limits,
                       lease_remaining_s=max(0, self.deadline-time.monotonic()), error=self.error,
                       progress=self.progress,tracking=self.tracking_status, playback_timing=dict(self.playback_timing),trajectory=None, validated_on_hardware=False,planning=self.planning,
                       default_initial_pose=list(self.reset_pose), reset_target_source=self.reset_target_source, joint_limits_deg=self.limits,joint_limit_source='controller',urdf_joint_limits_deg=self.urdf_limits,
                       reset_speed_percent=self.parameters['reset_speed_percent'],parameters=dict(self.parameters),parameter_bounds=PARAMETER_BOUNDS,parameter_error=self.parameter_error, reset_ready=self.reset_ready,reset_reason=self.reset_reason,
                       teach_pre_roll_s=self.teach_pre_roll_s,
                       teaching_movement_detected=bool(self.teaching_window and self.teaching_window.moved),
                       teaching_kept_frames=len(self.samples) if self.teaching_window and self.teaching_window.moved else len(self.teaching_window.waiting) if self.teaching_window else 0)
            recovery_available=(self.mode=='FAULT' and self.stop_confirmed and not self.recovering
                                and not self.start_pending and self.telemetry_sdk is not None and self.sdk is not None
                                and not (self.worker and self.worker.is_alive())
                                and not (self.stop_thread and self.stop_thread.is_alive()))
            out.update(stop_confirmed=self.stop_confirmed,recovering=self.recovering,
                       recovery_available=bool(recovery_available),
                       recovery_reason=self.recovery_error or ('正在核对停稳、受力和报警' if self.recovering
                         else '排除接触或受力原因后，点击检查并恢复；不会自动运动' if recovery_available
                         else '停止尚未确认，请先点击停止机械臂并等待检查完成' if self.mode=='FAULT' else None))
            if trajectory:
                samples = trajectory['samples']
                out['trajectory'] = dict(id=trajectory['id'], frames=len(samples), duration_s=samples[-1]['t'],
                                         start=samples[0]['joints_deg'], end=samples[-1]['joints_deg'],raw_saved=True,
                                         replay_ready=trajectory.get('replay_ready',True),replay_reason=trajectory.get('replay_reason'),
                                         planned_duration_s=trajectory.get('planned_duration_s'),retimed=False,restored=bool(trajectory.get('restored')))
        try:
            if self.external_guard:self.external_guard()
            if hasattr(self.monitor,'controller_snapshot'):self.monitor.controller_snapshot()
            m = self._monitor(baseline=self.mode != 'DISARMED')
            q, _ = self._feedback()
            if self.limits is not None:
                in_bounds(q, self.limits)
            out.update(force_n=math.sqrt(sum(x*x for x in vector(m['force'], 3))),
                       torque_nm=math.sqrt(sum(x*x for x in vector(m['torque'], 3))),
                       feedback_age_s=max(m['control_age_s'], m['force_age_s']), joints_deg=q)
            out['ready'] = self.mode in ('DISARMED', 'ARMED') and not self.fault_latched and not self.opening and not self.planning and m['idle_s'] >= .5 and not self.parameter_error
        except Exception as exc:
            out['reason'] = str(exc)
        return out

    def arm(self, client_id, confirmed):
        if self.parameter_error:raise MotionError(self.parameter_error)
        if confirmed is not True or not isinstance(client_id, str) or not 1 <= len(client_id) <= 128:
            raise MotionError('Explicit confirmation and browser client ID required')
        with self.lock:
            if self.mode != 'DISARMED' or self.opening or self.fault_latched or self.shutdown.is_set():
                raise MotionError('Manual session cannot be armed in current state')
            self.opening = True
            self.cancel.clear()
            self.owner = client_id
            self.deadline = time.monotonic()+LEASE_S
        try:
            if self.external_guard:self.external_guard()
            m = self._monitor(baseline=False)
            if m['idle_s'] < .5:
                raise MotionError('Other control must remain idle with unchanged dispatch sequence for 0.5 s')
            # URDF bounds describe the visualization/planning model; the controller's
            # actual configured limits govern real hardware, including native hand-guiding.
            self.urdf_limits=[tuple(vector(pair,2)) for pair in self.monitor.joint_limits()]
            q,_=self._feedback()
            self.baseline=(m['dispatch_seq'],m['generation'])
            if self.sdk is None:
                # Constructors and connection establishment are read/session only.
                with self.io_lock:
                    if self.cancel.is_set():
                        raise MotionError('Arm request cancelled')
                    self.sdk = self.factory()
                    handle = self.sdk.rm_create_robot_arm('192.168.1.18', 8080)
                    if not hasattr(handle, 'id') or handle.id <= 0:
                        raise MotionError('SDK connection failed')
            if getattr(self.sdk, 'arm_dof', None) != 6:
                raise MotionError('SDK did not verify a six-axis robot; refusing motion')
            lower = self._sdk('rm_get_joint_min_pos', query=True)
            upper = self._sdk('rm_get_joint_max_pos', query=True)
            lower, upper = vector(lower), vector(upper)
            self.limits = list(zip(lower,upper))
            in_bounds(q, self.limits)
            self.native_speed_limits=vector(self._sdk('rm_get_joint_max_speed',query=True))
            if any(v<=0 for v in self.native_speed_limits):raise MotionError('Invalid controller joint speed limits')
            if self.telemetry_sdk is None:
                with self.telemetry_lock:
                    self.telemetry_sdk=self.telemetry_factory()
                    handle=self.telemetry_sdk.rm_create_robot_arm('192.168.1.18',8080)
                    if not hasattr(handle,'id') or handle.id<=0:
                        raise MotionError('Read-only telemetry connection failed')
            self._check_controller()
            try:
                if self.reset_load_error:raise MotionError(self.reset_load_error)
                in_bounds(self.reset_pose,self.limits)
                self.reset_ready=True;self.reset_reason=None
            except MotionError as exc:
                self.reset_ready=False;self.reset_reason=str(exc)
            self._guard()
            self._assess_trajectory()
            with self.lock:
                if self.cancel.is_set():
                    raise MotionError('Arm request cancelled')
                self.operation_generation+=1
                self.mode = 'ARMED'
                self.error = None;self.primary_fault=None
                self.reason = '会话已启用；未发送运动指令。拖动、MoveJ 返回起点和回放需分别点击。'
            return self.status()
        except Exception as exc:
            # No motion is requested by arm(), but a connection/guard failure is
            # visible and never becomes a ready session.
            with self.lock:
                self.error = self.reason = str(exc)
                self.owner = None
                self.deadline = 0
                if self.mode != 'STOPPING':
                    self.mode = 'FAULT' if self.fault_latched else 'DISARMED'
            raise MotionError(str(exc)) from exc
        finally:
            self.opening = False

    def heartbeat(self, client_id):
        with self.lock:
            if client_id != self.owner or self.mode in ('DISARMED', 'FAULT', 'STOPPING') or time.monotonic() >= self.deadline:
                raise MotionError('Manual lease cannot be renewed')
            self.deadline = time.monotonic()+LEASE_S
        return self.status()

    def drag_start(self, client_id, pre_roll_s=None):
        self._owned(client_id)
        pre_roll=self.teach_pre_roll_s if pre_roll_s is None else pre_roll_seconds(pre_roll_s)
        q,_=self._guard()
        self._check_controller()
        with self.lock:
            self._owned(client_id)
            self.error = None
            tmp=self.teach_settings_file.with_suffix('.tmp')
            with tmp.open('w') as f:
                json.dump({'pre_roll_s':pre_roll},f);f.flush();os.fsync(f.fileno())
            os.replace(tmp,self.teach_settings_file)
            self.teach_pre_roll_s=pre_roll
            self.teaching_window=TeachingWindow(pre_roll)
            self.mode = 'TEACHING'
            self.operation_generation += 1
            ticket = self.operation_generation
            self.start_pending = True
            self.samples = []
            self.episode_home = list(q)
            self.teach_complete = False
            self.command = None
            self.progress = 0
        try:
            # old_code default: current-loop drag, sensitivity 100, singular wall on.
            if DRAG_MODE == 0:
                self._sdk('rm_set_drag_teach_sensitivity', DRAG_SENSITIVITY, ticket=ticket)
            self.drag_maybe_active = True
            self._sdk('rm_start_multi_drag_teach', mode=DRAG_MODE, singular_wall=1, ticket=ticket)
            with self.lock:
                self._ticket(ticket)
                self.worker = threading.Thread(target=self._teach, args=(ticket,), daemon=True)
                self.worker.start()
        except Exception as exc:
            # A watchdog may have cancelled the start. Preserve its original failure.
            if ticket==self.operation_generation and not self.cancel.is_set():
                self._request_stop(str(exc), fault=True)
            raise MotionError(self.error or self.reason or str(exc)) from exc
        finally:
            self.start_pending = False
        return self.status()

    def _teach(self, ticket):
        due = time.monotonic()
        first = previous = None
        try:
            while not self.cancel.is_set():
                now = time.monotonic()
                if due > now:
                    self.cancel.wait(due-now)
                    continue
                if now-due > LATE_S:
                    due = now
                try:
                    q, stamp = self._guard(ticket)
                except MotionError as exc:
                    if self.cancel.is_set() or 'lease' in str(exc):
                        raise
                    time.sleep(.002)
                    continue
                if stamp != previous and stamp:
                    if previous is not None and stamp < previous:
                        due = max(due+1/TEACH_HZ, time.monotonic())
                        continue
                    first = stamp if first is None else first
                    t = (stamp-first)/1e9
                    with self.lock:
                        self._ticket(ticket)
                        self.teaching_window.add(t,q)
                        self.samples=self.teaching_window.samples
                    previous = stamp
                due = max(due+1/TEACH_HZ, time.monotonic())
        except Exception as exc:
            if (ticket == self.operation_generation and not self.cancel.is_set()) or self.fault_latched:
                self._request_stop(str(exc), fault=True)

    def drag_stop(self, client_id):
        """Stop drag and save the path. Returning home is a separate MoveJ button."""
        self._owned(client_id, 'TEACHING')
        self.teach_complete = True
        self._request_stop('正在退出拖动并保存路径', save_teaching=True)
        return self.status()

    def set_reset_pose(self, client_id, source, joints_deg=None):
        """Save a software reset target; never calibrates or commands the robot."""
        self._owned(client_id)
        if source not in ('current','manual'):raise MotionError('请选择当前位置或手动输入')
        self._guard()
        self._check_controller()
        stationary=self._stable()
        if source=='manual':
            if not isinstance(joints_deg,list) or len(joints_deg)!=6 or any(type(v) not in (int,float) for v in joints_deg):
                raise MotionError('请输入六个有限关节角度（degree）')
            target=vector(joints_deg)
        else:
            target=stationary
        in_bounds(target,self.limits)
        with self.lock:
            self._owned(client_id)
            self._guard()
            record=dict(unit='degree',joints_deg=target,source='user_'+source+'_pose',
                        saved_unix_s=time.time(),previous_joints_deg=self.reset_pose,
                        joint_limits_deg=self.limits)
            temporary=self.reset_file.with_name('reset_pose.'+uuid.uuid4().hex+'.tmp')
            try:
                with temporary.open('x') as f:
                    json.dump(record,f,allow_nan=False);f.flush();os.fsync(f.fileno())
                os.replace(temporary,self.reset_file)
            finally:
                temporary.unlink(missing_ok=True)
            self.reset_pose=list(target)
            self.reset_target_source=record['source']
            self.reset_ready=True;self.reset_reason=None;self.reset_load_error=None
        return self.status()

    def reset(self,client_id):
        if self.reset_load_error:raise MotionError(self.reset_load_error)
        return self._movej_to(client_id,list(self.reset_pose),'reset','user_reset_pose')

    def _movej_to(self,client_id,target,kind,source,speed=None):
        parameters=self.parameters
        speed=parameters['reset_speed_percent'] if speed is None else int(speed)
        self._owned(client_id);self._guard();target=vector(target)
        in_bounds(target,self.limits);self._check_controller()
        audit=dict(kind=kind,target_deg=target,speed_percent=speed,source=source,created_unix_s=time.time())
        with (self.storage/(uuid.uuid4().hex+'.movej.json')).open('x') as f:
            json.dump(audit,f);f.flush();os.fsync(f.fileno())
        with self.lock:
            self._owned(client_id)
            if parameters is not self.parameters:raise MotionError('操作参数已改变，请重新点击')
            if kind=='reset' and (target!=self.reset_pose or self.reset_load_error):raise MotionError('零位已更新，请重新点击')
            self.mode='RESETTING' if kind=='reset' else 'HOMING';self.error=None;self.command=None;self.progress=0
            self.operation_generation+=1;ticket=self.operation_generation
            self.worker=threading.Thread(target=self._reset,args=(target,ticket,speed),daemon=True)
            self.worker.start()
        return self.status()

    def _reset(self,target,ticket,speed=None):
        try:
            start,_=self._guard(ticket)
            distance=maxdiff(start,target)
            speed=self.parameters['reset_speed_percent'] if speed is None else int(speed)
            # Same non-blocking MoveJ as old_code move_to_home. No streamed reverse path.
            self._sdk('rm_movej',target,v=speed,r=0,connect=0,block=0,ticket=ticket)
            end=time.monotonic()+60
            while not self.cancel.wait(.05):
                q,_=self._guard(ticket)
                error=maxdiff(q,target)
                self.progress=max(0,min(1,1-error/max(distance,.5)))
                if error<1.5:
                    self.progress=1
                    self._request_stop('MoveJ目标已到达，正在确认停稳')
                    return
                if time.monotonic()>end:raise MotionError('Reset did not reach target within 60 seconds')
        except Exception as exc:
            if (ticket==self.operation_generation and not self.cancel.is_set()) or self.fault_latched:
                self._request_stop(str(exc),fault=True)

    def home(self, client_id):
        if not self.trajectory or not self.trajectory.get('samples'):raise MotionError('没有已保存的示教起点')
        target=self.trajectory.get('home_joints_deg') or self.trajectory['samples'][0]['joints_deg']
        return self._movej_to(client_id,target,'home_movej',self.trajectory['id'],speed=LEGACY_HOME_SPEED)

    def replay(self, client_id):
        self._owned(client_id)
        trajectory=self.trajectory
        if not trajectory or not trajectory.get('samples'):
            raise MotionError('没有可回放的示教路径')
        if not trajectory.get('complete'):
            raise MotionError('原始路径已保存，但示教未完整结束，不能回放')
        samples=trajectory['samples']
        if len(samples)<2:
            raise MotionError('回放需要至少两帧')
        joints=[vector(frame['joints_deg']) for frame in samples]
        times=[float(frame['t']) for frame in samples]
        points,_,t_rel=smooth_and_resample(joints,[0]*len(joints),times,dt=REPLAY_DT,smooth_window=REPLAY_SMOOTH)
        speed=float(self.parameters['replay_speed'])
        with self.lock:
            self._owned(client_id)
            self.cancel.clear()
            self.error=None
            self.mode='REPLAYING'
            self.operation_generation+=1
            ticket=self.operation_generation
            self.command=list(points[0])
            self.progress=0
            self.reason='按原始时间高跟随回放'
            self.worker=threading.Thread(target=self._play,args=(points,t_rel,speed,True,ticket),daemon=True)
            self.worker.start()
        return self.status()

    def _play(self, points, times_or_recording=False, speed_or_ticket=None, recording=None, ticket=None):
        """High-follow CANFD replay paced by the taught timestamps.

        Existing tests call _play(points, recording, ticket). The panel calls
        _play(points, times, speed, recording, ticket).
        """
        if ticket is None:
            recording=bool(times_or_recording)
            ticket=speed_or_ticket
            times=[i*REPLAY_DT for i in range(len(points))]
            speed=1.0
        else:
            times=list(times_or_recording)
            speed=float(speed_or_ticket)
        completed=False
        try:
            self._ticket(ticket)
            if recording:
                if self.on_replay_start is None or self.on_replay_finish is None:
                    raise MotionError('Passive replay recording callbacks are not configured')
                episode_id=self.on_replay_start()
                if not isinstance(episode_id,str) or not episode_id:
                    raise MotionError('Replay recorder did not return a new episode ID')
                self.replay_recording=True
            self.playback_timing={'deferred_cycles':0,'max_lateness_s':0.0,'max_sdk_call_s':0.0}
            started=time.perf_counter()
            for i,target in enumerate(points):
                if self.cancel.is_set():
                    return
                self._ticket(ticket)
                sent_at=time.perf_counter()
                self._sdk('rm_movej_canfd', list(target), follow=True, ticket=ticket)
                elapsed=time.perf_counter()-sent_at
                self.playback_timing['max_sdk_call_s']=max(self.playback_timing['max_sdk_call_s'],elapsed)
                with self.lock:
                    self._ticket(ticket)
                    self.command=list(target)
                    self.progress=i/max(1,len(points)-1)
                target_t=started+times[i]/speed
                gap=target_t-time.perf_counter()
                self.playback_timing['max_lateness_s']=max(self.playback_timing['max_lateness_s'],max(0,-gap))
                if gap>0.002 and self.cancel.wait(gap-0.001):
                    return
                while time.perf_counter()<target_t:
                    if self.cancel.is_set():
                        return
            if not self.cancel.is_set() and points:
                self._sdk('rm_movej_canfd', list(points[-1]), follow=True, ticket=ticket)
            completed=True
            self._request_stop('轨迹发送完成，正在确认停稳', replay_success=True)
        except Exception as exc:
            if (ticket==self.operation_generation and not self.cancel.is_set()) or self.fault_latched:
                self._request_stop(str(exc), fault=True)
        finally:
            if not completed and self.replay_recording:
                self._finish_recording(False, self.error or 'Replay cancelled')

    def _finish_recording(self, success, error):
        with self.lock:
            if not self.replay_recording:
                return
            self.replay_recording = False
        try:
            self.on_replay_finish(success, error)
        except Exception as exc:
            self._fault('Replay recorder finalization failed: '+str(exc))

    def _alarm_watch(self):
        # Potentially blocking native calls MUST NOT run on the lease watchdog.
        while not self.shutdown.wait(.01):
            if self.mode in ('ARMED','TEACHING','HOMING','RESETTING','REPLAYING') and not self.cancel.is_set():
                try:
                    if time.monotonic()-self.controller_checked >= .02:
                        self._check_controller()
                except Exception as exc:
                    message=str(exc)
                    if self.mode in PASSIVE_MODES and not self.start_pending and ('older than' in message or 'blocked' in message or 'Waiting for' in message):
                        continue
                    if not self.cancel.is_set() or self.fault_latched:
                        self._request_stop(message, fault=True)

    def _watch(self):
        while not self.shutdown.wait(.02):
            try:
                with self.lock:
                    active = self.mode in ('ARMED', 'TEACHING', 'HOMING', 'RESETTING', 'REPLAYING')
                    stopping = self.mode == 'STOPPING'
                    stop_started = self.stop_started
                if stopping and stop_started and time.monotonic()-stop_started > STOP_TIMEOUT:
                    self._fault('Stop timed out; stationary state NOT confirmed. Use on-site hardware stop if needed.')
                if self.opening and time.monotonic() >= self.deadline:
                    self._request_stop('SDK session opening exceeded owner lease', fault=True, disarm=True)
                if active:
                    if time.monotonic() >= self.deadline:
                        self._request_stop('5 second owner lease expired', disarm=True)
                        continue
                    if self.mode in PASSIVE_MODES and not self.start_pending:
                        self._monitor(check_fresh=False, check_force=False)
                        continue
                    self._guard()
                    if time.monotonic()-self.controller_checked > FRESH_S:
                        raise MotionError('Controller alarm telemetry older than 150 ms')
            except Exception as exc:
                if not self.cancel.is_set() and self.mode in ('ARMED', 'TEACHING', 'HOMING', 'RESETTING', 'REPLAYING'):
                    self._request_stop(str(exc), fault=True)

    def _fault(self, reason):
        self.cancel.set()
        with self.lock:
            self.fault_latched = True
            self.mode = 'FAULT'
            if self.primary_fault is None:self.primary_fault=reason
            # Never recursively append a previously formatted message.
            self.error=self.reason=reason+('；最初触发：'+self.primary_fault if reason!=self.primary_fault else '')

    def _request_stop(self, reason, *, fault=False, disarm=False, save_teaching=False, replay_success=False):
        # Lease expiry can race the guard after the watchdog's deadline check.
        # In either ordering, stop normally and close ownership; never turn it into a fault.
        if reason=='5 second owner lease expired':fault=False;disarm=True
        teaching_was_active=self.mode=='TEACHING'
        save_teaching=save_teaching or teaching_was_active
        self.cancel.set()  # Cancel sender BEFORE acquiring any lock or invoking SDK.
        with self.lock:
            self.operation_generation += 1
            self.stop_confirmed = False
            if fault and not self.fault_latched:
                self.fault_latched = True
                self.primary_fault=reason
                self.error = reason
                print('manual_motion_fault',json.dumps(dict(reason=reason,mode=self.mode,
                      generation=self.operation_generation,start_pending=self.start_pending,
                      controller_age_s=time.monotonic()-self.controller_checked,tracking=self.tracking_violation or self.tracking_status),ensure_ascii=False),flush=True)
            self.disarm_pending = self.disarm_pending or disarm
            self.reason = reason
            if self.stop_thread and self.stop_thread.is_alive():
                return
            if self.sdk is None and not self.opening:
                self.mode = 'FAULT' if self.fault_latched else 'DISARMED'
                self.owner = None
                return
            self.mode = 'STOPPING'
            self.stop_started = time.monotonic()
            self.stop_thread = threading.Thread(target=self._stop_worker, args=(save_teaching, replay_success), daemon=True)
            self.stop_thread.start()

    def _stable(self, check=None):
        begin = time.monotonic()
        stable_begin = None
        anchor = previous_stamp = None
        while time.monotonic()-begin < 2:
            if check:check()
            q, stamp = self._feedback()
            # Stationarity is measured from fresh joints, independent of whether the
            # position is inside a commanded-motion envelope. Recovery still validates limits.
            if previous_stamp is not None and stamp <= previous_stamp:
                time.sleep(.01)
                continue
            previous_stamp = stamp
            if anchor is None or maxdiff(q, anchor) > STABLE_TOL:
                anchor = q
                stable_begin = time.monotonic()
            elif time.monotonic()-stable_begin >= STABLE_S:
                return q
            time.sleep(.01)
        raise MotionError('Fresh joints were not stable for 0.5 s; stop is NOT confirmed')

    def _assess_trajectory(self):
        if not self.trajectory:return
        try:
            samples=self.trajectory.get('samples') or []
            if not self.trajectory.get('complete') or len(samples)<2:
                raise MotionError('示教未完整结束或帧数不足，不能回放')
            times=[]
            for frame in samples:
                t=float(frame['t']);vector(frame['joints_deg'])
                if not math.isfinite(t) or (times and t<=times[-1]):
                    raise MotionError('示教时间戳未严格递增')
                times.append(t)
            duration=times[-1]/float(self.parameters['replay_speed'])
            self.trajectory.update(replay_ready=True,replay_reason=None,planned_duration_s=duration,retimed=False)
        except (ValueError,KeyError,TypeError,MotionError) as exc:
            self.trajectory.update(replay_ready=False,replay_reason=str(exc),planned_duration_s=None)

    def _save_teaching(self,complete=True):
        with self.lock:
            window=self.teaching_window
            if window and hasattr(window,'trim_tail'):
                window.trim_tail(1.0);self.samples=list(window.samples)
            frames=[dict(t=f['t'],joints_deg=list(f['joints_deg'])) for f in self.samples]
            moved=bool(window and window.moved)
            if not frames and window and window.waiting:
                first=window.waiting[0][0]
                frames=[dict(t=t-first,joints_deg=list(q)) for t,q in window.waiting]
        if not frames:raise MotionError('没有收到任何关节样本，无法保存')
        trajectory=dict(id=uuid.uuid4().hex,unit='degree',complete=bool(complete and moved),
                        sample_hz=TEACH_HZ,samples=frames,created_unix_s=time.time(),raw_saved=True,
                        home_joints_deg=list(self.episode_home) if self.episode_home else (list(frames[0]['joints_deg']) if frames else None),
                        pre_roll_s=self.teach_pre_roll_s,movement_threshold_deg=.3,
                        trimmed_leading_s=window.trimmed_s if window else 0,
                        detected_motion_t=window.detected_at-window.origin if moved else None,
                        capture_error=None if complete and moved else self.error or '示教未完整结束或未检测到移动')
        path=self.storage/(trajectory['id']+'.json');temporary=path.with_suffix('.tmp')
        with temporary.open('x') as f:
            json.dump(trajectory,f,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)  # Raw data is durable BEFORE replay feasibility is assessed.
        pointer=self.storage/'latest_teaching.json';temp=pointer.with_suffix('.tmp')
        with temp.open('w') as f:
            json.dump({'file':path.name},f);f.flush();os.fsync(f.fileno())
        os.replace(temp,pointer)
        self.trajectory=trajectory
        self._assess_trajectory()
        print('teaching_saved',json.dumps(dict(id=trajectory['id'],frames=len(frames),
              complete=trajectory['complete'],replay_ready=trajectory['replay_ready'],
              replay_reason=trajectory.get('replay_reason')),ensure_ascii=False),flush=True)

    def _stop_worker(self, save_teaching, replay_success):
        stop_error = None
        try:
            worker = self.worker
            if worker and worker is not threading.current_thread():
                worker.join(.3)
            # A cancelled drag-start may still be awaiting its native acknowledgement.
            # Wait boundedly for that call to leave, then stop; never let a late start
            # arrive after the stop command. The outer 3-second stop timeout still applies.
            pending_until=time.monotonic()+1.5
            while self.start_pending and time.monotonic()<pending_until:
                time.sleep(.01)
            errors = []
            if self.drag_maybe_active:
                try:
                    self._sdk('rm_stop_drag_teach', stopping=True)
                    self.drag_maybe_active = False
                except Exception as exc:
                    errors.append(str(exc))
            # Attempt slow stop even if exiting drag failed.
            try:
                self._sdk('rm_set_arm_slow_stop', stopping=True)
            except Exception as exc:
                errors.append(str(exc))
            if errors:
                raise MotionError('; '.join(errors))
            q = self._stable()
            controller_q = self._check_controller(validate_range=False)
            self.controller_checked = time.monotonic()
            if self.start_pending or (worker and worker.is_alive()):
                raise MotionError('Previous SDK/sender worker is still running; stop NOT confirmed')
            self.stop_confirmed = True
            if replay_success and self.command is not None:
                self.progress=1
        except Exception as exc:
            self.stop_confirmed = False
            stop_error = str(exc)
            self._fault('停止未确认：'+stop_error)
        finally:
            if save_teaching and (self.samples or (self.teaching_window and self.teaching_window.waiting)):
                try:
                    self._save_teaching(complete=bool(self.teach_complete and not self.fault_latched and not stop_error))
                except Exception as exc:
                    self.error=(self.error+'；' if self.error else '')+'原始示教保存失败：'+str(exc)
                    self.reason=self.error
            if self.replay_recording and not (self.worker and self.worker.is_alive()):
                self._finish_recording(bool(replay_success and not stop_error and not self.fault_latched),
                                       stop_error or self.error or (None if replay_success else 'Replay cancelled'))
            with self.lock:
                self.command = None;self.command_history.clear()
                if self.fault_latched:
                    self.mode = 'FAULT'
                elif self.disarm_pending or time.monotonic() >= self.deadline or self.shutdown.is_set():
                    self.mode = 'DISARMED'
                    self.owner = None
                    self.deadline = 0
                    self.reason = '会话关闭，SDK停止返回成功且新鲜关节已稳定至少0.5秒'
                else:
                    self.mode = 'ARMED'
                    self.cancel.clear()
                    if not self.error:
                        self.reason = 'SDK停止返回成功且新鲜关节已稳定至少0.5秒；后续动作需手动触发'
                self.disarm_pending = False

    def set_parameters(self,client_id,parameters):
        with self.lock:
            if self.mode not in ('DISARMED','ARMED') or self.opening or self.planning or self.recovering or self.shutdown.is_set():
                raise MotionError('仅在未开启会话或空闲会话时修改参数')
            if self.mode=='ARMED':self._owned(client_id)
            updated=validate_parameters(parameters,self.parameters)
            temp=self.parameters_file.with_name('parameters.'+uuid.uuid4().hex+'.tmp')
            try:
                with temp.open('x') as f:
                    json.dump(updated,f,allow_nan=False);f.flush();os.fsync(f.fileno())
                os.replace(temp,self.parameters_file)
            finally:temp.unlink(missing_ok=True)
            self.parameters=updated;self.parameter_error=None
            self._assess_trajectory()
        return self.status()

    def recover(self, client_id):
        """Clear only the panel latch after fresh checks; never enable/clear hardware or move."""
        if not isinstance(client_id,str) or not 1<=len(client_id)<=128:
            raise MotionError('Invalid browser client ID')
        with self.lock:
            if not self.status()['recovery_available'] or self.shutdown.is_set():
                raise MotionError('停止尚未确认或停止线程仍在运行，请先点击停止机械臂并等待')
            if self.owner not in (None,client_id) and time.monotonic()<self.deadline:
                raise MotionError('另一页面仍持有此会话，请使用原页面恢复')
            self.recovering=True;self.recovery_error=None
            generation=self.operation_generation
            previous_error=self.error
        try:
            initial=self._monitor(baseline=False)
            if initial['idle_s']<.5:raise MotionError('Other controller must be idle for 0.5 seconds')
            baseline=(initial['dispatch_seq'],initial['generation'])
            def check():
                if generation!=self.operation_generation or self.shutdown.is_set() or self.mode!='FAULT':
                    raise MotionError('恢复检查被新的停止请求取消')
                if self.external_guard:self.external_guard()
                m=self._monitor(baseline=False)
                if (m['dispatch_seq'],m['generation'])!=baseline:
                    raise MotionError('Control state changed during recovery')
                if time.monotonic()-self.controller_checked>=.075:self._check_controller()
            self._check_controller()
            self._stable(check=check)
            enabled=self._sdk('rm_get_joint_en_state',query=True,stopping=True)
            if len(enabled)!=6 or any(v!=1 for v in enabled):
                raise MotionError('关节未全部使能；请检查现场急停和控制器，面板不会自动上电使能')
            self._check_controller();check()
            with self.lock:
                check()
                if not self.stop_confirmed:raise MotionError('停止确认状态已改变')
                path=self.storage/(uuid.uuid4().hex+'.recovery.json')
                with path.open('x') as f:
                    json.dump(dict(recovered_unix_s=time.time(),previous_error=previous_error,
                                   client_id=client_id,motion_sent=False),f,ensure_ascii=False)
                    f.flush();os.fsync(f.fileno())
                self.baseline=baseline;self.owner=client_id;self.deadline=time.monotonic()+LEASE_S
                self.operation_generation+=1
                self.fault_latched=False;self.error=None;self.primary_fault=None;self.command=None
                self.samples=[];self.teaching_window=None;self.teach_complete=False
                self.cancel.clear();self.mode='ARMED'
                self.reason='会话已恢复，未启动运动；请手动重新开启拖动或复位'
            return self.status()
        except Exception as exc:
            self.recovery_error='不能恢复：'+str(exc)
            raise MotionError(self.recovery_error) from exc
        finally:
            self.recovering=False

    def stop(self):
        self._request_stop('操作者请求软件停止')
        return self.status()

    def disarm(self, client_id):
        with self.lock:
            if client_id != self.owner:
                raise MotionError('Only the owner can close this manual session; software stop is available to everyone')
        self._request_stop('关闭手动会话', disarm=True)
        return self.status()

    def close(self):
        self._request_stop('面板服务关闭', disarm=True)
        if self.stop_thread:
            self.stop_thread.join(STOP_TIMEOUT+.5)
            if self.stop_thread.is_alive():
                self._fault('Service close cannot confirm stop; SDK call may be blocked')
        self.shutdown.set()
        self.watch.join(.5)
        self.alarm_watch.join(.5)
        # Never disconnect a session while its stop call may still be in flight.
        if self.sdk is not None and not (self.stop_thread and self.stop_thread.is_alive()):
            try:
                self._sdk('rm_delete_robot_arm', stopping=True)
            except Exception as exc:
                self._fault('SDK session close failed: '+str(exc))
        if self.telemetry_sdk is not None:
            if self.telemetry_lock.acquire(timeout=.5):
                try:
                    result=self.telemetry_sdk.rm_delete_robot_arm()
                    if result!=0:self._fault('Telemetry connection close failed: '+str(result))
                    self.telemetry_sdk=None
                finally:self.telemetry_lock.release()
            else:self._fault('Telemetry query still running during close')
        if self.monitor:
            self.monitor.close()
        if self.fault_latched:
            raise MotionError(self.error or 'Manual controls ended in FAULT')
