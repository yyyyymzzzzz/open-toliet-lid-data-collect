"""Passive JODELL feedback guard; an explicit prepare_open() may request opening.

Construction/status/require_open never publish or call a service. Hardware is not
validated. /joint_states is deliberately unused: its finger cache is restamped.
The existing Modbus result lacks a header, request ID and address, so we require
one expected requester/responder, observe the fixed requests, and fail closed on
any ambiguous request/publisher. Receipt freshness is not a hardware timestamp.

No blind initialization is implemented. Existing initialization topics have no
correlated completion/abort contract; resetting rACT can home the fingers. An
unactivated, faulty, or unreadable device requires on-site initialization. The
explicit operation only requests the existing maximum-allowed-open service.

Injectable transport contract (no real transport is needed for tests):
  start(on_request, on_result, on_graph): install passive callbacks; no commands.
  on_request({address, device, type, num}); on_result({state, read_data}).
  on_graph(valid: bool, reason: str, ambiguous: bool=False).
  open(config, cancel_event, timeout_s): one explicit service request, returns
    {success: bool}; must honor cancellation/timeouts without retrying.
  close(): release only its own resources, never issue gripper commands.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import threading
import time
import uuid

CONFIG_PATH = Path('/mnt/Data/robot-stack/current/arm_control/share/arm_control/config/robot_config.yaml')
FEEDBACK_MAX_AGE_S = .25
OPEN_STABLE_S = .3
OPEN_TIMEOUT_S = 5.0
TARGET_RAW = 12
TOLERANCE_RAW = 2
REQUEST_TOPIC = '/rm_driver/read_modbus_rtu_input_registers_cmd'
RESULT_TOPIC = '/rm_driver/read_modbus_rtu_input_registers_result'
SERVICE = '/arm_control/gripper_control'
MEASUREMENT_NOTE = '夹爪设备寄存器报告；接收时间新鲜度，不是独立物理测量。请现场确认。'


def _config(value=None):
    if value is None:
        import yaml
        value = yaml.safe_load(CONFIG_PATH.read_text())
    if not isinstance(value, dict):
        raise ValueError('夹爪配置不是对象')
    g = value.get('gripper', value)
    if not isinstance(g, dict) or g.get('type') != 'realman_jodell':
        raise ValueError('夹爪配置必须为 realman_jodell')
    exact = {'open_limit': 12, 'close_limit': 255, 'tool_voltage': 3,
             'rs485_baud': 115200, 'modbus_slave_id': 9}
    for key, expected in exact.items():
        if type(g.get(key)) is not int or g[key] != expected:
            raise ValueError('夹爪配置 '+key+' 与已核实部署不符；不会自动更改范围或初始化')
    for key in ('default_speed', 'default_force'):
        if type(g.get(key)) is not int or not 0 < g[key] <= 200:
            raise ValueError('夹爪 '+key+' 无效或超过已核实的 200；不会提高速度/夹持力')
    return dict(g)


class ROSTransport:
    """Uses the already running ROS context. No SDK, init, reset or publisher."""
    def __init__(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from robot_interfaces.srv import GripperControl
        if not rclpy.ok():
            raise ValueError('需要已运行的观测 ROS context；夹爪模块不会自行启动机器人')
        self.rclpy = rclpy
        self.service_type = GripperControl
        self.node = rclpy.create_node('r5_gripper_guard_'+str(os.getpid())+'_'+uuid.uuid4().hex[:6])
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.client = self.node.create_client(GripperControl, SERVICE)
        self.dead = threading.Event()
        self.thread = None
        self.subscriptions = []

    def start(self, on_request, on_result, on_graph):
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from rm_ros_interfaces.msg import Modbusreaddata, Modbusrtureadparams
        # Keep a short queue to observe unexpected requests and failed replies,
        # while the independent receipt deadline refuses stale successes.
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.subscriptions = [
            self.node.create_subscription(Modbusrtureadparams, REQUEST_TOPIC,
                lambda m: on_request(dict(address=m.address, device=m.device, type=m.type, num=m.num)), qos),
            self.node.create_subscription(Modbusreaddata, RESULT_TOPIC,
                lambda m: on_result(dict(state=m.state, read_data=list(m.read_data))), qos)]

        def graph():
            try:
                requesters = [(p.node_name, p.node_namespace) for p in self.node.get_publishers_info_by_topic(REQUEST_TOPIC)]
                responders = [(p.node_name, p.node_namespace) for p in self.node.get_publishers_info_by_topic(RESULT_TOPIC)]
                valid = requesters == [('arm_control', '/')] and responders == [('rm_driver', '/')]
                ambiguous = (len(requesters) > 1 or len(responders) > 1 or
                    any(p != ('arm_control', '/') for p in requesters) or
                    any(p != ('rm_driver', '/') for p in responders))
                on_graph(valid, '' if valid else 'Modbus 发布者缺失或来源不唯一，不能关联夹爪反馈', ambiguous)
            except Exception as exc:
                on_graph(False, 'ROS graph 检查失败：'+str(exc), False)

        def spin():
            last_graph = 0
            try:
                while not self.dead.is_set():
                    if time.monotonic()-last_graph >= .1:
                        graph(); last_graph = time.monotonic()
                    self.executor.spin_once(timeout_sec=.02)
            except Exception as exc:
                on_graph(False, '夹爪只读订阅退出：'+str(exc), False)
        self.thread = threading.Thread(target=spin, name='r5-gripper-feedback', daemon=True)
        self.thread.start()

    def open(self, config, cancel_event, timeout_s):
        if cancel_event.is_set() or self.dead.is_set():
            raise ValueError('夹爪请求已取消')
        if not self.client.service_is_ready():
            raise ValueError('夹爪控制服务不可用；请现场检查初始化')
        request = self.service_type.Request()
        request.open = True
        request.position = 0.0  # Boolean-open mode => configured open_limit (12).
        # Explicitly preserve the verified raw defaults instead of relying on a
        # potentially different live service default. nextafter avoids truncating
        # 200 / 2.55 * 2.55 to 199 in the deployed C++ integer conversion.
        request.speed = math.nextafter(config['default_speed']/2.55, math.inf)
        request.force = math.nextafter(config['default_force']/2.55, math.inf)
        future = self.client.call_async(request)
        deadline = time.monotonic()+timeout_s
        while not future.done():
            if cancel_event.wait(.02) or self.dead.is_set():
                future.cancel()
                raise ValueError('本地等待已取消；已发出的服务请求不能保证撤销，请现场确认夹爪')
            if time.monotonic() >= deadline:
                future.cancel()
                raise ValueError('夹爪服务响应超时；已发出的请求不能保证撤销，请现场确认')
        result = future.result()
        if result is None:
            raise ValueError('夹爪服务没有返回结果')
        return {'success': bool(result.success)}

    def close(self):
        self.dead.set()
        if self.thread:
            self.thread.join(1)
            if self.thread.is_alive():
                raise ValueError('夹爪订阅线程尚未退出；未发送停止、复位或关夹爪命令')
        self.executor.remove_node(self.node)
        self.node.destroy_node()
        self.executor.shutdown()


class GripperGuard:
    def __init__(self, source=None, *, transport=None, config=None):
        self.source = source  # Context owner only; never use its /joint_states.
        self.lock = threading.RLock()
        self.close_lock = threading.Lock()
        self.closed = threading.Event()
        self.transport_closed = False
        self.worker = None
        self.transport = None
        self.cfg = None
        self.mode = 'IDLE'
        self.operation_error = None
        self.feedback_error = '尚无可关联的新鲜夹爪反馈；需现场检查初始化'
        self.routing_error = None
        self.graph_good = False
        self.graph_at = self.request_at = self.feedback_at = None
        self.open_since = None
        self.position = self.raw_status = self.fault = None
        self.invalidations = self.feedback_seq = 0
        try:
            self.cfg = _config(config)
            self.transport = transport if transport is not None else ROSTransport()
            self.transport.start(self._request, self._result, self._graph)
        except Exception as exc:
            self.mode = 'ERROR'
            self.operation_error = str(exc)

    def _invalidate(self, reason):
        self.feedback_error = reason
        self.feedback_at = self.open_since = None
        self.position = self.raw_status = self.fault = None
        self.invalidations += 1

    def _graph(self, valid, reason='', ambiguous=False):
        with self.lock:
            if self.closed.is_set():
                return
            self.graph_at = time.monotonic()
            self.graph_good = valid is True
            if ambiguous:
                self.routing_error = reason or 'Modbus 发布者不唯一；需检查并重启夹爪守护'
            if not self.graph_good or self.routing_error:
                self.request_at = None
                self._invalidate(self.routing_error or reason or 'Modbus 来源不可确认')

    def _request(self, message):
        with self.lock:
            if self.closed.is_set():
                return
            expected = dict(address=0x07D0, device=9, type=1, num=2)
            if (not isinstance(message, dict) or
                any(type(message.get(k)) is not int or message[k] != v for k, v in expected.items())):
                self.routing_error = '发现不同 Modbus 读请求，回复无关联ID；需检查并重启夹爪守护'
                self.request_at = None
                self._invalidate(self.routing_error)
                return
            self.request_at = time.monotonic()

    def _routing_reason(self, now):
        if self.routing_error:
            return self.routing_error
        if not self.graph_good or self.graph_at is None or now-self.graph_at > FEEDBACK_MAX_AGE_S:
            return 'Modbus 发布者检查缺失或已过期'
        if self.request_at is None or now-self.request_at > FEEDBACK_MAX_AGE_S:
            return '既有夹爪位置轮询缺失或已过期'
        return None

    def _result(self, message):
        with self.lock:
            if self.closed.is_set():
                return
            now = time.monotonic()
            reason = self._routing_reason(now)
            if reason:
                self._invalidate(reason); return
            if not isinstance(message, dict) or message.get('state') is not True:
                self._invalidate('夹爪 Modbus 读取失败；需现场检查初始化'); return
            data = message.get('read_data')
            if not isinstance(data, (list, tuple)) or len(data) != 4 or any(type(v) is not int or not 0 <= v <= 255 for v in data):
                self._invalidate('夹爪反馈格式错误：期望已核实的 4 字节 V3 返回'); return
            previous = self.feedback_at
            self.feedback_at = now
            self.feedback_seq += 1
            self.raw_status, self.position, self.fault = data[1], data[2], data[3]
            self.feedback_error = None
            if self._open_bits() and previous is not None and now-previous <= FEEDBACK_MAX_AGE_S and self.open_since is not None:
                pass
            else:
                self.open_since = now if self._open_bits() else None

    def _healthy_bits(self):
        return (self.raw_status is not None and self.fault == 0 and
                bool(self.raw_status & 1) and (self.raw_status >> 4) & 3 == 3)

    def _open_bits(self):
        return (self._healthy_bits() and (self.raw_status >> 6) & 3 == 3 and
                self.position is not None and 0 <= self.position <= TARGET_RAW+TOLERANCE_RAW)

    def _snapshot(self):
        now = time.monotonic()
        age = None if self.feedback_at is None else max(0, now-self.feedback_at)
        reason = self._routing_reason(now) or self.feedback_error
        fresh = age is not None and age <= FEEDBACK_MAX_AGE_S and not reason and not self.closed.is_set()
        if not fresh and not reason:
            reason = '夹爪设备反馈超过 250 ms；不能使用重发的 joint_states'
        healthy = fresh and self._healthy_bits()
        if fresh and not healthy:
            reason = '夹爪故障码 '+str(self.fault) if self.fault else '夹爪未完成激活；需要现场初始化'
        at_open = bool(fresh and self._open_bits())
        stable = max(0, self.feedback_at-self.open_since) if at_open and self.open_since is not None else 0.0
        ready = at_open and stable >= OPEN_STABLE_S and self.mode == 'IDLE' and not self.operation_error
        if self.closed.is_set():
            reason = '夹爪守护已关闭'
        return dict(at_max_open=at_open, ready=bool(ready), position_raw=self.position,
                    target_raw=TARGET_RAW, open_target_raw=TARGET_RAW, tolerance_raw=TOLERANCE_RAW,
                    feedback_age_s=age, mode=self.mode, error=self.operation_error or reason,
                    raw_status=self.raw_status, fault_code=self.fault,
                    verified_on_hardware=False, stable_s=stable,
                    can_prepare=bool(healthy and self.mode != 'OPENING' and not self.closed.is_set()),
                    initialization_supported=False, measurement_note=MEASUREMENT_NOTE,
                    requires_on_site_initialization=not bool(healthy))

    def status(self):
        with self.lock:
            return self._snapshot()

    def require_open(self):
        with self.lock:
            state = self._snapshot()
            if not state['ready']:
                raise ValueError(state['error'] or '夹爪尚未在最大允许张开位置持续确认 0.3 秒')
            return dict(state, observed_unix_s=time.time(), feedback_sequence=self.feedback_seq,verification_source='device_feedback',device_feedback_verified=True)

    def prepare_open(self):
        """Explicit user operation only. Returns OPENING; never blind-initializes."""
        with self.lock:
            if self.closed.is_set():
                raise ValueError('夹爪守护已关闭')
            if self.mode == 'OPENING' or (self.worker and self.worker.is_alive()):
                raise ValueError('夹爪张开操作已在进行；不能并发或自动重试')
            state = self._snapshot()
            if not self.cfg or not self.transport or not state['can_prepare']:
                self.mode = 'ERROR'
                self.operation_error = ('需要现场初始化/排查夹爪，未发出动作。当前接口不能确认或撤销电压/RS485/复位/回零初始化：'+
                                        str(state['error'] or '没有新鲜已激活的设备反馈'))
                raise ValueError(self.operation_error)
            self.mode = 'OPENING'
            self.operation_error = None
            self.open_since = None  # Pre-click samples cannot complete this operation.
            seq, invalidations = self.feedback_seq, self.invalidations
            self.worker = threading.Thread(target=self._prepare, args=(seq, invalidations),
                                           name='r5-gripper-explicit-open', daemon=True)
            self.worker.start()
            return self._snapshot()

    def _prepare(self, first_seq, first_invalidations):
        try:
            with self.lock:
                state = self._snapshot()
                if self.closed.is_set() or state['error'] or self.invalidations != first_invalidations or not self._healthy_bits():
                    raise ValueError(state['error'] or '发令前夹爪反馈失效；未调用开夹爪服务')
            response = self.transport.open(self.cfg, self.closed, OPEN_TIMEOUT_S)
            if not isinstance(response, dict) or response.get('success') is not True:
                raise ValueError('夹爪服务拒绝请求；未自动重试')
            # A service acknowledgement is NOT a completed physical movement.
            deadline = time.monotonic()+OPEN_TIMEOUT_S
            while not self.closed.wait(.02):
                with self.lock:
                    state = self._snapshot()
                    if self.invalidations != first_invalidations:
                        raise ValueError(state['error'] or '操作中夹爪反馈失效；请现场确认')
                    if state['error'] or not self._healthy_bits():
                        raise ValueError(state['error'] or '操作中夹爪失去激活状态')
                    if state['at_max_open'] and state['stable_s'] >= OPEN_STABLE_S and self.feedback_seq > first_seq:
                        self.mode = 'IDLE'; self.operation_error = None
                        return
                if time.monotonic() >= deadline:
                    raise ValueError('夹爪张开未由新反馈持续确认，操作超时；请现场检查，未自动重试')
            raise ValueError('本地等待取消；不能保证已发出的请求撤销，请现场检查夹爪')
        except Exception as exc:
            with self.lock:
                self.mode = 'ERROR'
                self.operation_error = str(exc)

    def close(self):
        """No reset/stop/close-gripper command; only release our ROS resources."""
        with self.close_lock:
            self.closed.set()
            with self.lock:
                self.open_since = None
                worker = self.worker
            if worker:
                worker.join(1)
                if worker.is_alive():
                    raise ValueError('夹爪请求等待线程未退出；无法保证已发请求撤销，未自动发任何补偿动作')
            if self.transport and not self.transport_closed:
                self.transport.close()
                self.transport_closed = True
