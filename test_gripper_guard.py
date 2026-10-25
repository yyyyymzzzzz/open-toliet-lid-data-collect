"""Fake transport tests only: no ROS import, service calls or SDK instances."""
import json
import threading
import time
import types
import unittest
from unittest.mock import patch

from gripper_guard import GripperGuard, ROSTransport

CONFIG = dict(type='realman_jodell', open_limit=12, close_limit=255,
              tool_voltage=3, rs485_baud=115200, modbus_slave_id=9,
              default_speed=200, default_force=200)


class FakeTransport:
    def __init__(self):
        self.commands = []
        self.open_called = threading.Event()
        self.closed = False
        self.block = None
        self.reply = {'success': True}

    def start(self, request, result, graph):
        self.request, self.result, self.graph = request, result, graph

    def emit(self, position=12, status=0xF1, fault=0):
        self.graph(True, '', False)
        self.request(dict(address=2000, device=9, type=1, num=2))
        self.result(dict(state=True, read_data=[0, status, position, fault]))

    def open(self, config, cancel, timeout_s):
        self.commands.append(dict(target=config['open_limit'], speed=config['default_speed'], force=config['default_force']))
        self.open_called.set()
        if self.block:
            while not self.block.wait(.01):
                if cancel.is_set():
                    raise ValueError('fake request cancelled')
        return self.reply

    def close(self):
        self.closed = True


def until(fn, timeout=2):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if fn():
            return
        time.sleep(.01)
    raise AssertionError('fake gripper timeout')


class GripperTests(unittest.TestCase):
    def setUp(self):
        self.transport = FakeTransport()
        self.guard = GripperGuard(transport=self.transport, config=CONFIG)

    def tearDown(self):
        if self.transport.block:
            self.transport.block.set()
        self.guard.close()

    def stable(self, position=12):
        for _ in range(9):
            self.transport.emit(position=position)
            time.sleep(.04)

    def test_constructor_status_require_and_close_never_command(self):
        self.assertEqual(self.transport.commands, [])
        self.assertFalse(self.guard.status()['ready'])
        with self.assertRaises(ValueError):
            self.guard.require_open()
        self.stable()
        audit = self.guard.require_open()
        json.dumps(audit, allow_nan=False)
        self.assertEqual(audit['position_raw'], 12)
        self.assertFalse(audit['verified_on_hardware'])
        self.guard.close()
        self.assertEqual(self.transport.commands, [])

    def test_default_transport_constructor_is_passive(self):
        fake = FakeTransport()
        with patch('gripper_guard.ROSTransport', return_value=fake):
            guard = GripperGuard(config=CONFIG)
            self.assertEqual(fake.commands, [])
            guard.status()
            guard.close()
            self.assertEqual(fake.commands, [])

    def test_failed_result_invalidates_cached_success(self):
        self.transport.emit()
        self.assertTrue(self.guard.status()['at_max_open'])
        self.transport.result(dict(state=False, read_data=[]))
        status = self.guard.status()
        self.assertFalse(status['at_max_open'])
        self.assertIsNone(status['position_raw'])
        self.assertIn('读取失败', status['error'])

    def test_invalid_shapes_and_bytes_are_not_feedback(self):
        for data in ([0, 241], [0, 241, -1, 0], [0, 241, 12, True], [0, 241, 256, 0]):
            self.transport.emit()
            self.transport.result(dict(state=True, read_data=data))
            self.assertFalse(self.guard.status()['at_max_open'])

    def test_cache_expiry_without_new_result(self):
        self.transport.emit()
        time.sleep(.27)
        self.transport.graph(True, '', False)
        self.transport.request(dict(address=2000, device=9, type=1, num=2))
        self.assertFalse(self.guard.status()['at_max_open'])
        self.assertGreater(self.guard.status()['feedback_age_s'], .25)

    def test_blocked_object_and_fault_bits_rejected(self):
        self.transport.emit(status=0x71)
        self.assertFalse(self.guard.status()['at_max_open'])
        for fault in (1, 2, 4, 8, 16, 32, 64, 128):
            self.transport.emit(fault=fault)
            self.assertFalse(self.guard.status()['can_prepare'])
            self.assertEqual(self.guard.status()['fault_code'], fault)

    def test_position_below_target_allowed_without_lowering_target(self):
        for pos in (0, 9, 12, 14):
            self.transport.emit(position=pos)
            self.assertTrue(self.guard.status()['at_max_open'])
            self.assertEqual(self.guard.status()['target_raw'], 12)
        self.transport.emit(position=15)
        self.assertFalse(self.guard.status()['at_max_open'])

    def test_no_goto_requirement_for_arrived_f1(self):
        self.stable()
        self.assertEqual(self.guard.status()['raw_status'], 0xF1)
        self.assertTrue(self.guard.require_open()['ready'])

    def test_extra_request_latches_ambiguity(self):
        self.transport.emit()
        self.transport.request(dict(address=2000, device=9, type=1, num=3))
        self.transport.emit()
        self.assertFalse(self.guard.status()['at_max_open'])
        with self.assertRaisesRegex(ValueError, '现场初始化'):
            self.guard.prepare_open()
        self.assertEqual(self.transport.commands, [])

    def test_extra_publisher_and_missing_graph_refuse(self):
        self.transport.emit()
        self.transport.graph(False, 'extra publisher', True)
        self.transport.emit()
        self.assertFalse(self.guard.status()['ready'])
        self.assertIn('extra publisher', self.guard.status()['error'])

    def test_unactivated_requires_on_site_initialization_no_blind_packets(self):
        self.transport.emit(status=0)
        with self.assertRaisesRegex(ValueError, '未发出动作'):
            self.guard.prepare_open()
        self.assertEqual(self.guard.status()['mode'], 'ERROR')
        self.assertEqual(self.transport.commands, [])

    def test_only_explicit_prepare_sends_preserved_target_and_defaults(self):
        self.transport.emit(position=200)
        self.assertEqual(self.transport.commands, [])
        self.guard.prepare_open()
        self.assertTrue(self.transport.open_called.wait(1))
        self.assertEqual(self.transport.commands, [dict(target=12, speed=200, force=200)])
        self.assertEqual(self.guard.status()['mode'], 'OPENING')
        self.assertFalse(self.guard.status()['ready'])
        self.stable()
        until(lambda:self.guard.status()['mode']=='IDLE')
        self.assertTrue(self.guard.require_open()['ready'])
        self.assertEqual(len(self.transport.commands), 1)

    def test_concurrent_prepare_rejected(self):
        self.transport.block = threading.Event()
        self.transport.emit(position=200)
        self.guard.prepare_open()
        self.assertTrue(self.transport.open_called.wait(1))
        with self.assertRaisesRegex(ValueError, '并发'):
            self.guard.prepare_open()
        self.assertEqual(len(self.transport.commands), 1)

    def test_service_failure_is_not_completion(self):
        self.transport.reply = {'success': False}
        self.transport.emit()
        self.guard.prepare_open()
        until(lambda:self.guard.status()['mode']=='ERROR')
        self.assertFalse(self.guard.status()['ready'])
        self.assertEqual(len(self.transport.commands), 1)

    def test_failed_feedback_after_ack_aborts_without_retry(self):
        self.transport.emit(position=200)
        self.guard.prepare_open()
        self.assertTrue(self.transport.open_called.wait(1))
        self.transport.result(dict(state=False, read_data=[]))
        until(lambda:self.guard.status()['mode']=='ERROR')
        self.assertEqual(len(self.transport.commands), 1)

    def test_stale_feedback_after_ack_is_not_completion(self):
        self.transport.emit(position=200)
        self.guard.prepare_open()
        until(lambda:self.guard.status()['mode']=='ERROR')
        self.assertFalse(self.guard.status()['ready'])
        self.assertEqual(len(self.transport.commands), 1)

    def test_changed_limit_rejected_without_command(self):
        fake = FakeTransport()
        guard = GripperGuard(transport=fake, config={**CONFIG, 'open_limit': 0})
        self.assertEqual(guard.status()['mode'], 'ERROR')
        with self.assertRaises(ValueError):
            guard.prepare_open()
        self.assertEqual(fake.commands, [])
        guard.close()

    def test_open_stability_restarts_after_failure(self):
        self.stable()
        self.assertTrue(self.guard.status()['ready'])
        self.transport.result(dict(state=False, read_data=[]))
        self.transport.emit()
        self.assertTrue(self.guard.status()['at_max_open'])
        self.assertFalse(self.guard.status()['ready'])

    def test_close_cancels_wait_without_compensating_commands(self):
        self.transport.block = threading.Event()
        self.transport.emit(position=200)
        self.guard.prepare_open()
        self.assertTrue(self.transport.open_called.wait(1))
        self.guard.close()
        self.assertTrue(self.transport.closed)
        self.assertEqual(len(self.transport.commands), 1)
        self.assertFalse(self.guard.status()['ready'])
        with self.assertRaises(ValueError):
            self.guard.prepare_open()

    def test_invalidated_before_worker_cannot_send(self):
        self.transport.emit(position=200)
        seq, invalidations = self.guard.feedback_seq, self.guard.invalidations
        self.guard.mode = 'OPENING'
        self.transport.result(dict(state=False, read_data=[]))
        self.guard._prepare(seq, invalidations)
        self.assertEqual(self.transport.commands, [])
        self.assertEqual(self.guard.status()['mode'], 'ERROR')

    def test_ros_request_shape_without_constructing_ros_transport(self):
        transport = object.__new__(ROSTransport)
        transport.dead = threading.Event()
        transport.service_type = types.SimpleNamespace(Request=types.SimpleNamespace)
        sent = []
        future = types.SimpleNamespace(done=lambda:True, result=lambda:types.SimpleNamespace(success=True))
        transport.client = types.SimpleNamespace(service_is_ready=lambda:True,
            call_async=lambda request:(sent.append(request) or future))
        self.assertEqual(transport.open(CONFIG, threading.Event(), .1), {'success': True})
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0].open)
        self.assertEqual(sent[0].position, 0.0)
        self.assertEqual(int(sent[0].speed*2.55), 200)
        self.assertEqual(int(sent[0].force*2.55), 200)

    def test_ros_request_timeout_no_retry_or_false_success(self):
        transport = object.__new__(ROSTransport)
        transport.dead = threading.Event()
        transport.service_type = types.SimpleNamespace(Request=types.SimpleNamespace)
        sent, cancelled = [], []
        future = types.SimpleNamespace(done=lambda:False, cancel=lambda:cancelled.append(True))
        transport.client = types.SimpleNamespace(service_is_ready=lambda:True,
            call_async=lambda request:(sent.append(request) or future))
        with self.assertRaisesRegex(ValueError, '不能保证撤销'):
            transport.open(CONFIG, threading.Event(), .01)
        self.assertEqual(len(sent), 1)
        self.assertEqual(cancelled, [True])


if __name__ == '__main__':
    unittest.main(verbosity=2)
