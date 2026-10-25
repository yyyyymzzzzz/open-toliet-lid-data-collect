import tempfile
import unittest
from pathlib import Path
from fixed_gripper import MechanicalFixedGuard,ConfiguredFixedGuard
from gripper_contract import verify_audit,verify_rows


class FixedGripperTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.guard=MechanicalFixedGuard(self.tmp.name)
    def tearDown(self):self.guard.close();self.tmp.cleanup()
    def test_no_implicit_confirmation(self):
        self.assertFalse(self.guard.status()['ready'])
        with self.assertRaises(ValueError):self.guard.require_open()
        with self.assertRaises(ValueError):self.guard.confirm(True,False)
    def test_explicit_attestation_is_not_sensor_measurement(self):
        self.guard.confirm(True,True);audit=verify_audit(self.guard.require_open())
        self.assertFalse(audit['device_feedback_verified']);self.assertIsNone(audit['position_raw'])
        summary=verify_rows([{'gripper':audit}]);self.assertEqual(summary['verification_source'],'mechanical_fixture')
        self.assertTrue(list((Path(self.tmp.name)/'panel/gripper_confirmations').glob('*.json')))
        forged=dict(audit,device_feedback_verified=True)
        with self.assertRaises(ValueError):verify_audit(forged)
    def test_expiry_and_revoke(self):
        self.guard.confirm(True,True);self.guard.started-=1801
        with self.assertRaises(ValueError):self.guard.require_open()
        self.guard.confirm(True,True);self.guard.revoke()
        self.assertFalse(self.guard.status()['ready'])
    def test_restart_does_not_reuse_attestation(self):
        self.guard.confirm(True,True);other=MechanicalFixedGuard(self.tmp.name)
        self.assertFalse(other.status()['ready']);other.close()


class ConfiguredTests(unittest.TestCase):
    def test_user_configuration_never_claims_measurement(self):
        guard=ConfiguredFixedGuard();audit=verify_audit(guard.require_open())
        self.assertTrue(audit['ready']);self.assertIsNone(audit['at_max_open'])
        summary=verify_rows([{'gripper':audit}])
        self.assertFalse(summary['fixed_max_open_verified'])
        self.assertTrue(summary['policy_validated'])
        for key,value in [('position_raw',12),('at_max_open',True),('device_feedback_verified',True),('operator_confirmed',True)]:
            with self.assertRaises(ValueError):verify_audit(dict(audit,**{key:value}))
        guard.close()
        with self.assertRaises(ValueError):guard.require_open()

if __name__=='__main__':unittest.main(verbosity=2)
