import copy,unittest
from types import SimpleNamespace
from ros_controller_telemetry import validate_packets,ROSControllerTelemetry

class UDPTests(unittest.TestCase):
    def packets(self):
        return {'system':(10.,dict(count=1,codes=[0])),'joints':(10.,dict(dof=6,codes=[0]*6)),'enabled':(10.,[True]*6)}
    def test_current_feedback_and_errors_are_preserved(self):
        p=self.packets();self.assertEqual(validate_packets(p,10.01),([0]*7,[True]*6))
        p['joints'][1]['codes'][2]=128
        self.assertEqual(validate_packets(p,10.01)[0][3],128)
    def test_missing_stale_and_invalid_feedback_rejected(self):
        for k in self.packets():
            p=self.packets();del p[k]
            with self.assertRaises(ValueError):validate_packets(p,10.01)
        with self.assertRaisesRegex(ValueError,'older'):validate_packets(self.packets(),10.2)
        p=self.packets();p['joints'][1]['dof']=7
        with self.assertRaises(ValueError):validate_packets(p,10.01)
    def test_adapter_only_reads_existing_monitor_and_joint_source(self):
        monitor=SimpleNamespace(controller_snapshot=lambda:([0]*7,[True]*6))
        source=SimpleNamespace(joint_feedback=lambda:([1,2,3,4,5,6],1))
        r=ROSControllerTelemetry(monitor,source)
        self.assertEqual(r.rm_create_robot_arm('unused',0).id,1)
        self.assertEqual(r.rm_get_current_arm_state(),(0,{'joint':[1,2,3,4,5,6],'err':{'err':[0]*7}}))
        monitor.controller_snapshot=lambda:([0]*7,[False]*6)
        with self.assertRaisesRegex(ValueError,'disabled'):r.rm_get_current_arm_state()
        self.assertEqual(r.rm_delete_robot_arm(),0)

if __name__=='__main__':unittest.main()
