"""Read-only adapter for existing rm_driver UDP-derived ROS feedback. No TCP socket."""
import time
from types import SimpleNamespace


def validate_packets(packets,now=None):
    now=time.monotonic() if now is None else now
    for key in ('system','joints','enabled'):
        if key not in packets:raise ValueError('Waiting for rm_driver UDP '+key+' feedback')
        if not 0<=now-packets[key][0]<=.15:raise ValueError('rm_driver UDP '+key+' feedback older than 150 ms')
    system=packets['system'][1];joints=packets['joints'][1];enabled=packets['enabled'][1]
    if system['count']!=len(system['codes']) or any(type(v) is not int for v in system['codes']):
        raise ValueError('Invalid UDP system alarm payload')
    if joints['dof']!=6 or len(joints['codes'])!=6 or any(type(v) is not int for v in joints['codes']):
        raise ValueError('Invalid UDP joint alarm payload')
    if len(enabled)!=6 or any(type(v) is not bool for v in enabled):raise ValueError('Invalid UDP enable payload')
    return list(system['codes'])+list(joints['codes']),list(enabled)


class ROSControllerTelemetry:
    def __init__(self,monitor,source):self.monitor=monitor;self.source=source
    def rm_create_robot_arm(self,ip,port):
        self.monitor.controller_snapshot()
        return SimpleNamespace(id=1)
    def rm_get_current_arm_state(self):
        codes,enabled=self.monitor.controller_snapshot()
        if not all(enabled):raise ValueError('UDP feedback reports disabled joint; inspect controller')
        joints,_=self.source.joint_feedback()
        return 0,dict(joint=joints,err={'err':codes})
    def rm_delete_robot_arm(self):return 0
