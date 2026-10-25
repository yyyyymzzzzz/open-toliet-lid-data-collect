"""Operator-attested mechanical fixation. No ROS, SDK, power or gripper commands."""
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid
from datetime import datetime,timezone


class MechanicalFixedGuard:
    def __init__(self,storage,ttl_s=1800):
        if not math.isfinite(ttl_s) or not 60<=ttl_s<=1800:raise ValueError('Invalid fixture confirmation lifetime')
        self.root=Path(storage)/'panel/gripper_confirmations';self.root.mkdir(parents=True,exist_ok=True)
        self.ttl=ttl_s;self.lock=threading.RLock();self.confirmation=None;self.started=None;self.closed=False

    def status(self):
        with self.lock:
            age=time.monotonic()-self.started if self.started is not None else None
            valid=bool(self.confirmation and age is not None and 0<=age<self.ttl and not self.closed)
            error=None if valid else ('固定确认已过期，请重新检查并确认' if self.confirmation else '请现场张开到最大并机械锁住，然后点击确认')
            if self.closed:error='固定确认已撤销'
            return {'verification_source':'mechanical_fixture','at_max_open':valid,'ready':valid,
                    'operator_confirmed':valid,'fixture_locked':valid,'mode':'FIXED' if valid else 'AWAITING_CONFIRMATION',
                    'confirmation_id':self.confirmation['id'] if self.confirmation else None,
                    'confirmed_at':self.confirmation['confirmed_at'] if self.confirmation else None,
                    'confirmation_unix_s':self.confirmation['unix_s'] if self.confirmation else None,
                    'confirmation_age_s':age,'remaining_s':max(0,self.ttl-age) if age is not None else 0,
                    'position_raw':None,'target_raw':None,'feedback_age_s':None,'raw_status':None,'fault_code':None,
                    'device_feedback_verified':False,'verified_on_hardware':False,'error':error,
                    'measurement_note':'人工确认机械固定在最大开度；无电控位置测量，也不向夹爪发令。'}

    def confirm(self,confirmed,fixture_locked):
        if confirmed is not True or fixture_locked is not True:raise ValueError('需要确认已经张开到最大并机械固定，不只是可手动张开')
        with self.lock:
            if self.closed:raise ValueError('夹爪固定守卫已关闭')
            evidence={'id':uuid.uuid4().hex,'confirmed_at':datetime.now(timezone.utc).isoformat(),'unix_s':time.time(),
                      'verification_source':'mechanical_fixture','operator_confirmed':True,'fixture_locked':True,
                      'device_feedback_verified':False,'ttl_s':self.ttl}
            target=self.root/(evidence['id']+'.json');temp=target.with_suffix('.tmp')
            with temp.open('x') as f:json.dump(evidence,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
            os.replace(temp,target);self.confirmation=evidence;self.started=time.monotonic()
        return self.status()

    def require_open(self):
        result=self.status()
        if not result['ready']:raise ValueError(result['error'])
        return dict(result,observed_unix_s=time.time())

    def revoke(self):
        with self.lock:
            previous=self.confirmation;self.confirmation=None;self.started=None
        if previous:
            path=self.root/(previous['id']+'.revoked.json')
            path.write_text(json.dumps({'confirmation_id':previous['id'],'revoked_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
        return self.status()

    def prepare_open(self):raise ValueError('当前为机械固定模式，不驱动夹爪；请固定张开后确认')

    def close(self):
        with self.lock:self.closed=True;self.confirmation=None;self.started=None


class ConfiguredFixedGuard:
    """Explicit user policy, no physical measurement or per-session attestation."""
    def __init__(self): self.closed=False
    def status(self):
        return dict(verification_source='user_configured_fixed',ready=not self.closed,
                    at_max_open=None,assumed_max_open=True,operator_confirmed=False,
                    device_feedback_verified=False,verified_on_hardware=False,
                    position_raw=None,mode='CONFIGURED_FIXED',
                    error='Guard closed' if self.closed else None,
                    measurement_note='用户配置机械固定张开；没有测量实际开度，不向夹爪发令。')
    def require_open(self):
        result=self.status()
        if not result['ready']:raise ValueError(result['error'])
        return dict(result,observed_unix_s=time.time())
    def close(self):self.closed=True
