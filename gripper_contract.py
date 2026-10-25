"""Dataset audit for the user's fixed maximum-allowed-open gripper requirement."""
import math


POLICY={'mode':'fixed_maximum_allowed_open','require_verified_open':True,'target_raw':12,
        'tolerance_raw':2,'model_gripper_value':0,'feedback_max_age_s':.25,
        'note':'0 is a fixed model slot; actual maximum-allowed opening is verified from raw JODELL feedback.'}


def verify_audit(audit):
    if isinstance(audit,dict) and audit.get('verification_source')=='user_configured_fixed':
        if (audit.get('ready') is not True or audit.get('assumed_max_open') is not True
            or audit.get('device_feedback_verified') is not False or audit.get('operator_confirmed') is not False
            or audit.get('position_raw') is not None or audit.get('at_max_open') is not None):
            raise ValueError('Configured fixation must not claim measured or attested aperture')
        return audit
    if not isinstance(audit,dict) or audit.get('ready') is not True or audit.get('at_max_open') is not True:
        raise ValueError('本帧缺少最大允许张开确认')
    if audit.get('verification_source','device_feedback') not in ('device_feedback','mechanical_fixture'):
        raise ValueError('未知夹爪确认来源')
    if audit.get('verification_source')=='mechanical_fixture':
        age=audit.get('confirmation_age_s')
        if audit.get('operator_confirmed') is not True or audit.get('fixture_locked') is not True or not audit.get('confirmation_id'):
            raise ValueError('缺少操作者的机械固定确认')
        if audit.get('device_feedback_verified') is not False or audit.get('position_raw') is not None:
            raise ValueError('人工确认不能伪装成设备位置测量')
        if type(age) not in (int,float) or not math.isfinite(age) or not 0<=age<1800:
            raise ValueError('机械固定确认已过期')
        return audit
    position=audit.get('position_raw');target=audit.get('open_target_raw',audit.get('target_raw'))
    status=audit.get('raw_status');fault=audit.get('fault_code');age=audit.get('feedback_age_s');stable=audit.get('stable_s')
    if type(position) is not int or not 0<=position<=14 or target!=12 or fault!=0:
        raise ValueError('夹爪位置、目标或故障状态不符合固定张开契约')
    if type(status) is not int or not 0<=status<=255 or not(status&1) or (status>>4)&3!=3 or (status>>6)&3!=3:
        raise ValueError('夹爪未激活、未到位或张开受阻')
    if type(age) not in (int,float) or not math.isfinite(age) or not 0<=age<=.25:
        raise ValueError('夹爪反馈不新鲜')
    if type(stable) not in (int,float) or not math.isfinite(stable) or stable<.3:
        raise ValueError('夹爪张开未持续确认0.3秒')
    return audit


def verify_rows(rows):
    if not rows:raise ValueError('无夹爪审计样本')
    audits=[verify_audit(row.get('gripper')) for row in rows]
    sources={a.get('verification_source','device_feedback') for a in audits}
    if len(sources)!=1:raise ValueError('同一条样本不能混用夹爪确认方式')
    if sources=={'user_configured_fixed'}:
        return dict(source_frames=len(audits),verification_source='user_configured_fixed',
                    fixed_max_open_verified=False,device_feedback_verified=False,policy_validated=True,
                    measurement='user-configured mechanical fixation; aperture not measured or session-attested')
    if sources=={'mechanical_fixture'}:
        return {'source_frames':len(audits),'verification_source':'mechanical_fixture',
                'confirmation_ids':sorted(set(a['confirmation_id'] for a in audits)),
                'fixed_max_open_verified':True,'device_feedback_verified':False,
                'measurement':'operator-confirmed mechanical fixation; no device position measurement'}
    values=[a['position_raw'] for a in audits]
    return {'source_frames':len(values),'target_raw':12,'min_reported_raw':min(values),
            'max_reported_raw':max(values),'fixed_max_open_verified':True,
            'verification_source':'device_feedback','device_feedback_verified':True,
            'measurement':'device register and status; not independent physical metrology'}
