"""Local-only robot-5 data workbench. No robot publishers, motion services, or SDK calls."""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime,timezone
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
import uuid
import cv2
import numpy as np
from observations import Observations,load_config,decode_image
from record_episode import Writer,quality
from export_v21 import export
from validate_dataset import validate
from gripper_contract import verify_audit,verify_rows

HERE=Path(__file__).resolve().parent
CAMERAS=('cam_bottom','cam_wrist')
SAFE=re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,100}$')


def now():return datetime.now(timezone.utc).isoformat()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def read_json(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default


class Conflict(ValueError):pass


class Workbench:
    def __init__(self,cfg,source):
        self.cfg=cfg;self.source=source;self.instance_id=uuid.uuid4().hex
        self.root=Path(cfg['data_root']).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.storage=self.root.parent;self.panel=self.storage/'panel'
        for name in ['jobs','evaluations']:(self.panel/name).mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock();self.capture_stop=threading.Event();self.capture_thread=None
        self.export_thread=None;self.last_id=None;self.capture_started=None
        self.recorder={'id':None,'frames':0,'elapsed_s':0,'error':None,'diagnostic':False}
        self.model_lock=threading.Lock();self.model_payload=None
        self.mode='IDLE';self.images={};self.closed=False;self.motion=None;self.gripper=None;self.replay_episode_id=None
        self.lockfile=(self.panel/'service.lock').open('a')
        try:fcntl.flock(self.lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise Conflict('另一个面板服务正在使用此数据目录')
        # A process interruption never converts an unfinished export into a completed dataset.
        for path in (self.panel/'jobs').glob('*.json'):
            job=read_json(path)
            if job.get('status')=='RUNNING':
                job.update(status='FAILED',error='服务中断；导出未确认完成');atomic_json(path,job)

    def robot_model(self):
        with self.model_lock:
            if self.model_payload is None:
                result=subprocess.run([sys.executable,str(HERE/'robot_model.py')],capture_output=True,check=True,timeout=60)
                self.model_payload=result.stdout
            return self.model_payload

    def episode_dir(self,episode_id):
        if not isinstance(episode_id,str) or not SAFE.fullmatch(episode_id):raise ValueError('无效样本编号')
        path=(self.root/episode_id).resolve()
        if not path.is_relative_to(self.root) or not path.is_dir():raise ValueError('样本不存在')
        return path

    def episode(self,path):
        q=read_json(path/'quality.json',{})
        meta=read_json(path/'capture.json',{})
        is_recording=self.mode=='RECORDING' and self.recorder['id']==path.name
        status=('RECORDING' if is_recording else 'DIAGNOSTIC' if q.get('diagnostic') else
                'ACCEPTED' if q.get('accepted') and q.get('status')=='PASS' else
                'FAILED' if q.get('status')!='PASS' else 'REJECTED' if q.get('review_decision')=='reject' else 'PENDING')
        return {'id':path.name,'created_at':meta.get('created_at',datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()),
                'frames':self.recorder['frames'] if is_recording else q.get('frames',0),
                'actual_source_hz':q.get('actual_source_hz'),'max_gap_s':q.get('max_gap_s'),
                'status':status,'diagnostic':bool(q.get('diagnostic',meta.get('diagnostic',False))),
                'accepted':bool(q.get('accepted')),'quality':q,'task':self.cfg['task']}

    def episodes(self):
        items=[]
        for path in sorted(self.root.iterdir(),reverse=True):
            if path.is_dir() and not path.is_symlink() and SAFE.fullmatch(path.name) and (path/'config.json').is_file():
                items.append(self.episode(path))
        return items

    def status(self):
        obs_state=None;reason='';ready=False
        try:obs,_=self.source.snapshot(decode=False);obs_state=obs['observation.state'].tolist();ready=True
        except Exception as exc:reason=str(exc)[:300]
        gripper=self.gripper.status() if self.gripper else {'at_max_open':False,'ready':False,'error':'夹爪反馈模块未接入','mode':'ERROR'}
        if not gripper.get('ready'):
            ready=False;reason=gripper.get('error') or '夹爪尚未在最大允许开度持续确认'
        cams={}
        with self.source.lock:
            for cam in CAMERAS:
                sample=self.source.samples.get(cam);history=list(self.source.history[cam])
                stamps=sorted(set(v[1] for v in history))
                hz=(len(stamps)-1)*1e9/(stamps[-1]-stamps[0]) if len(stamps)>1 and stamps[-1]>stamps[0] else 0
                age=(time.monotonic_ns()-sample[2])/1e9 if sample else None
                cams[cam]={'available':age is not None and age<=self.cfg['max_age_s'],'age_s':age,'hz':hz if age is not None and age<=self.cfg['max_age_s'] else 0}
        episodes=self.episodes()
        disks={}
        for name,path in [('system','/'),('data',self.storage)]:
            usage=shutil.disk_usage(path);disks[name]={'free_bytes':usage.free,'total_bytes':usage.total}
        with self.lock:
            recorder=dict(self.recorder)
            if self.mode=='RECORDING' and self.capture_started:recorder['elapsed_s']=time.monotonic()-self.capture_started
            mode=self.mode
        return {'now':now(),'service_instance_id':self.instance_id,'mode':mode,'recorder':recorder,
                'observation':{'ready':ready,'reason':reason,'state':obs_state,'cameras':cams},'disks':disks,
                'summary':{'total':len(episodes),'accepted':sum(e['status']=='ACCEPTED' for e in episodes),
                           'diagnostic':sum(e['diagnostic'] for e in episodes),'frames':sum(e['frames'] for e in episodes)},
                'contract':{'fps':30,'state_dim':17,'action_dim':7,'horizon':16},'training':{'connected':False},
                'gripper':gripper,
                'motion':self.motion.status() if self.motion else {'enabled':False,'mode':'DISARMED','ready':False,'reason':'手动控制模块未接入','owner':None,'trajectory':None,'error':None,'progress':0}}

    def frame(self,cam,episode_id=None,index=None):
        if cam not in CAMERAS:raise ValueError('无效相机')
        if episode_id is None:
            with self.source.lock:
                sample=self.source.samples.get(cam)
                if sample is None or time.monotonic_ns()-sample[2]>self.cfg['max_age_s']*1e9:raise Conflict('相机画面过期')
                data,stamp,_=sample
            with self.lock:
                cached=self.images.get(cam)
                if cached and cached[0]==stamp:return cached[1]
                self.images[cam]=(stamp,data)
            return data
        else:
            path=self.episode_dir(episode_id)
            if not isinstance(index,int) or not 0<=index<=1_000_000:raise ValueError('无效帧序号')
            file=(path/cam/f'{index:06d}.jpg').resolve()
            if not file.is_relative_to(path) or not file.is_file():raise ValueError('帧不存在')
            return file.read_bytes()

    def rows(self,episode_id):
        path=self.episode_dir(episode_id)/'frames.jsonl'
        if path.stat().st_size>64*1024**2:raise ValueError('单条样本超过面板读取上限')
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def detail(self,episode_id):
        path=self.episode_dir(episode_id);rows=self.rows(episode_id)
        t0=rows[0]['timing']['source_ns']['cam_bottom'] if rows else 0
        step=max(1,math.ceil(len(rows)/1000));indices=list(range(0,len(rows),step))
        if rows and indices[-1]!=len(rows)-1:indices.append(len(rows)-1)
        series=[]
        for i in indices:
            row=rows[i];t=row['timing']['source_ns']['cam_bottom']
            previous=rows[i-1]['timing']['source_ns']['cam_bottom'] if i else t
            series.append({'frame_index':row['frame_index'],'t':(t-t0)/1e9,'joints':row['state'][:6],
                           'xyz':row['state'][6:9],'dt_ms':(t-previous)/1e6})
        joints=[];times=[]
        for row in rows:
            state=row.get('state') or []
            joints.append([round(float(v),3) for v in list(state)[:6]])
            times.append(round((row['timing']['source_ns']['cam_bottom']-t0)/1e9,4))
        ep=self.episode(path)
        return {'episode':ep,'frame_count':len(rows),'series':series,'joints':joints,'times':times,'quality':ep['quality']}

    def start(self,diagnostic=False,_motion_replay=False):
        if type(diagnostic) is not bool:raise ValueError('diagnostic必须为布尔值')
        self.check_gripper()
        self.source.snapshot(decode=False)
        with self.lock:
            if self.closed:raise Conflict('服务正在退出')
            if self.capture_thread and self.capture_thread.is_alive():raise Conflict('已有录制正在进行')
            if self.export_thread and self.export_thread.is_alive():raise Conflict('导出进行中，完成后可继续采集')
            if not _motion_replay and self.motion and self.motion.status()['mode'] in ('TEACHING','HOMING','RESETTING','REPLAYING','STOPPING','FAULT'):
                raise Conflict('手动控制任务进行中或尚未确认停止')
            if shutil.disk_usage(self.storage).free<self.cfg['minimum_free_gib']*1024**3:raise Conflict('数据盘剩余空间不足')
            episode_id=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            writer=Writer(self.root/episode_id)
            atomic_json(writer.root/'config.json',self.cfg)
            atomic_json(writer.root/'capture.json',{'created_at':now(),'diagnostic':diagnostic,'source':'dashboard','motion_commands_sent':0})
            self.capture_stop.clear();self.capture_started=time.monotonic();self.last_id=episode_id
            self.recorder={'id':episode_id,'frames':0,'elapsed_s':0,'error':None,'diagnostic':diagnostic}
            self.mode='RECORDING'
            if _motion_replay:self.replay_episode_id=episode_id
            self.capture_thread=threading.Thread(target=self._record,args=(writer,diagnostic),daemon=True)
            self.capture_thread.start()
        return {'id':episode_id,'mode':'RECORDING'}

    def _record(self,writer,diagnostic):
        rows=[];previous={};last_good=time.monotonic();error=None
        try:
            while not self.capture_stop.is_set():
                if time.monotonic()-self.capture_started>600:raise RuntimeError('达到单条录制10分钟上限，请分段采集')
                grip=self.check_gripper()
                try:
                    obs,timing=self.source.snapshot(decode=False);stamps=timing['source_ns']
                    if any(stamps[c]<=previous.get(c,0) for c in CAMERAS):
                        if time.monotonic()-last_good>.25:raise RuntimeError('相机帧停止更新')
                        time.sleep(.002);continue
                    row={'frame_index':len(rows),'state':obs['observation.state'].tolist(),'timing':timing,'gripper':grip}
                    writer.add(row,{c:obs['observation.images.'+c] for c in CAMERAS});rows.append(row)
                    previous=stamps;last_good=time.monotonic()
                    with self.lock:self.recorder['frames']=len(rows)
                    if len(rows)%150==0 and shutil.disk_usage(self.storage).free<self.cfg['minimum_free_gib']*1024**3:
                        raise RuntimeError('数据盘空间不足，停止采集')
                except ValueError as exc:
                    if time.monotonic()-last_good>.25:raise RuntimeError('观测中断：'+str(exc)) from exc
                time.sleep(.002)
        except Exception as exc:
            error=str(exc)
            self.cancel_replay_for_record_failure(writer.root.name)
        finally:
            try:writer.close()
            except Exception as exc:
                error='写盘失败：'+str(exc)
                self.cancel_replay_for_record_failure(writer.root.name)
            try:report=quality(rows,self.cfg['fps'])
            except Exception as exc:report={'status':'FAIL','errors':[str(exc)],'frames':len(rows)}
            report.update(accepted=False,diagnostic=diagnostic,motion_commands_sent=0,source='dashboard',gripper_policy_validated=bool(rows) and not error,gripper_open_verified=bool(rows) and not error and all(r['gripper'].get('verification_source')!='user_configured_fixed' for r in rows))
            if error:report['status']='FAIL';report.setdefault('errors',[]).append(error)
            if self.closed:report['status']='FAIL';report.setdefault('errors',[]).append('服务退出，片段未接受')
            if report.get('status')=='PASS' and not diagnostic and not error and not self.closed:
                report.update(accepted=True,review_decision='auto_accept',review_note='质量通过，自动接受',reviewed_at=now())
            try:atomic_json(writer.root/'quality.json',report)
            except Exception as exc:error='质量报告写入失败：'+str(exc)
            with self.lock:
                self.recorder.update(error=error,elapsed_s=time.monotonic()-self.capture_started)
                self.mode='ERROR' if error else 'REVIEW'

    def cancel_replay_for_record_failure(self,episode_id):
        if self.motion and self.replay_episode_id==episode_id:
            # ManualMotion.stop sets cancellation and launches its stop worker without joining it.
            # Do this before writer.close/quality/fsync, which may be blocked by a failing disk.
            try:self.motion.stop()
            except Exception as exc:print('motion_stop_after_record_failure',str(exc),flush=True)

    def stop(self,_motion_replay=False):
        with self.lock:
            if not _motion_replay and self.replay_episode_id==self.recorder['id'] and self.replay_episode_id is not None:
                raise Conflict('回放中的采集由回放流程结束；需要中断请使用“停止机械臂”')
            thread=self.capture_thread
            if not thread or not thread.is_alive():raise Conflict('当前没有正在录制的片段')
            self.capture_stop.set()
        thread.join(timeout=8)
        if thread.is_alive():raise Conflict('仍在保存数据，请稍后查看状态')
        return {'id':self.last_id,'mode':self.mode}

    def review(self,episode_id,decision,note=''):
        if decision not in ('accept','reject') or not isinstance(note,str) or len(note)>500:raise ValueError('无效审核内容')
        with self.lock:
            if self.mode=='RECORDING' and self.recorder['id']==episode_id:raise Conflict('请先结束录制')
            if self.replay_episode_id==episode_id:raise Conflict('回放尚未完成停止和数据收尾，不能审核')
            if self.export_thread and self.export_thread.is_alive():raise Conflict('导出时不能修改接受状态')
            path=self.episode_dir(episode_id);q=read_json(path/'quality.json')
            if not q:raise Conflict('样本尚未保存完整质量报告')
            if decision=='accept' and (q.get('diagnostic') or q.get('status')!='PASS'):raise Conflict('诊断或质量失败样本不能作为成功示教')
            if decision=='accept':
                rows=self.rows(episode_id);verify_rows(rows)
                fresh=quality(rows,self.cfg['fps'])
                if fresh['status']!='PASS':raise Conflict('重新检查质量未通过')
            q.update(accepted=decision=='accept',review_decision=decision,review_note=note,reviewed_at=now())
            atomic_json(path/'quality.json',q)
            if self.last_id==episode_id:self.mode='IDLE'
        return self.episode(path)

    def exports(self):return sorted([read_json(p) for p in (self.panel/'jobs').glob('*.json')],key=lambda j:j['created_at'],reverse=True)

    def start_export(self,name=''):
        if not isinstance(name,str):raise ValueError('导出名称必须为字符串')
        name=name or 'r5_joint_abs_'+datetime.now().strftime('%Y%m%d_%H%M%S')
        if not SAFE.fullmatch(name):raise ValueError('名称仅允许字母、数字、下划线和短横线')
        with self.lock:
            if self.mode=='RECORDING' or (self.capture_thread and self.capture_thread.is_alive()):raise Conflict('录制结束后再导出')
            if self.motion and self.motion.status()['mode']!='DISARMED':raise Conflict('请先停止并释放手动控制会话，再导出数据')
            if self.export_thread and self.export_thread.is_alive():raise Conflict('已有导出任务')
            sources=[self.episode_dir(e['id']) for e in self.episodes() if e['accepted'] and not e['diagnostic'] and e['quality'].get('status')=='PASS']
            if not sources:raise Conflict('没有可导出的已接受成功示教')
            target=self.storage/'datasets'/name
            if target.exists():raise Conflict('导出名称已存在，请换一个名称')
            job={'id':uuid.uuid4().hex,'name':name,'status':'RUNNING','frames':0,'episodes':len(sources),
                 'error':None,'path':str(target),'created_at':now(),'source_ids':[p.name for p in sources]}
            atomic_json(self.panel/'jobs'/(job['id']+'.json'),job)
            self.export_thread=threading.Thread(target=self._export,args=(sources,target,job),daemon=True);self.export_thread.start()
        return job

    def _export(self,sources,target,job):
        try:
            info=export(list(reversed(sources)),target);report=validate(target,for_training=True)
            atomic_json(target/'meta/validation.json',report)
            job.update(status='DONE',frames=info['total_frames'],episodes=info['total_episodes'],validation=report)
        except Exception as exc:job.update(status='FAILED',error=str(exc)[:1000])
        finally:atomic_json(self.panel/'jobs'/(job['id']+'.json'),job)

    def evaluations(self):
        results=sorted([read_json(p) for p in (self.panel/'evaluations').glob('*.json')],key=lambda j:j['created_at'],reverse=True)
        for run in results:
            values=run['training'];run['training_points_total']=len(values)
            if len(values)>1000:
                selected={0,len(values)-1};width=math.ceil(len(values)/240)
                for start in range(0,len(values),width):
                    indices=range(start,min(start+width,len(values)))
                    for key in ('loss','validation_loss'):
                        valid=[i for i in indices if values[i].get(key) is not None]
                        if valid:selected.update([min(valid,key=lambda i:values[i][key]),max(valid,key=lambda i:values[i][key])])
                run['training']=[values[i] for i in sorted(selected)]
        return results

    def import_evaluation(self,run):
        if not isinstance(run,dict):raise ValueError('结果必须是JSON对象')
        def number(value,name,nonnegative=False):
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or (nonnegative and value<0):raise ValueError('无效数值：'+name)
            return value
        def text(value,name,limit):
            if not isinstance(value,str) or len(value)>limit:raise ValueError('无效文本：'+name)
            return value
        name=text(run.get('name',''),'name',100)
        if not name.strip():raise ValueError('请填写实验名称')
        training=run.get('training',[])
        if not isinstance(training,list) or len(training)>50000:raise ValueError('训练曲线长度无效')
        points=[];last=-1
        for row in training:
            if not isinstance(row,dict):raise ValueError('训练记录必须是对象')
            step=number(row.get('step'),'step',True)
            if int(step)!=step or step<=last:raise ValueError('训练step必须为严格递增的非负整数')
            point={'step':int(step),'loss':number(row.get('loss'),'loss')};last=step
            if row.get('validation_loss') is not None:point['validation_loss']=number(row['validation_loss'],'validation_loss')
            points.append(point)
        evaluation=run.get('evaluation',{})
        if not isinstance(evaluation,dict):raise ValueError('evaluation必须是对象')
        metrics={}
        if evaluation.get('trials') is not None or evaluation.get('successes') is not None:
            trials=number(evaluation.get('trials'),'trials',True);successes=number(evaluation.get('successes'),'successes',True)
            if int(trials)!=trials or int(successes)!=successes or successes>trials:raise ValueError('成功次数必须在总试验次数之内')
            metrics.update(trials=int(trials),successes=int(successes))
        for key in ['mean_joint_mae_deg','mean_latency_ms']:
            if evaluation.get(key) is not None:metrics[key]=number(evaluation[key],key,True)
        if evaluation.get('per_joint_mae_deg') is not None:
            values=evaluation['per_joint_mae_deg']
            if not isinstance(values,list) or len(values)!=6:raise ValueError('关节误差须包含6个值')
            metrics['per_joint_mae_deg']=[number(v,'per_joint_mae_deg',True) for v in values]
        if not points and not metrics:raise ValueError('请导入真实训练曲线或评测结果，空模板不作为实验')
        result={'id':uuid.uuid4().hex,'name':name,'checkpoint':text(run.get('checkpoint',''),'checkpoint',300),
                'dataset':text(run.get('dataset',''),'dataset',300),'created_at':now(),'training':points,'evaluation':metrics,
                'notes':text(run.get('notes',''),'notes',2000),'source':'user_import'}
        atomic_json(self.panel/'evaluations'/(result['id']+'.json'),result)
        return result

    def close(self):
        with self.lock:self.closed=True;self.capture_stop.set()
        errors=[]
        if self.motion:
            try:self.motion.close()
            except Exception as exc:errors.append('motion: '+str(exc))
        if self.capture_thread:self.capture_thread.join(timeout=12)
        if self.export_thread:self.export_thread.join(timeout=12)
        if self.gripper:
            try:self.gripper.close()
            except Exception as exc:errors.append('gripper: '+str(exc))
        self.source.close();self.lockfile.close()
        if errors:
            atomic_json(self.panel/'shutdown_error.json',{'time':now(),'errors':errors})
            raise RuntimeError('; '.join(errors))

    def check_gripper(self):
        if not self.gripper:raise Conflict('夹爪反馈模块未接入，不能采集固定张开数据')
        return verify_audit(self.gripper.require_open())

    def prepare_gripper(self,body):
        if body.get('confirmed') is not True:raise ValueError('请先现场确认夹爪周围安全')
        with self.lock:
            if self.closed or self.mode=='RECORDING' or (self.export_thread and self.export_thread.is_alive()):raise Conflict('采集或导出进行中')
            if self.motion and self.motion.status()['mode']!='DISARMED':raise Conflict('请先停止并释放机械臂手动会话')
        if not self.gripper:raise Conflict('夹爪模块未接入')
        return self.gripper.prepare_open()

    def confirm_fixed_gripper(self,body):
        with self.lock:
            if self.closed or self.mode=='RECORDING' or (self.export_thread and self.export_thread.is_alive()):raise Conflict('采集或导出进行中')
            if self.motion and self.motion.status()['mode']!='DISARMED':raise Conflict('请先停止并释放手动会话，再确认机械固定')
            if not self.gripper or not hasattr(self.gripper,'confirm'):raise ValueError('当前未选择机械固定模式')
            return self.gripper.confirm(body.get('confirmed'),body.get('fixture_locked'))

    def revoke_fixed_gripper(self):
        if not self.gripper or not hasattr(self.gripper,'revoke'):raise ValueError('当前未选择机械固定模式')
        result=self.gripper.revoke()
        if self.motion and self.motion.status()['mode'] not in ('DISARMED','FAULT'):self.motion.stop()
        return result

    def replay_start(self):
        episode_id=self.start(diagnostic=False,_motion_replay=True)['id']
        return episode_id

    def replay_finish(self,success,error=None):
        episode_id=self.replay_episode_id
        if not episode_id:return
        if self.recorder['id']==episode_id and self.capture_thread and self.capture_thread.is_alive():self.stop(_motion_replay=True)
        if episode_id and not success:
            path=self.episode_dir(episode_id)/'quality.json';q=read_json(path,{})
            q.update(status='FAIL',accepted=False,motion_completed=False)
            q.setdefault('errors',[]).append('机械臂回放未完成：'+str(error or '用户中止'))
            atomic_json(path,q)
        elif episode_id:
            path=self.episode_dir(episode_id)/'quality.json';q=read_json(path,{})
            q['motion_completed']=True
            if q.get('status')=='PASS' and not q.get('diagnostic'):
                q.update(accepted=True,review_decision='auto_accept',review_note='回放完成且质量通过，自动接受',reviewed_at=now())
            atomic_json(path,q)
        self.replay_episode_id=None

    def motion_command(self,command,body):
        if not self.motion:raise Conflict('手动控制模块尚未接入')
        if command in ('home','replay'):raise Conflict('返回与回放方式已更新，请刷新页面使用新的MoveJ/原速回放按钮')
        if command not in ('arm','heartbeat','drag_start','drag_stop','home_movej','reset','set_reset_pose','replay_recorded','stop','disarm','recover','set_parameters'):raise ValueError('无效手动控制操作')
        if command in ('arm','recover','drag_start','home_movej','reset','set_reset_pose','replay_recorded'):
            self.check_gripper()
            with self.lock:
                if self.closed or self.mode=='RECORDING' or (self.export_thread and self.export_thread.is_alive()):
                    raise Conflict('采集/导出正忙，请先结束当前数据任务')
        if command=='stop':return self.motion.stop()
        client=body.get('client_id')
        if not isinstance(client,str) or not SAFE.fullmatch(client):raise ValueError('无效控制会话编号')
        if command=='set_parameters':
            with self.lock:
                if self.closed or self.mode=='RECORDING' or (self.export_thread and self.export_thread.is_alive()):raise Conflict('采集/导出期间不能修改操作参数')
                return self.motion.set_parameters(client,body.get('parameters'))
        if command=='home_movej':return self.motion.home(client)
        if command=='replay_recorded':return self.motion.replay(client)
        if command=='arm':
            return self.motion.arm(client,confirmed=True)
        if command=='drag_start':return self.motion.drag_start(client,pre_roll_s=body.get('pre_roll_s'))
        if command=='set_reset_pose':
            with self.lock:
                if self.closed or self.mode=='RECORDING' or (self.export_thread and self.export_thread.is_alive()):
                    raise Conflict('采集/导出期间不能修改复位零位')
                return self.motion.set_reset_pose(client,body.get('source'),body.get('joints_deg'))
        return getattr(self.motion,command)(client)


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        server_version='R5Workbench/1'
        def setup(self):
            super().setup();self.connection.settimeout(15)
        def log_message(self,format,*args):
            if ' 200 ' not in (format%args):print(datetime.now().isoformat(),format%args,flush=True)
        def send(self,value,status=200,mime='application/json',cache=None):
            data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode() if mime=='application/json' else value
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control',cache or 'no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('X-Frame-Options','DENY');self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass
        def host_ok(self):
            host=self.headers.get('Host','')
            try:return urlparse('http://'+host).hostname in ('127.0.0.1','localhost','::1')
            except ValueError:return False
        def do_GET(self):
            if not self.host_ok():return self.send({'error':'仅允许本机访问'},403)
            p=urlparse(self.path);query=parse_qs(p.query);path=p.path
            try:
                if path=='/':return self.send((HERE/'dashboard/index.html').read_bytes(),mime='text/html; charset=utf-8')
                if path=='/favicon.ico':return self.send(b'',mime='image/x-icon')
                if path=='/api/robot-model':
                    data=service.robot_model()
                    self.send_response(200);self.send_header('Content-Type','application/json')
                    self.send_header('Content-Encoding','gzip');self.send_header('Content-Length',str(len(data)))
                    self.send_header('Cache-Control','private, max-age=60');self.end_headers()
                    try:self.wfile.write(data)
                    except (BrokenPipeError,ConnectionResetError):pass
                    return
                if path=='/api/status':return self.send(service.status())
                if path=='/api/episodes':return self.send({'episodes':service.episodes()})
                if path=='/api/exports':return self.send({'exports':service.exports()})
                if path=='/api/evaluations':return self.send({'evaluations':service.evaluations()})
                if path=='/api/live/frame':return self.send(service.frame(query.get('camera',[''])[0]),mime='image/jpeg')
                match=re.fullmatch(r'/api/episodes/([^/]+)(/frame)?',path)
                if match:
                    if match[2]:return self.send(service.frame(query.get('camera',[''])[0],match[1],int(query.get('index',['-1'])[0])),mime='image/jpeg',cache='private, max-age=86400')
                    return self.send(service.detail(match[1]))
                return self.send({'error':'接口不存在'},404)
            except Conflict as exc:self.send({'error':str(exc)},409)
            except (ValueError,FileNotFoundError,KeyError) as exc:self.send({'error':str(exc)},400)
            except Exception as exc:
                print('request_error',type(exc).__name__,str(exc),flush=True);self.send({'error':'读取失败，请检查服务日志'},500)
        def do_POST(self):
            if not self.host_ok() or self.headers.get('X-R5-Panel')!='1':return self.send({'error':'请求来源无效'},403)
            origin=self.headers.get('Origin')
            if origin and urlparse(origin).netloc!=self.headers.get('Host'):return self.send({'error':'不允许跨站写入'},403)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4*1024*1024:return self.send({'error':'请求为空或超过4MB'},413)
                body=json.loads(self.rfile.read(length))
                if not isinstance(body,dict):raise ValueError('请求必须为对象')
                path=urlparse(self.path).path
                if path=='/api/record/start':return self.send(service.start(body.get('diagnostic',False)))
                if path=='/api/record/stop':return self.send(service.stop())
                if path=='/api/exports':return self.send(service.start_export(body.get('name','')))
                if path=='/api/evaluations/import':return self.send(service.import_evaluation(body.get('run')))
                if path=='/api/gripper/prepare':return self.send(service.prepare_gripper(body))
                if path=='/api/gripper/fixed/confirm':return self.send(service.confirm_fixed_gripper(body))
                if path=='/api/gripper/fixed/revoke':return self.send(service.revoke_fixed_gripper())
                match=re.fullmatch(r'/api/motion/([a-z_]+)',path)
                if match:return self.send(service.motion_command(match[1],body))
                match=re.fullmatch(r'/api/episodes/([^/]+)/review',path)
                if match:return self.send(service.review(match[1],body.get('decision'),body.get('note','')))
                return self.send({'error':'接口不存在'},404)
            except Conflict as exc:self.send({'error':str(exc)},409)
            except (ValueError,KeyError) as exc:self.send({'error':str(exc)},400)
            except Exception as exc:
                print('request_error',type(exc).__name__,str(exc),flush=True);self.send({'error':'操作失败，请检查服务日志'},500)
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8765);args=p.parse_args()
    cfg=load_config();source=Observations(cfg);service=Workbench(cfg,source)
    if cfg.get('gripper_policy',{}).get('verification')=='user_configured_fixed':
        from fixed_gripper import ConfiguredFixedGuard
        service.gripper=ConfiguredFixedGuard()
    elif cfg.get('gripper_policy',{}).get('verification')=='mechanical_fixture':
        from fixed_gripper import MechanicalFixedGuard
        service.gripper=MechanicalFixedGuard(service.storage)
    else:
        from gripper_guard import GripperGuard
        service.gripper=GripperGuard(source)
    from manual_motion import ManualMotion
    service.motion=ManualMotion(source,service.storage,on_replay_start=service.replay_start,on_replay_finish=service.replay_finish,external_guard=service.check_gripper)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(service))
    def shutdown(*_):threading.Thread(target=server.shutdown,daemon=True).start()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,shutdown)
    print(f'R5 workbench: http://127.0.0.1:{args.port} (passive capture; no motion commands)',flush=True)
    try:server.serve_forever(poll_interval=.2)
    finally:server.server_close();service.close()


if __name__=='__main__':main()
