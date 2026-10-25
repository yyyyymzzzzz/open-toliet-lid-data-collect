"""Export accepted R5 episodes into the original openpi LeRobot v2.1 contract."""
import argparse
import json
from pathlib import Path
import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation, Slerp
from observations import decode_image
from record_episode import quality
from gripper_contract import verify_rows

STATE_NAMES = ['j0','j1','j2','j3','j4','j5','x','y','z','rx','ry','rz','qw','qx','qy','qz','gripper']
ACTION_NAMES = ['j0','j1','j2','j3','j4','j5','gripper']


def resample(rows, fps=30):
    source_ns = np.asarray([r['timing']['source_ns']['cam_bottom'] for r in rows],np.int64)
    t = (source_ns-source_ns[0])/1e9
    if len(t)<2 or np.any(np.diff(t)<=0) or np.max(np.diff(t))>.1:
        raise ValueError('Invalid or gapped source times')
    grid = np.arange(int(np.floor(t[-1]*fps))+1)/fps
    original = np.asarray([r['state'] for r in rows],np.float64)
    if original.shape != (len(rows),17) or not np.isfinite(original).all():
        raise ValueError('Invalid state')
    if not np.allclose(original[:,16],0):raise ValueError('Gripper must be zero')
    if not np.allclose(np.linalg.norm(original[:,12:16],axis=1),1,atol=.01):
        raise ValueError('Invalid quaternion')
    state=np.zeros((len(grid),17),np.float32)
    for j in range(9):state[:,j]=np.interp(grid,t,original[:,j])
    rotations=Slerp(t,Rotation.from_quat(original[:,[13,14,15,12]]))(grid)
    q=rotations.as_quat();q[q[:,3]<0]*=-1
    state[:,9:12]=rotations.as_euler('xyz');state[:,12:16]=q[:,[3,0,1,2]]
    targets=np.concatenate([state[1:],state[-1:]],axis=0)
    actions=np.concatenate([targets[:,:6],targets[:,16:17]],axis=1)
    nearest={}
    for cam in ('cam_bottom','cam_wrist'):
        stamps=np.asarray([r['timing']['source_ns'][cam] for r in rows],np.int64)
        ct=(stamps-source_ns[0])/1e9
        if np.any(np.diff(ct)<=0):raise ValueError('Non-monotonic camera: '+cam)
        idx=np.searchsorted(ct,grid).clip(0,len(ct)-1)
        left=np.maximum(idx-1,0)
        idx=np.where(np.abs(ct[left]-grid)<np.abs(ct[idx]-grid),left,idx)
        if np.max(np.abs(ct[idx]-grid))>.05:raise ValueError('Camera alignment exceeds 50ms')
        nearest[cam]=idx
    return state,actions,grid,nearest


def stats(x):
    return {k:np.asarray(v).tolist() for k,v in {
        'min':x.min(0),'max':x.max(0),'mean':x.mean(0),'std':x.std(0),'count':[len(x)]}.items()}


def export(episodes, target, *, enforce_gripper=True):
    target.mkdir(parents=True,exist_ok=False)
    (target/'meta').mkdir()
    all_states=[];all_actions=[];entries=[];epstats=[];provenance=[];index=0
    reference=None; image_stats={};gripper_audits=[]
    for ep,source in enumerate(episodes):
        cfg=json.loads((source/'config.json').read_text())
        if reference is None:reference=cfg
        keys=['cameras','base_frame','tcp_frame','image_size','image_transform','state_semantics','fps','task','gripper_policy']
        if any(cfg.get(k)!=reference.get(k) for k in keys):raise ValueError('Mixed robot/camera/gripper contracts')
        rows=[json.loads(line) for line in (source/'frames.jsonl').read_text().splitlines()]
        audit=quality(rows,cfg['fps'])
        if audit['status']!='PASS':raise ValueError(str(audit))
        if enforce_gripper:
            gripper_audit=verify_rows(rows)
            expected=cfg.get('gripper_policy',{}).get('verification','device_feedback')
            if gripper_audit['verification_source']!=expected:raise ValueError('夹爪确认方式与样本配置不一致')
            gripper_audits.append(dict(episode_index=ep,**gripper_audit))
        state,actions,t,images=resample(rows,cfg['fps'])
        n=len(t);chunk=ep//1000
        path=target/f'data/chunk-{chunk:03d}/episode_{ep:06d}.parquet'
        path.parent.mkdir(parents=True,exist_ok=True)
        columns={'observation.state':pa.array(state.tolist(),pa.list_(pa.float32(),17)),
                 'action':pa.array(actions.tolist(),pa.list_(pa.float32(),7)),
                 'timestamp':pa.array(t,pa.float32()), 'frame_index':pa.array(np.arange(n),pa.int64()),
                 'episode_index':pa.array(np.full(n,ep),pa.int64()),
                 'index':pa.array(np.arange(index,index+n),pa.int64()),
                 'task_index':pa.array(np.zeros(n,dtype=np.int64))}
        pq.write_table(pa.table(columns),path)
        es={'observation.state':stats(state),'action':stats(actions)}
        for cam,indices in images.items():
            video=target/f'videos/chunk-{chunk:03d}/observation.images.{cam}/episode_{ep:06d}.mp4'
            video.parent.mkdir(parents=True,exist_ok=True)
            sums=np.zeros(3);squares=np.zeros(3);lo=np.ones(3);hi=np.zeros(3);pixels=0
            with av.open(str(video),'w') as container:
                stream=container.add_stream('libx264',rate=cfg['fps'])
                stream.width=cfg['image_size'][1];stream.height=cfg['image_size'][0]
                stream.pix_fmt='yuv420p';stream.options={'crf':'18','preset':'fast'}
                for i,ri in enumerate(indices):
                    data=(source/cam/f"{rows[int(ri)]['frame_index']:06d}.jpg").read_bytes()
                    rgb=decode_image(data,cfg['image_size'])
                    frame=av.VideoFrame.from_ndarray(rgb,format='rgb24');frame.pts=i
                    for packet in stream.encode(frame):container.mux(packet)
                    sample=rgb[::8,::8].reshape(-1,3).astype(np.float64)/255
                    sums+=sample.sum(0);squares+=(sample**2).sum(0);pixels+=len(sample)
                    lo=np.minimum(lo,sample.min(0));hi=np.maximum(hi,sample.max(0))
                for packet in stream.encode():container.mux(packet)
            with av.open(str(video)) as container:
                if sum(1 for _ in container.decode(video=0))!=n:raise RuntimeError('Video frame count mismatch')
            image_key='observation.images.'+cam
            es[image_key]={'min':lo[:,None,None].tolist(),'max':hi[:,None,None].tolist(),
                           'mean':(sums/pixels)[:,None,None].tolist(),
                           'std':np.sqrt(np.maximum(0,squares/pixels-(sums/pixels)**2))[:,None,None].tolist(),
                           'count':[n]}
            agg=image_stats.setdefault(image_key,{'sum':np.zeros(3),'sq':np.zeros(3),'pixels':0,'min':np.ones(3),'max':np.zeros(3)})
            agg['sum']+=sums;agg['sq']+=squares;agg['pixels']+=pixels
            agg['min']=np.minimum(agg['min'],lo);agg['max']=np.maximum(agg['max'],hi)
        entries.append({'episode_index':ep,'tasks':[cfg['task']],'length':n})
        epstats.append({'episode_index':ep,'stats':es})
        provenance.append({'episode_index':ep,'raw_path':str(source.resolve()),'source_quality':audit,
                           'resampling':'30Hz joint/xyz linear, quaternion SLERP; nearest image <=50ms'})
        all_states.append(state);all_actions.append(actions);index+=n
    if reference is None:raise ValueError('No episodes')
    features={'observation.state':{'dtype':'float32','shape':[17],'names':STATE_NAMES},
              'action':{'dtype':'float32','shape':[7],'names':ACTION_NAMES}}
    for k in ['timestamp','frame_index','episode_index','index','task_index']:
        features[k]={'dtype':'float32' if k=='timestamp' else 'int64','shape':[1],'names':None}
    for cam in reference['cameras']:
        features['observation.images.'+cam]={'dtype':'video','shape':[480,640,3],'names':['height','width','channels'],
            'info':{'video.fps':30,'video.codec':'h264','video.pix_fmt':'yuv420p','video.is_depth_map':False,'has_audio':False}}
    info={'codebase_version':'v2.1','robot_type':'realman_eco63_r5','total_episodes':len(entries),
          'gripper_policy_validated':enforce_gripper,
          'gripper_open_verified':enforce_gripper and all(a['fixed_max_open_verified'] for a in gripper_audits),'diagnostic_only':not enforce_gripper,
          'gripper_verification_sources':sorted(set(a['verification_source'] for a in gripper_audits)),
          'gripper_device_feedback_verified':bool(gripper_audits) and all(a['device_feedback_verified'] for a in gripper_audits),
          'total_frames':index,'total_tasks':1,'total_videos':len(entries)*2,'total_chunks':(len(entries)+999)//1000,
          'chunks_size':1000,'fps':30,'splits':{'train':f'0:{len(entries)}'},
          'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
          'video_path':'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4','features':features}
    total={'observation.state':stats(np.concatenate(all_states)), 'action':stats(np.concatenate(all_actions))}
    for key,s in image_stats.items():
        mean=s['sum']/s['pixels']
        total[key]={'min':s['min'][:,None,None].tolist(),'max':s['max'][:,None,None].tolist(),
                    'mean':mean[:,None,None].tolist(),
                    'std':np.sqrt(np.maximum(0,s['sq']/s['pixels']-mean**2))[:,None,None].tolist(),'count':[index]}
    for name,items in [('episodes',entries),('episodes_stats',epstats),('tasks',[{'task_index':0,'task':reference['task']}])]:
        (target/'meta'/f'{name}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in items))
    (target/'meta/stats.json').write_text(json.dumps(total,indent=2,allow_nan=False)+'\n')
    (target/'meta/robot_contract.json').write_text(json.dumps(reference,indent=2)+'\n')
    (target/'meta/provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (target/'meta/gripper_audit.json').write_text(json.dumps(gripper_audits,indent=2)+'\n')
    # Completion marker written last; partial exports have no info.json.
    (target/'meta/info.json').write_text(json.dumps(info,indent=2)+'\n')
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sources=[]
    for path in sorted(args.episodes.glob('*/quality.json')):
        q=json.loads(path.read_text())
        if q.get('accepted') is True and q.get('status')=='PASS' and not q.get('diagnostic'):
            sources.append(path.parent)
    if not sources:raise SystemExit('No accepted, quality-passing demonstrations; diagnostics are excluded')
    info=export(sources,args.output)
    print(json.dumps({'output':str(args.output),'episodes':info['total_episodes'],'frames':info['total_frames']}))


if __name__=='__main__':main()
