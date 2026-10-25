"""Read back a complete v2.1 dataset; validate alignment, finite states and every decoded video."""
import argparse
import json
from pathlib import Path
import av
import numpy as np
import pyarrow.parquet as pq


def validate(root, for_training=False):
    root=Path(root)
    info=json.loads((root/'meta/info.json').read_text())
    contract=json.loads((root/'meta/robot_contract.json').read_text())
    if for_training and (info.get('diagnostic_only') or contract.get('data_purpose')=='synthetic_integration_test'):
        raise ValueError('Diagnostic/test dataset cannot be used for training')
    if for_training and info.get('gripper_open_verified') is not True and not (info.get('gripper_policy_validated') is True and info.get('gripper_verification_sources')==['user_configured_fixed'] and contract.get('gripper_policy',{}).get('verification')=='user_configured_fixed'):
        raise ValueError('Training requires verified maximum-allowed-open gripper feedback')
    if info['codebase_version']!='v2.1' or info['fps']!=30:raise ValueError('Unexpected dataset version/fps')
    episodes=[json.loads(x) for x in (root/'meta/episodes.jsonl').read_text().splitlines()]
    actions_all=[];frames=0;videos=[]
    if len(episodes)!=info['total_episodes'] or not episodes:raise ValueError('Episode count mismatch')
    for i,ep in enumerate(episodes):
        if ep['episode_index']!=i:raise ValueError('Noncontiguous episode indices')
        table=pq.read_table(root/info['data_path'].format(episode_chunk=i//1000,episode_index=i))
        n=len(table);state=np.array(table['observation.state'].to_pylist());action=np.array(table['action'].to_pylist())
        if n!=ep['length'] or n<2 or state.shape!=(n,17) or action.shape!=(n,7):raise ValueError('Shape/length mismatch')
        if not np.isfinite(state).all() or not np.isfinite(action).all():raise ValueError('Nonfinite data')
        if not np.allclose(state[:,16],0) or not np.allclose(action[:,6],0):raise ValueError('Nonzero gripper')
        if not np.allclose(np.linalg.norm(state[:,12:16],axis=1),1,atol=1e-4):raise ValueError('Quaternion norm mismatch')
        targets=np.concatenate([state[1:,:6],state[-1:,:6]])
        if not np.allclose(action[:,:6],targets,atol=1e-5):raise ValueError('Action must match next state, with final hold')
        for key,expected in [('frame_index',np.arange(n)),('index',np.arange(frames,frames+n)),('episode_index',np.full(n,i)),('task_index',np.zeros(n))]:
            if not np.array_equal(table[key].to_numpy(),expected):raise ValueError('Invalid '+key)
        if not np.allclose(table['timestamp'].to_numpy(),np.arange(n)/30,atol=1e-5):raise ValueError('Invalid frame times')
        for cam in ('cam_bottom','cam_wrist'):
            path=root/info['video_path'].format(episode_chunk=i//1000,episode_index=i,video_key='observation.images.'+cam)
            with av.open(str(path)) as container:
                stream=container.streams.video[0]
                if stream.width!=640 or stream.height!=480 or float(stream.average_rate)!=30:raise ValueError('Video format mismatch')
                count=sum(1 for _ in container.decode(video=0))
                if count!=n:raise ValueError('Video/frame count mismatch')
            videos.append({'episode':i,'camera':cam,'decoded_frames':count})
        frames+=n;actions_all.append(action)
    if frames!=info['total_frames']:raise ValueError('Total frames mismatch')
    values=np.concatenate(actions_all);stats=json.loads((root/'meta/stats.json').read_text())['action']
    if not np.allclose(stats['min'],values.min(0),atol=1e-5) or not np.allclose(stats['max'],values.max(0),atol=1e-5):raise ValueError('Stale action statistics')
    return {'status':'PASS','episodes':len(episodes),'frames':frames,'videos':videos,
            'diagnostic_only':bool(info.get('diagnostic_only')),'gripper_open_verified':bool(info.get('gripper_open_verified')),
            'gripper_verification_sources':info.get('gripper_verification_sources',[]),'gripper_device_feedback_verified':bool(info.get('gripper_device_feedback_verified')),
            'official_training_loader_verified':False,'physical_task_success_verified':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--for-training',action='store_true');a=p.parse_args()
    print(json.dumps(validate(a.dataset,a.for_training),indent=2))
