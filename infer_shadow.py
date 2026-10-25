"""π0.5 WebSocket inference using R5 observations. Never sends robot commands."""
import argparse
import json
import time
import numpy as np
from observations import Observations,load_config
from pi05_joint_core import BoundedClient,make_request,validate_actions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8000)
    p.add_argument('--probe-only',action='store_true',help='Synthetic inputs; no ROS or SDK connection')
    p.add_argument('--count',type=int,default=1)
    args=p.parse_args()
    if args.count<1:p.error('count must be positive')
    source=None;client=None
    try:
        client=BoundedClient(args.host,args.port,timeout_s=5)
        if not args.probe_only:
            source=Observations(load_config());time.sleep(2)
        for _ in range(args.count):
            if source is None:
                state=np.zeros(17,np.float32);state[12]=1
                obs={'observation.state':state,
                    'observation.images.cam_bottom':np.zeros((480,640,3),np.uint8),
                    'observation.images.cam_wrist':np.zeros((480,640,3),np.uint8)}
            else:obs,_=source.snapshot()
            start=time.monotonic()
            actions=validate_actions(client.infer(make_request(obs))['actions'])
            if not np.allclose(actions[:,6],0,atol=1e-5):raise ValueError('Nonzero gripper action')
            print(json.dumps({'mode':'SHADOW','shape':list(actions.shape),'latency_ms':(time.monotonic()-start)*1000,
                              'first_joints_deg':actions[0,:6].tolist(),'motion_commands_sent':0}))
    finally:
        if client:client.close()
        if source:source.close()


if __name__=='__main__':main()
