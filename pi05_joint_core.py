"""Dependency-light contracts, bounded transport and time-aligned pi05 planning."""
from __future__ import annotations
import math
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

TASK = 'Open the toilet lid'
ASSET_ID = 'local/open_toilet_lid_joint_abs_v21'
PINNED_COMMIT = '215abfb217dbac7d5f1273282331b9b1866c0479'
CONTRACT = dict(protocol='pi05_joint_absolute_v1', model='pi05', action_semantics='absolute_joint_target',
                action_units='degrees', state_dim=17, action_dim=7, action_horizon=16, fps=30,
                cameras=['cam_bottom', 'cam_wrist'], image_format='RGB_uint8_HWC',
                state_layout=['j0','j1','j2','j3','j4','j5','x','y','z','rx','ry','rz','qw','qx','qy','qz','gripper'],
                euler_units='radians', xyz_units='metres', task=TASK)

def validate_metadata(metadata):
    for key, expected in CONTRACT.items():
        if metadata.get(key) != expected:
            raise ValueError(f'Server contract mismatch: {key}={metadata.get(key)!r}, expected {expected!r}')

def validate_request(request):
    state = np.asarray(request['observation/state'])
    if state.shape != (17,) or not np.isfinite(state).all():
        raise ValueError('Expected finite 17-D raw state (joints deg, XYZ m, Euler rad, quaternion wxyz, gripper)')
    if not np.isclose(state[16], 0, atol=1e-6):
        raise ValueError('Training used constant-zero gripper state')
    if not 0.95 <= np.linalg.norm(state[12:16]) <= 1.05:
        raise ValueError('Invalid FK quaternion; expected wxyz unit quaternion')
    if request.get('prompt') != TASK:
        raise ValueError(f'This task was trained with prompt {TASK!r}')
    for key in ('observation/image','observation/wrist_image'):
        frame = np.asarray(request[key])
        if frame.dtype != np.uint8 or frame.shape != (480,640,3):
            raise ValueError(f'{key} must be RGB uint8 (480,640,3), got {frame.shape}/{frame.dtype}')

def make_request(observation, *, color_order='rgb'):
    if color_order not in ('rgb','bgr'):
        raise ValueError('color_order must be rgb or bgr')
    request = {'observation/state': np.asarray(observation['observation.state'], dtype=np.float32).copy(), 'prompt': TASK}
    for src,dst in [('cam_bottom','observation/image'),('cam_wrist','observation/wrist_image')]:
        image = np.asarray(observation[f'observation.images.{src}'])
        # No resize, float normalization, angle conversion or state normalization here.
        request[dst] = np.ascontiguousarray(image[..., ::-1] if color_order == 'bgr' else image).copy()
    validate_request(request)
    return request

def validate_actions(actions):
    values = np.asarray(actions, dtype=np.float32)
    if values.shape != (16,7) or not np.isfinite(values).all():
        raise ValueError(f'Expected finite, already unnormalized absolute actions (16,7), got {values.shape}')
    return values.copy()

class BoundedClient:
    """Official openpi msgpack protocol with finite connect/receive timeouts."""
    def __init__(self, host='127.0.0.1', port=8000, timeout_s=2.0):
        import websockets.sync.client
        from openpi_client import msgpack_numpy
        if timeout_s <= 0 or not math.isfinite(timeout_s):
            raise ValueError('timeout_s must be positive and finite')
        self.timeout = timeout_s
        self.codec = msgpack_numpy
        self.packer = msgpack_numpy.Packer()
        self.ws = websockets.sync.client.connect(f'ws://{host}:{port}', compression=None,
            max_size=16*1024*1024, open_timeout=timeout_s, close_timeout=1)
        try:
            self.metadata = msgpack_numpy.unpackb(self.ws.recv(timeout=timeout_s))
            validate_metadata(self.metadata)
        except BaseException:
            self.ws.close(); raise
    def infer(self, request):
        self.ws.send(self.packer.pack(request))
        response = self.ws.recv(timeout=self.timeout)
        if isinstance(response, str):
            raise RuntimeError('Policy server returned an error: '+response[:500])
        return self.codec.unpackb(response)
    def close(self):
        self.ws.close()

class TimedPlanner:
    """One in-flight RPC, absolute-time plan indexing, no stale reset results."""
    def __init__(self, client, *, prefetch_steps=8, blend_steps=0, max_plan_gap_s=0.75, clock=time.monotonic, action_hz=30):
        if not 1 <= prefetch_steps <= 15 or not 0 <= blend_steps <= 15:
            raise ValueError('prefetch_steps must be 1..15; blend_steps 0..15')
        if not math.isfinite(max_plan_gap_s) or max_plan_gap_s <= 0:
            raise ValueError('max_plan_gap_s must be finite and positive')
        if not math.isfinite(action_hz) or not 1 <= action_hz <= 30:
            raise ValueError("action_hz must be in [1,30]")
        self.action_hz = float(action_hz)
        self.client, self.clock = client, clock
        self.prefetch_steps, self.blend_steps, self.max_gap = prefetch_steps, blend_steps, max_plan_gap_s
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pi05-rpc')
        self.future = None
        self.generation = 0
        self.plan = None
        self.anchor = 0.0
        self.gap_since = None
        self.last_infer_s = 0.0
        self.last_skip = 0
        self.chunks = 0
        self.expired = 0
        self.error = None
    def index(self, now, anchor):
        return max(0, int(math.floor((now-anchor)*self.action_hz + 1e-7)))
    @property
    def remaining(self):
        return 0 if self.plan is None else max(0,16-self.index(self.clock(),self.anchor))
    @property
    def needs_observation(self):
        # Adaptive prefetch includes observed total network/model latency.
        threshold = min(15,max(self.prefetch_steps,math.ceil(self.last_infer_s*self.action_hz)+2))
        return self.future is None and self.remaining <= threshold
    def reset(self):
        self.generation += 1
        self.plan = None
        self.gap_since = None
        self.error = None
        self.chunks = self.expired = self.last_skip = 0
        # An in-flight call retains its generation until poll discards it.
    def request(self, request, captured_at):
        if not self.needs_observation:
            return False
        # Inputs are copied by make_request; worker owns only its snapshot.
        self.request_generation = self.generation
        self.request_anchor = captured_at
        self.future = self.pool.submit(self.client.infer, request)
        return True
    def install(self, actions, anchor, now):
        actions = validate_actions(actions)
        skip = self.index(now,anchor)
        self.last_skip = min(skip,16)
        if skip >= 16:
            self.expired += 1
            return False
        # Blend only predictions for the SAME physical timestamp, not adjacent frames.
        if self.plan is not None and self.blend_steps:
            for j in range(self.blend_steps):
                i = skip+j
                if i >= 16: break
                old_i = self.index(anchor+i/self.action_hz,self.anchor)
                if anchor+i/self.action_hz < self.anchor or old_i >= 16: break
                weight = (j+1)/(self.blend_steps+1)
                actions[i] = (1-weight)*self.plan[old_i]+weight*actions[i]
        self.plan,self.anchor = actions,anchor
        self.chunks += 1
        return True
    def poll(self):
        if self.future is None or not self.future.done(): return
        future,self.future = self.future,None
        if self.request_generation != self.generation:
            # Consume both success and failure from the previous episode without applying it.
            try: future.result()
            except Exception: pass
            return
        now = self.clock()
        self.last_infer_s = now-self.request_anchor
        try: self.install(future.result()['actions'],self.request_anchor,now)
        except Exception as exc: self.error = exc
    def action(self):
        if self.error is not None: raise RuntimeError(f'pi05 inference failed: {self.error}')
        now = self.clock()
        i = self.index(now,self.anchor)
        if self.plan is not None and i < 16:
            self.gap_since = None
            return self.plan[i].copy()
        if self.gap_since is None: self.gap_since = now
        if now-self.gap_since > self.max_gap:
            raise RuntimeError('No fresh pi05 action plan; stopped instead of replaying stale targets')
        return None
    def close(self):
        self.reset()
        self.client.close()  # Interrupt a blocked recv before joining worker.
        self.pool.shutdown(wait=True,cancel_futures=True)


class ChunkPausePlanner(TimedPlanner):
    """Sequential full chunks; observe AFTER holding, never reuse a pre-pause plan.

    action() is called only when the client can commit an action. The control
    loop and CANFD heartbeat remain live while the planner returns None.
    """
    def __init__(self, client, *, chunk_pause_s, **kwargs):
        if not math.isfinite(chunk_pause_s) or chunk_pause_s <= 0:
            raise ValueError('chunk_pause_s must be positive and finite')
        super().__init__(client, **kwargs)
        self.chunk_pause_s = chunk_pause_s
        self.cursor = 0
        self.pause_until = 0.0
    @property
    def remaining(self):
        return 0 if self.plan is None else 16-self.cursor
    @property
    def pause_remaining(self):
        return max(0.0,self.pause_until-self.clock())
    @property
    def needs_observation(self):
        return self.future is None and self.plan is None and self.clock() >= self.pause_until
    def reset(self):
        super().reset()
        self.cursor = 0
        self.pause_until = 0.0
    def install(self, actions, anchor, now):
        if now-anchor > self.max_gap:
            raise RuntimeError('Chunk inference exceeded max-plan-gap-s; stale result rejected')
        self.plan = validate_actions(actions)
        self.cursor = 0
        self.anchor = now
        self.gap_since = None
        self.last_skip = 0
        self.chunks += 1
        return True
    def action(self):
        if self.error is not None:
            raise RuntimeError(f'pi05 inference failed: {self.error}')
        now=self.clock()
        if self.plan is not None:
            action=self.plan[self.cursor].copy()
            self.cursor+=1
            if self.cursor==16:
                self.plan=None
                # Last action receives its full execution period, then the pause.
                self.pause_until=now+1/self.action_hz+self.chunk_pause_s
                self.gap_since=None
            return action
        if now < self.pause_until:
            return None
        if self.gap_since is None: self.gap_since=now
        if now-self.gap_since > self.max_gap:
            raise RuntimeError('No fresh pi05 chunk after pause')
        return None
