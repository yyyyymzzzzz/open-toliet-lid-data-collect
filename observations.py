"""Passive ROS2 observation adapter. No publishers, services, or robot SDK calls."""
from __future__ import annotations
import json
import os
import threading
import time
from collections import Counter, deque
from pathlib import Path
import numpy as np


def load_config(path=None):
    return json.loads(Path(path or Path(__file__).with_name('config.json')).read_text())


def state_vector(joints_rad, xyz, xyzw):
    from scipy.spatial.transform import Rotation
    j, p, q = (np.asarray(v, dtype=np.float64) for v in (joints_rad, xyz, xyzw))
    if j.shape != (6,) or p.shape != (3,) or q.shape != (4,):
        raise ValueError('Invalid joint/pose dimensions')
    if not all(np.isfinite(v).all() for v in (j, p, q)):
        raise ValueError('Non-finite robot feedback')
    if abs(np.linalg.norm(q)-1) > .01:
        raise ValueError('Invalid TF quaternion')
    q = q / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return np.concatenate((np.rad2deg(j), p, Rotation.from_quat(q).as_euler('xyz'),
                           q[[3, 0, 1, 2]], [0])).astype(np.float32)


def decode_image(data, size):
    import cv2
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError('Image decode failed')
    h, w = size
    scale = min(h/bgr.shape[0], w/bgr.shape[1])
    nh, nw = max(1, round(bgr.shape[0]*scale)), max(1, round(bgr.shape[1]*scale))
    image = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    y, x = (h-nh)//2, (w-nw)//2
    out[y:y+nh, x:x+nw] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return out


def validate_timing(now_ns, times, cfg):
    # All stamps below share ROS system time. Monotonic receipt time is checked separately.
    if any(t <= 0 or now_ns-t < -50_000_000 or now_ns-t > cfg['max_age_s']*1e9 for t in times.values()):
        raise ValueError('Missing, future, or stale source timestamps')
    cameras = [times[k] for k in cfg['cameras']]
    if max(cameras)-min(cameras) > cfg['max_camera_skew_s']*1e9:
        raise ValueError('Camera synchronization gap')
    if max(abs(times['joints']-t) for t in cameras) > cfg['max_joint_camera_skew_s']*1e9:
        raise ValueError('Joint/camera synchronization gap')
    if abs(times['tf']-times['joints']) > cfg['max_tf_joint_skew_s']*1e9:
        raise ValueError('TF/joint synchronization gap')


class Observations:
    def __init__(self, config):
        self.cfg = config
        os.environ['ROS_DOMAIN_ID'] = str(config['ros_domain_id'])
        os.environ['ROS_LOCALHOST_ONLY'] = '1' if config['ros_localhost_only'] else '0'
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState, CompressedImage
        from tf2_ros import Buffer, TransformListener
        self.rclpy = rclpy
        rclpy.init()
        self.node = rclpy.create_node('pi05_r5_observations_' + str(os.getpid()))
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self.node, spin_thread=False)
        self.lock = threading.Lock()
        self.samples = {}
        self.history = {k: deque(maxlen=80) for k in ['joints', *config['cameras']]}
        self.counts = Counter()
        self.errors = Counter()
        self.stopping = threading.Event()
        self.thread_error = None
        self.subs = [self.node.create_subscription(JointState, config['joints_topic'],
                     self._joint, qos_profile_sensor_data)]
        for key, cam in config['cameras'].items():
            self.subs.append(self.node.create_subscription(CompressedImage, cam['topic'],
                             lambda msg, k=key: self._image(k, msg), qos_profile_sensor_data))
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    @staticmethod
    def stamp(message):
        t = message.header.stamp
        return t.sec*1_000_000_000 + t.nanosec

    def _store(self, key, value, stamp):
        with self.lock:
            self.counts[key] += 1
            sample = (value, stamp, time.monotonic_ns())
            self.samples[key] = sample
            self.history[key].append(sample)

    def _joint(self, msg):
        try:
            mapping = dict(zip(msg.name, msg.position))
            value = np.asarray([mapping[n] for n in self.cfg['joint_names']], np.float64)
            if not np.isfinite(value).all():
                raise ValueError('NaN joint feedback')
            self._store('joints', value, self.stamp(msg))
        except (KeyError, ValueError) as exc:
            with self.lock:
                self.samples.pop('joints', None)
                self.history['joints'].clear()
                self.errors[str(exc)] += 1

    def _image(self, key, msg):
        self._store(key, bytes(msg.data), self.stamp(msg))

    def _spin(self):
        try:
            while not self.stopping.is_set():
                self.rclpy.spin_once(self.node, timeout_sec=.02)
        except Exception as exc:
            self.thread_error = exc

    def joint_feedback(self):
        """Latest joint feedback for control; independent of camera alignment."""
        if self.thread_error is not None:raise ValueError('ROS subscriber failed')
        with self.lock:
            if 'joints' not in self.samples:raise ValueError('Missing joint feedback')
            q,stamp,receipt=self.samples['joints']
            q=np.rad2deg(q).tolist()
        now=self.node.get_clock().now().nanoseconds
        if not 0<=time.monotonic_ns()-receipt<=150_000_000 or not 0<=now-stamp<=150_000_000:
            raise ValueError('Joint feedback older than 150 ms or timestamp in future')
        return q,stamp

    def snapshot(self, decode=True):
        if self.thread_error is not None:
            raise RuntimeError('ROS subscriber failed') from self.thread_error
        keys = ['joints', *self.cfg['cameras']]
        with self.lock:
            if any(k not in self.samples for k in keys):
                raise ValueError('Missing topics: ' + ', '.join(k for k in keys if k not in self.samples))
            # Anchor on the older camera's newest stamp, then take nearest camera/joint samples.
            anchor = min(self.samples[k][1] for k in self.cfg['cameras'])
            s = {k: min(self.history[k], key=lambda x: abs(x[1]-anchor)) for k in keys}
        mono = time.monotonic_ns()
        if any(mono-v[2] > self.cfg['max_age_s']*1e9 for v in s.values()):
            raise ValueError('Stale ROS receipt')
        import rclpy.time
        try:
            # Query the same timestamp as joint feedback, not an unlabelled mutable robot_state pose.
            tf = self.tf.lookup_transform(self.cfg['base_frame'], self.cfg['tcp_frame'],
                                         rclpy.time.Time(nanoseconds=s['joints'][1]))
        except Exception as exc:
            raise ValueError('TF unavailable at joint timestamp: ' + str(exc)) from exc
        p, q = tf.transform.translation, tf.transform.rotation
        state = state_vector(s['joints'][0], [p.x,p.y,p.z], [q.x,q.y,q.z,q.w])
        stamps = {k:v[1] for k,v in s.items()}
        stamps['tf'] = self.stamp(tf)
        validate_timing(self.node.get_clock().now().nanoseconds, stamps, self.cfg)
        obs = {'observation.state':state}
        for key in self.cfg['cameras']:
            obs['observation.images.'+key] = decode_image(s[key][0], self.cfg['image_size']) if decode else s[key][0]
        return obs, {'source_ns':stamps, 'receive_mono_ns':{k:v[2] for k,v in s.items()},
                     'snapshot_mono_ns':mono, 'snapshot_unix_ns':time.time_ns()}

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError('ROS thread did not stop')
        self.node.destroy_node()
        self.rclpy.shutdown()


def camera_snapshot(source, key):
    with source.lock:
        if key not in source.samples:
            raise ValueError('No camera sample: '+key)
        data, stamp, receipt = source.samples[key]
    if time.monotonic_ns()-receipt > source.cfg['max_age_s']*1e9:
        raise ValueError('Stale camera: '+key)
    return decode_image(data, source.cfg['image_size'])
