#!/usr/bin/env python3
"""
Realman ECO63F — open_toilet_lid 拖动示教采集工具

纯键盘控制，无手柄、无夹爪依赖。

状态机 (4 状态)：
  IDLE → [Enter] → RECORDING → [Enter] → 自动归位 + 保存 CSV → [r] → REPLAYING → REVIEW
  任意状态 [q] 安全退出

键盘按键：
  Enter       开始/结束拖动示教
  h           IDLE 下归位到 init_home
  r           确认回放 / REVIEW 下再次回放
  s           REVIEW 下保存到 LeRobot dataset
  d           REVIEW 下丢弃 / RECORDING 下放弃
  b           RECORDING 下放弃（同 d）/ REPLAYING 下中止回放
  q           安全退出

用法：
  python collect_open_toilet_lid.py
  python collect_open_toilet_lid.py --ip 192.168.1.18 --freq 30
  python collect_open_toilet_lid.py --repo-id local/open_toilet_lid_v1 --num-episodes 50
  python collect_open_toilet_lid.py --init-home 0 10 -90 0 90 0
  python collect_open_toilet_lid.py --no-cameras

────────────────────────────────────────────────────────────────
observation.state（17 维，float32）
  [0:6]   j0~j5     关节角，单位 deg
  [6:9]   x, y, z   末端位置，单位 m
  [9:12]  rx,ry,rz  末端姿态欧拉角 XYZ，单位 rad
  [12:16] qw,qx,qy,qz  末端姿态四元数
  [16]    gripper       夹爪归一化位置 [0,1]（固定为 0）

observation.force（6 维，float32）
  [0:3]   Fx, Fy, Fz  末端力，单位 N（已重力补偿）
  [3:6]   Mx, My, Mz  末端力矩，单位 N·m

action（7 维，float32）— 末端位姿 delta + 夹爪归一化绝对值
  [0:3]   dx, dy, dz          末端位置变化量 (m)
  [3:6]   drx, dry, drz       末端姿态变化量 (rad)
  [6]     gripper              夹爪归一化绝对位置 [0,1]（固定为 0）
────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import atexit
import csv
import json
import os

# 离线模式 —— 阻止 HuggingFace Hub 联网（Jetson 无外网）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import queue
import select
import shutil
import signal
import sys
import termios
import threading
import time
import traceback
import tty
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

try:
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
except ImportError:
    print("ERROR: Robotic_Arm SDK 未安装。请执行: pip install Robotic_Arm")
    sys.exit(1)

from drag_teach_utils import (
    setup_tool_frame, move_to_home, replay_csv,
    load_recording, CSV_HEADER,
    c, ok, warn, err, info, banner,
    _BOLD, _GREEN, _YELLOW, _RED, _CYAN, _GRAY,
)

# ── Keyboard input ────────────────────────────────────────────────────────────

_orig_terminal_settings = None


def _init_keyboard() -> None:
    """Switch terminal to cbreak mode for non-blocking raw key reads.
    Saves original terminal settings for later restoration."""
    global _orig_terminal_settings
    _orig_terminal_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())


def _restore_keyboard() -> None:
    """Restore terminal to its original settings."""
    global _orig_terminal_settings
    if _orig_terminal_settings is not None:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _orig_terminal_settings)


def read_key_raw() -> str | None:
    """Non-blocking single-character read. Returns None if no key pressed."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


# ── State machine ─────────────────────────────────────────────────────────────

class State(Enum):
    IDLE      = auto()
    RECORDING = auto()
    REPLAYING = auto()
    REVIEW    = auto()


# ── 后台保存 ──────────────────────────────────────────────────────────────────

_save_queue: "queue.Queue[dict]" = queue.Queue()
_bg_save = {
    "running":    False,
    "done":       False,
    "progress":   "",
    "success":    False,
    "err_msg":    "",
    "episode":    0,
    "frames":     0,
    "queue_len":  0,
}


def _save_worker(dataset, dataset_root: Path | None = None):
    """后台保存 worker：循环从队列取任务，串行写入 dataset。"""
    while True:
        job = _save_queue.get()
        if job is None:
            break

        frames      = job["frames"]
        task        = job["task"]
        ep_num      = job["ep_num"]
        n_frames    = job["n_frames"]
        episode_dir = job.get("episode_dir")  # may be None for CSV-only recordings

        _bg_save["running"] = True
        _bg_save["done"]    = False
        _bg_save["episode"] = ep_num
        _bg_save["frames"]  = n_frames
        _bg_save["queue_len"] = _save_queue.qsize()
        _bg_save["progress"] = f"开始保存 ep{ep_num} ({n_frames} 帧)..."

        def _update_progress(msg: str):
            _bg_save["progress"] = msg

        try:
            ok_flag, msg = save_lerobot_episode(
                frames, dataset, task,
                progress_cb=_update_progress,
                episode_dir=Path(episode_dir) if episode_dir else None,
                dataset_root=dataset_root,
            )
            _bg_save["success"] = ok_flag
            _bg_save["err_msg"] = msg if not ok_flag else ""
            _bg_save["progress"] = msg if ok_flag else f"✗ {msg}"
        except Exception as exc:
            traceback.print_exc()
            _bg_save["success"] = False
            _bg_save["err_msg"] = str(exc)
            _bg_save["progress"] = f"✗ 后台保存异常: {exc}"
        finally:
            _bg_save["running"] = False
            _bg_save["done"]    = True
            _bg_save["queue_len"] = _save_queue.qsize()


_STATE_LABEL = {
    State.IDLE:      c("IDLE      ", _GRAY),
    State.RECORDING: c("RECORDING ", _GREEN + "\033[5m"),
    State.REPLAYING: c("REPLAYING ", _YELLOW),
    State.REVIEW:    c("REVIEW    ", _YELLOW),
}

_STATE_HINTS = {
    State.IDLE:      "[Enter] 开始拖动示教  |  [h] 归位到 init_home  |  [q] 退出",
    State.RECORDING: "[Enter] 结束示教+自动归位  |  [b] 放弃  |  [q] 退出",
    State.REPLAYING: "[b] 中止回放  （相机+关节数据采集中）",
    State.REVIEW:    "[s] 保存  |  [d] 丢弃  |  [r] 再次归位回放  |  [q] 退出",
}

# ── Display ───────────────────────────────────────────────────────────────────

_N_LINES = 0
_key_feedback: str = ""


def _up(n: int) -> None:
    sys.stdout.write(f"\033[{n}A")


def _eol() -> None:
    sys.stdout.write("\033[2K")


def render_hud(
    state: State,
    arm_pose: list | None,
    arm_joints: list | None,
    frame_count: int,
    elapsed_s: float,
    freq_actual: float,
    episode_idx: int,
    total_episodes: int,
    first_hud: bool,
    cameras_ok: bool,
    last_event: str = "",
    pending_replay: bool = False,
    key_feedback: str = "",
) -> None:
    global _N_LINES
    lines: list[str] = []
    W = 68

    def div(ch="─"): lines.append(ch * W)

    now = datetime.now().strftime("%H:%M:%S")

    div("═")
    lines.append(
        f"  {c('Realman ECO63F — 拖动示教 + LeRobot Dataset', _BOLD)}   {now}"
    )
    hz_col = _GREEN if freq_actual >= 25 else _YELLOW
    cam_str = c("相机 ✓", _GREEN) if cameras_ok else c("相机 ✗", _RED)
    lines.append(
        f"  状态: {_STATE_LABEL[state]}   "
        f"频率: {c(f'{freq_actual:.1f} Hz', hz_col)}   {cam_str}"
    )
    if pending_replay:
        lines.append(f"  {c('>> 已归位，按 [r] 开始回放，[b] 放弃 <<', _YELLOW + '\033[5m')}")
    else:
        lines.append(f"  提示: {_STATE_HINTS[state]}")
    div("═")

    if arm_pose is not None:
        x, y, z, rx, ry, rz = arm_pose
        lines.append(f"  EE  x={x:+.4f}m  y={y:+.4f}m  z={z:+.4f}m")
        lines.append(f"      rx={rx:+.4f}  ry={ry:+.4f}  rz={rz:+.4f} (rad)")
    else:
        lines.append(c("  ARM 未连接", _GRAY))
        lines.append("")

    if arm_joints is not None:
        j = arm_joints
        lines.append(
            f"  J  {j[0]:+7.2f}°  {j[1]:+7.2f}°  {j[2]:+7.2f}°"
            f"  {j[3]:+7.2f}°  {j[4]:+7.2f}°  {j[5]:+7.2f}°"
        )
    else:
        lines.append("")

    div()

    lines.append(
        f"  夹爪: {c('无 (固定)', _GRAY)}   "
        f"帧数: {c(str(frame_count), _CYAN)}   "
        f"时长: {elapsed_s:.1f}s"
    )
    lines.append(
        f"  进度: {c(f'Episode {episode_idx + 1}', _CYAN)}"
        f"  (目标 {total_episodes} 条)"
    )

    ev_col = _GREEN if last_event.startswith("✓") else _YELLOW if last_event.startswith("!") else _CYAN
    lines.append(f"  事件: {c(last_event, ev_col) if last_event else c('—', _GRAY)}")

    # 后台保存进度条
    if _bg_save["running"]:
        prog = _bg_save["progress"]
        if prog.startswith("✓"):
            pct = 100
        elif "编码视频" in prog:
            pct = 75
        elif "写入帧数据" in prog:
            pct = 40
        elif "计算动作" in prog:
            pct = 10
        else:
            pct = 5
        bar_w = 30
        done_w = int(bar_w * pct / 100)
        bar = "\u2588" * done_w + "\u2591" * (bar_w - done_w)
        qinfo = f" 队列+{_bg_save['queue_len']}" if _bg_save["queue_len"] > 0 else ""
        lines.append(
            f"  {c('⏳ 后台保存', _YELLOW)} ep{_bg_save['episode']}: "
            f"[{c(bar, _YELLOW)}] {pct}%{qinfo}"
        )
        lines.append(f"       {c(_bg_save['progress'], _CYAN)}")
    elif _bg_save["done"]:
        icon = c("✓", _GREEN) if _bg_save["success"] else c("✗", _RED)
        lines.append(
            f"  {icon} 后台保存 ep{_bg_save['episode']}: "
            f"{c(_bg_save['progress'], _GREEN if _bg_save['success'] else _RED)}"
        )
        _bg_save["done"] = False

    if key_feedback:
        lines.append(f"  按键: {c(key_feedback, _BOLD + _YELLOW)}")

    div("═")

    n = len(lines)
    if not first_hud:
        _up(_N_LINES)
    for line in lines:
        _eol()
        print(line)
    sys.stdout.flush()
    _N_LINES = n


# ── Camera helpers ────────────────────────────────────────────────────────────

def build_cameras(args: argparse.Namespace):
    """Connect 3 Orbbec cameras. Returns dict of {name: cam} or empty dict if disabled."""
    if args.no_cameras:
        return {}

    try:
        from lerobot.cameras.orbbec import OrbbecCamera, OrbbecCameraConfig
    except ImportError:
        warn("lerobot.cameras.orbbec 未找到，跳过相机初始化")
        return {}

    cams = {}
    specs = [
        ("cam_top",    args.sn_top),
        ("cam_bottom", args.sn_bottom),
        ("cam_wrist",  args.sn_wrist),
    ]
    for name, sn in specs:
        if sn is None:
            warn(f"相机 {name} 未提供序列号，跳过")
            continue
        try:
            cfg = OrbbecCameraConfig(
                serial_number=sn,
                fps=args.cam_fps,
                width=args.cam_width,
                height=args.cam_height,
            )
            cam = OrbbecCamera(cfg)
            cam.connect()
            cams[name] = cam
            ok(f"相机 {name} (SN={sn}) 已连接")
        except Exception as exc:
            warn(f"相机 {name} 连接失败: {exc}")

    return cams


def read_cameras(cameras: dict) -> dict:
    """Read one frame from each camera. Returns {name: np.ndarray H×W×3 uint8}."""
    frames = {}
    for name, cam in cameras.items():
        try:
            frames[name] = cam.async_read()
        except Exception as exc:
            warn(f"相机 {name} 读取失败: {exc}")
    return frames


def close_cameras(cameras: dict) -> None:
    for name, cam in cameras.items():
        try:
            cam.disconnect()
            info(f"相机 {name} 已断开")
        except Exception:
            pass


# ── LeRobot dataset helpers ───────────────────────────────────────────────────

def build_dataset(args: argparse.Namespace, camera_names: list[str]):
    """Create or open a LeRobotDataset."""
    try:
        from lerobot.datasets import LeRobotDataset
        from lerobot.configs.video import VideoEncoderConfig
    except ImportError:
        warn("lerobot.datasets 未找到，dataset 功能不可用")
        return None

    img_names = ["height", "width", "channels"]
    features: dict = {}

    for cam_name in camera_names:
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": (args.cam_height, args.cam_width, 3),
            "names": img_names,
        }

    features["observation.state"] = {
        "dtype": "float32",
        "shape": (17,),
        "names": ["j0","j1","j2","j3","j4","j5","x","y","z","rx","ry","rz","qw","qx","qy","qz","gripper"],
    }
    features["observation.force"] = {
        "dtype": "float32",
        "shape": (6,),
        "names": ["fx","fy","fz","mx","my","mz"],
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": ["dx","dy","dz","drx","dry","drz","gripper"],
    }

    root = Path(args.dataset_root) if args.dataset_root else None
    info_path   = root / "meta" / "info.json"    if root else None
    tasks_path  = root / "meta" / "tasks.parquet" if root else None

    # LeRobot 新建零 episode dataset 时只写 info.json，tasks/episodes 要等首次
    # save_episode 才会出现。上次在首条数据前退出或崩溃时，将这个空目录
    # 改名备份后重新创建，而不是删除它。
    if root is not None and info_path.exists() and not tasks_path.exists():
        try:
            info_json = json.loads(info_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            err(f"无法读取 dataset 元数据: {exc}")
            return None
        if int(info_json.get("total_episodes", -1)) == 0:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = root.with_name(f"{root.name}_empty_backup_{suffix}")
            root.rename(backup)
            info(f"已备份上次未录制 episode 的空 dataset: {backup}")
        else:
            err(f"dataset 声明有 episode 但缺少 tasks.parquet: {root}")
            err("为避免数据丢失，程序不会自动修改它。")
            return None

    if root is not None and info_path.exists() and tasks_path.exists():
        try:
            dataset = LeRobotDataset.resume(
                repo_id=args.repo_id,
                root=root,
                force_cache_sync=False,
                camera_encoder=VideoEncoderConfig(vcodec=args.vcodec, pix_fmt="yuv420p", preset="ultrafast"),
            )
            ok(f"LeRobot dataset 已加载（继续追加）: {args.repo_id}  已有 {dataset.num_episodes} 条")
            return dataset
        except Exception as exc:
            err(f"LeRobot dataset 加载失败: {exc}")
            err("为避免误删已采集数据，程序不会自动清理该目录。")
            return None

    if root is not None and root.exists() and not info_path.exists():
        err(f"dataset 目录已存在但缺少 meta/info.json: {root}")
        err("请手动检查或指定新的 --dataset-root，程序不会覆盖它。")
        return None

    try:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.freq,
            features=features,
            root=root,
            robot_type="realman_eco63f",
            camera_encoder=VideoEncoderConfig(vcodec=args.vcodec, pix_fmt="yuv420p", preset="ultrafast"),
        )
        ok(f"LeRobot dataset 已创建: {args.repo_id}")
        return dataset
    except Exception as exc:
        traceback.print_exc()
        warn(f"LeRobot dataset 创建失败: {exc}，将只保存 CSV")
        return None


def read_arm_obs(arm) -> tuple[list | None, list | None, list | None]:
    """
    Read arm state.
    Returns (joints_deg, pose6_euler, force6) or Nones on failure.
    """
    joints = pose6 = force6 = None

    ret_s = arm.rm_get_current_arm_state()
    if ret_s[0] == 0:
        s = ret_s[1]
        joints = list(s["joint"][:6])
        pose6  = list(s["pose"][:6])

    ret_f = arm.rm_get_force_data()
    if ret_f[0] == 0:
        force6 = list(ret_f[1]["zero_force_data"])[:6]
    else:
        force6 = [0.0] * 6

    return joints, pose6, force6


def compute_episode_actions(frames: list) -> list:
    """
    Compute delta actions for a complete episode.
    action[t] = [dx, dy, dz, drx, dry, drz, gripper_norm]  (7-D, float32)
    """
    from scipy.spatial.transform import Rotation

    n = len(frames)
    actions = []
    for t in range(n):
        if t == n - 1:
            actions.append(np.zeros(7, dtype="float32"))
            continue

        s_t  = frames[t]["observation.state"]
        s_t1 = frames[t + 1]["observation.state"]

        dxyz = (s_t1[6:9] - s_t[6:9]).astype("float32")

        q_t   = s_t[12:16]
        q_t1  = s_t1[12:16]
        R_t   = Rotation.from_quat([q_t[1], q_t[2], q_t[3], q_t[0]])
        R_t1  = Rotation.from_quat([q_t1[1], q_t1[2], q_t1[3], q_t1[0]])
        deuler = (R_t1 * R_t.inv()).as_euler("xyz", degrees=False).astype("float32")

        gripper = np.float32(0.0)
        actions.append(np.concatenate([dxyz, deuler, [gripper]]).astype("float32"))

    return actions


def build_observation(
    joints: list, pose6: list, force6: list, cam_frames: dict, gripper_pos: int = 0, arm=None
) -> dict:
    """
    Pack arm state + cameras into a frame dict for the dataset buffer.
    """
    gripper_norm = np.float32(gripper_pos / 255.0)

    # SDK 位姿的 XYZ 欧拉角单位为 rad；数据集和 action 也统一使用 rad。
    # 离线重建时没有 arm 句柄，由欧拉角恢复有效四元数。
    from scipy.spatial.transform import Rotation

    quat = None
    if arm is not None:
        try:
            fk = arm.rm_algo_forward_kinematics(joints, flag=0)
            quat = np.array(fk[3:7], dtype="float32")  # SDK: w, x, y, z
        except Exception as exc:
            warn(f"正运动学四元数读取失败，改用欧拉角计算: {exc}")
    if quat is None or quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-6:
        xyzw = Rotation.from_euler("xyz", pose6[3:6], degrees=False).as_quat()
        quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype="float32")

    state = np.array(joints + pose6[:6], dtype="float32")
    state = np.concatenate([state, quat, [gripper_norm]])
    force = np.array(force6, dtype="float32")

    obs = {
        "observation.state": state,
        "observation.force": force,
        "_gripper_pos": 0,
    }
    for name, frame in cam_frames.items():
        obs[f"observation.images.{name}"] = frame

    return obs


def save_lerobot_episode(frames: list[dict], dataset, task: str,
                         progress_cb = None,
                         episode_dir: Path | None = None,
                         dataset_root: Path | None = None) -> tuple[bool, str]:
    """Write buffered frames to the dataset. Returns (success, error_message).

    On failure, if *episode_dir* is given, temporary PNG frames are rescued from
    ``{dataset_root}/images/`` into ``{episode_dir}/frames_dump/`` so that an
    offline build script can re-encode them later.
    """
    def _p(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    if len(frames) < 2:
        msg = f"帧数过少 ({len(frames)})，丢弃"
        _p(msg)
        return False, msg

    try:
        _p(f"计算动作序列 ({len(frames)} 帧)...")
        actions = compute_episode_actions(frames)

        _p(f"写入帧数据...")
        for obs, action in zip(frames, actions):
            frame_data = {k: v for k, v in obs.items() if not k.startswith("_")}
            frame_data["action"] = action
            frame_data["task"] = task
            dataset.add_frame(frame_data)

        _p("编码视频 + 写入磁盘...")
        dataset.save_episode(parallel_encoding=False)
        _p(f"✓ 已保存 ({len(frames)} 帧)")
        return True, ""
    except Exception as exc:
        _p(f"✗ 保存异常: {exc}")
        traceback.print_exc()
        sys.stderr.flush()
        if exc.__cause__ is not None:
            print("─ 内部异常 ─", file=sys.stderr)
            traceback.print_exception(type(exc.__cause__), exc.__cause__, exc.__cause__.__traceback__)
            sys.stderr.flush()

        # 抢救临时 PNG 帧到 episode 目录
        if episode_dir is not None and dataset_root is not None:
            try:
                ep_idx = dataset.meta.total_episodes  # episode_index before the failed save
                dump_dir = episode_dir / "frames_dump"
                dump_dir.mkdir(parents=True, exist_ok=True)
                for vk in dataset.meta.video_keys:
                    img_dir = (
                        dataset_root / "images" / vk / f"episode-{ep_idx:06d}"
                    )
                    dest = dump_dir / vk.replace("observation.images.", "")
                    if img_dir.is_dir() and any(img_dir.iterdir()):
                        _p(f"抢救帧: {vk} → {dest}")
                        shutil.copytree(img_dir, dest, dirs_exist_ok=True)
                _p(f"帧已导出到 {dump_dir}")
            except Exception as rescue_exc:
                _p(f"抢救帧失败: {rescue_exc}")

        # 清理残留在 dataset 内部的 buffer 和临时图像
        try:
            dataset.clear_episode_buffer(delete_images=True)
        except Exception:
            pass

        return False, str(exc)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ip",           default="192.168.1.18")
    p.add_argument("--port",         type=int, default=8080)
    p.add_argument("--freq",         type=int,   default=30,     help="录制/回放频率 Hz (默认30)")
    p.add_argument("--drag-mode",    type=int,   default=0,     help="拖动模式: 0=电流环, 1=六维力仅位置, 2=仅姿态, 3=位姿同动")
    p.add_argument("--drag-sensitivity", type=int, default=100,
                   help="电流环拖动灵敏度 0..100，越小越沉（默认100）")
    p.add_argument("--home-speed",   type=int,   default=20,    help="归位速度 %% (默认20)")
    p.add_argument("--resample-dt",  type=float, default=0.010, help="回放重采样步长(s)，0=禁用")
    p.add_argument("--smooth-window",type=int,   default=11,    help="回放滑动平均窗口，1=禁用")
    p.add_argument("--init-home",    type=float, nargs=6,
                   default=(3.2, 45.72, -132.48, 88.6, 85.96, 0.73),
                   metavar=("J0","J1","J2","J3","J4","J5"),
                   help="固定 init_home 关节角(deg)，IDLE 下按 [h] 归位")
    p.add_argument("--no-cameras",   action="store_true",       help="不使用相机（只录 CSV）")
    p.add_argument("--sn-top",       default="CP0P76300026",    help="俯视相机序列号")
    p.add_argument("--sn-bottom",    default="CP0N7630000A",    help="平视相机序列号")
    p.add_argument("--sn-wrist",     default="CP0N763000HE",    help="腕部相机序列号")
    p.add_argument("--cam-fps",      type=int,   default=30,    help="相机帧率 (默认30)")
    p.add_argument("--cam-width",    type=int,   default=640,   help="相机宽度 (默认640)")
    p.add_argument("--cam-height",   type=int,   default=480,   help="相机高度 (默认480)")
    p.add_argument("--save-cam-frames", type=int, default=30,
                   help="回放时每隔N帧保存一张截图到episode目录 (默认30帧=1秒, 0=禁用)")
    p.add_argument("--max-save-backlog", type=int, default=2,
                   help="后台保存中+排队的最大episode数；达到后暂停开始新示教 (默认2)")
    p.add_argument("--allow-partial-cameras", action="store_true",
                   help="允许部分相机缺失（默认三台都必须连接）")
    p.add_argument("--repo-id",      default="local/open_toilet_lid_v1", help="LeRobot dataset repo id")
    p.add_argument("--dataset-root", default="",
                   help="dataset 存储根目录 (默认: ./episodes/datasets/)")
    p.add_argument("--num-episodes", type=int,   default=100,   help="目标采集条数 (默认100)")
    p.add_argument("--vcodec",       default="h264",           help="视频编码器 (默认h264, 也可用h264_v4l2m2m)")
    p.add_argument("--task",         default="Open the toilet lid", help="任务描述文字")
    p.add_argument("--output-dir",   default="",               help="CSV episode 存放目录 (默认: ./episodes/)")
    args = p.parse_args()
    if args.freq <= 0 or args.cam_fps <= 0:
        p.error("--freq 和 --cam-fps 必须大于 0")
    if args.cam_width <= 0 or args.cam_height <= 0:
        p.error("相机分辨率必须大于 0")
    if args.drag_mode not in (0, 1, 2, 3):
        p.error("--drag-mode 必须是 0/1/2/3")
    if not 0 <= args.drag_sensitivity <= 100:
        p.error("--drag-sensitivity 必须在 0..100 之间")
    if not 1 <= args.home_speed <= 100:
        p.error("--home-speed 必须在 1..100 之间")
    if args.resample_dt < 0 or args.smooth_window < 1:
        p.error("--resample-dt 不能为负，--smooth-window 必须大于等于 1")
    if args.num_episodes <= 0:
        p.error("--num-episodes 必须大于 0")
    if args.max_save_backlog <= 0:
        p.error("--max-save-backlog 必须大于 0")
    return args


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not sys.stdin.isatty():
        err("采集程序需要交互式终端，请使用 ssh -t 连接后再运行。")
        sys.exit(2)

    _running = [True]

    def _sig(signum, _frame):
        _running[0] = False

    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    # ── Connect arm ────────────────────────────────────────────────────────
    banner("Realman ECO63F — Open Toilet Lid 采集")
    info(f"连接机械臂 {args.ip}:{args.port} ...")
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    arm.rm_create_robot_arm(args.ip, args.port)
    arm_cleanup_pending = [True]

    def _cleanup_arm_at_exit() -> None:
        if arm_cleanup_pending[0]:
            try:
                arm.rm_delete_robot_arm()
            except Exception:
                pass

    atexit.register(_cleanup_arm_at_exit)
    ret = arm.rm_get_current_arm_state()
    if ret[0] != 0:
        err(f"连接失败 (code {ret[0]})")
        arm.rm_delete_robot_arm()
        arm_cleanup_pending[0] = False
        sys.exit(1)
    ok(f"机械臂已连接: {args.ip}")

    setup_tool_frame(arm)

    # ── Keyboard ───────────────────────────────────────────────────────────
    # 终端模式在相机检查通过后再切换。

    # ── Gripper ────────────────────────────────────────────────────────────
    # 无电机夹爪，固定为 None
    gripper = None
    gripper_target = 0

    # ── Cameras ────────────────────────────────────────────────────────────
    info("初始化相机...")
    cameras = build_cameras(args)
    camera_cleanup_pending = [True]

    def _cleanup_cameras_at_exit() -> None:
        if camera_cleanup_pending[0]:
            close_cameras(cameras)

    atexit.register(_cleanup_cameras_at_exit)
    camera_names = list(cameras.keys())
    expected_cameras = sum(sn is not None for sn in (args.sn_top, args.sn_bottom, args.sn_wrist))
    if not args.no_cameras and len(cameras) != expected_cameras and not args.allow_partial_cameras:
        err(f"相机连接不完整: 期望 {expected_cameras} 台，实际 {len(cameras)} 台。")
        err("请检查相机后重试；调试时可使用 --allow-partial-cameras 或 --no-cameras。")
        close_cameras(cameras)
        camera_cleanup_pending[0] = False
        arm.rm_delete_robot_arm()
        arm_cleanup_pending[0] = False
        sys.exit(1)
    cameras_ok = len(cameras) == expected_cameras or args.no_cameras

    _init_keyboard()
    atexit.register(_restore_keyboard)
    info("键盘控制已就绪（非阻塞模式）")

    # ── CSV output root ────────────────────────────────────────────────────
    output_root = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "episodes"
    output_root.mkdir(parents=True, exist_ok=True)

    # Dataset 存储也放在 episodes/datasets/ 下
    if not args.dataset_root:
        args.dataset_root = str(output_root / "datasets")

    dataset_root_path = Path(args.dataset_root) if args.dataset_root else None

    # ── LeRobot dataset ────────────────────────────────────────────────────
    dataset = None
    if camera_names or not args.no_cameras:
        dataset = build_dataset(args, camera_names)
        if dataset is None:
            err("LeRobot dataset 初始化失败，未开始采集。如果只需要 CSV，请显式使用 --no-cameras。")
            close_cameras(cameras)
            camera_cleanup_pending[0] = False
            arm.rm_delete_robot_arm()
            arm_cleanup_pending[0] = False
            _restore_keyboard()
            sys.exit(1)

    _save_worker_thread = threading.Thread(
        target=_save_worker, args=(dataset, dataset_root_path), daemon=False,
    )
    _save_worker_thread.start()

    dt = 1.0 / args.freq

    # ── State ──────────────────────────────────────────────────────────────
    state                 = State.IDLE
    episode_dir: Path | None = None
    tmp_csv_path: Path | None = None
    csv_file              = None
    csv_writer            = None
    episode_home: list[float] | None = None
    arm_pose              = None
    arm_joints            = None
    frame_count           = 0
    record_start_t        = 0.0
    record_duration_s     = 0.0
    freq_actual           = args.freq
    first_hud             = True
    replay_stop_flag      = [False]
    replay_done_event     = threading.Event()
    last_event            = ""
    _key_feedback         = ""
    pending_replay        = False   # RECORDING 结束后等待 [r] 确认
    episode_frames: list[dict] = []
    _replay_frame_idx     = 0       # 回放帧计数（用于截图命名）
    _cam_frames_dir: Path | None = None
    episode_idx = dataset.num_episodes if dataset is not None else 0
    _replay_gripper = [0]

    banner("就绪 — 按 [Enter] 开始拖动示教")
    print()

    while _running[0]:
        t0 = time.perf_counter()

        if state == State.IDLE and not pending_replay and episode_idx >= args.num_episodes:
            ok(f"已达到目标条数: {args.num_episodes}")
            break

        key = read_key_raw()

        # ── Global exit keys ───────────────────────────────────────────────
        if key in ("q", "Q", "\x1b"):   # q / ESC
            info("退出")
            break

        # ── State transitions ──────────────────────────────────────────────
        if state == State.IDLE:
            if pending_replay:
                # 等待 [r] 确认回放或 [b] 放弃
                if key == "r":
                    _key_feedback = "r → 开始回放"
                    pending_replay = False
                    if tmp_csv_path is None or not tmp_csv_path.exists():
                        err("找不到录制文件，无法回放")
                        last_event = "✗ 找不到 CSV，无法回放"
                        state = State.IDLE
                    else:
                        episode_frames = []
                        _replay_frame_idx = 0
                        if args.save_cam_frames > 0 and episode_dir:
                            _cam_frames_dir = episode_dir / "cam_frames"
                            _cam_frames_dir.mkdir(parents=True, exist_ok=True)
                        state = State.REPLAYING
                        replay_stop_flag = [False]
                        replay_done_event = threading.Event()
                        joints_r, grippers_r, t_rels_r = load_recording(tmp_csv_path)
                        last_event = f"✓ 进入 REPLAYING — 共 {len(joints_r)} 帧，[b] 可中止"
                        info(f"回放 {len(joints_r)} 帧...")

                        def _do_replay():
                            _replay_gripper[0] = 0
                            replay_csv(arm, gripper, joints_r, grippers_r, t_rels_r,
                                       stop_flag=replay_stop_flag,
                                       resample_dt=args.resample_dt,
                                       smooth_window=args.smooth_window,
                                       initial_gripper=0,
                                       gripper_feedback=_replay_gripper)
                            replay_done_event.set()

                        threading.Thread(target=_do_replay, daemon=True).start()

                elif key in ("b", "d"):
                    _key_feedback = "b → 已放弃录制"
                    pending_replay = False
                    if episode_dir and episode_dir.exists():
                        shutil.rmtree(episode_dir)
                        episode_dir = None
                    tmp_csv_path = None
                    episode_frames = []
                    state = State.IDLE
                    last_event = "! 已放弃录制 → 回到 IDLE"
                    warn("已放弃本次录制")

                elif key == "h" and args.init_home is not None:
                    _key_feedback = "h → 归位到 init_home"
                    warn("归位到 init_home，请确保路径无障碍物...")
                    ok_home = move_to_home(arm, args.init_home, speed=args.home_speed)
                    if ok_home:
                        last_event = "✓ 已归位到 init_home"
                    else:
                        last_event = "✗ 归位到 init_home 失败"

            else:
                if key == "h" and args.init_home is not None:
                    _key_feedback = "h → 归位到 init_home"
                    warn("归位到 init_home，请确保路径无障碍物...")
                    ok_home = move_to_home(arm, args.init_home, speed=args.home_speed)
                    if ok_home:
                        last_event = "✓ 已归位到 init_home"
                    else:
                        last_event = "✗ 归位到 init_home 失败"

                elif key == "\n":  # Enter
                    _key_feedback = "Enter → 开始拖动示教"
                    save_backlog = _save_queue.qsize() + int(_bg_save["running"])
                    if dataset is not None and save_backlog >= args.max_save_backlog:
                        last_event = (
                            f"⏳ 保存积压 {save_backlog} 条，达到限制 {args.max_save_backlog}；"
                            "请等待后台保存后再开始"
                        )
                        warn(last_event)
                        continue
                    ret_j = arm.rm_get_joint_degree()
                    if ret_j[0] != 0:
                        err("读取关节角失败，无法开始录制")
                        last_event = "✗ 读取关节角失败"
                    else:
                        episode_home = list(ret_j[1])

                        if args.drag_mode == 0:
                            ret_sens = arm.rm_set_drag_teach_sensitivity(args.drag_sensitivity)
                            if ret_sens != 0:
                                err(f"设置拖动灵敏度失败 (code {ret_sens})，未开始录制")
                                last_event = f"✗ 灵敏度设置失败 (code {ret_sens})"
                                continue
                            info(f"电流环拖动灵敏度: {args.drag_sensitivity}")

                        ret_d = arm.rm_start_multi_drag_teach(mode=args.drag_mode, singular_wall=1)
                        if ret_d != 0:
                            err(f"rm_start_multi_drag_teach 失败 (code {ret_d})")
                            last_event = f"✗ 拖动模式启动失败 (code {ret_d})"
                        else:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            episode_dir  = output_root / ts
                            episode_dir.mkdir(parents=True, exist_ok=True)
                            tmp_csv_path = episode_dir / "recording.csv"
                            try:
                                csv_file   = open(tmp_csv_path, "w", newline="")
                                csv_writer = csv.writer(csv_file)
                                csv_writer.writerow(CSV_HEADER)
                            except Exception as exc:
                                err(f"无法创建录制文件: {exc}")
                                if csv_file:
                                    csv_file.close()
                                    csv_file = None
                                shutil.rmtree(episode_dir)
                                episode_dir = None
                                last_event = "✗ 创建录制文件失败"
                                continue
                            frame_count    = 0
                            record_duration_s = 0.0
                            episode_frames = []
                            record_start_t = time.perf_counter()
                            state          = State.RECORDING
                            suffix = (f", sensitivity={args.drag_sensitivity}"
                                      if args.drag_mode == 0 else "")
                            last_event     = f"✓ 进入 RECORDING — 拖动模式已开启 (mode={args.drag_mode}{suffix})"
                            ok(f"开始拖动示教，录制到: {tmp_csv_path}")

                elif key == "r":
                    _key_feedback = "r → 开始回放"
                    # 直接回放上一次的 CSV（如有）
                    if tmp_csv_path is not None and tmp_csv_path.exists():
                        episode_frames = []
                        _replay_frame_idx = 0
                        if args.save_cam_frames > 0 and episode_dir:
                            _cam_frames_dir = episode_dir / "cam_frames"
                            _cam_frames_dir.mkdir(parents=True, exist_ok=True)
                        state = State.REPLAYING
                        replay_stop_flag = [False]
                        replay_done_event = threading.Event()
                        joints_r, grippers_r, t_rels_r = load_recording(tmp_csv_path)
                        last_event = f"✓ 进入 REPLAYING — 共 {len(joints_r)} 帧，[b] 可中止"
                        info(f"回放 {len(joints_r)} 帧...")

                        def _do_replay2():
                            _replay_gripper[0] = 0
                            replay_csv(arm, gripper, joints_r, grippers_r, t_rels_r,
                                       stop_flag=replay_stop_flag,
                                       resample_dt=args.resample_dt,
                                       smooth_window=args.smooth_window,
                                       initial_gripper=0,
                                       gripper_feedback=_replay_gripper)
                            replay_done_event.set()

                        threading.Thread(target=_do_replay2, daemon=True).start()

        elif state == State.RECORDING:
            if key in ("b", "d"):
                _key_feedback = "b → 已放弃录制"
                arm.rm_stop_drag_teach()
                if csv_file:
                    csv_file.close()
                    csv_file = None
                if episode_dir and episode_dir.exists():
                    shutil.rmtree(episode_dir)
                    episode_dir = None
                tmp_csv_path   = None
                episode_frames = []
                state          = State.IDLE
                pending_replay = False
                last_event     = "! 已放弃 → 回到 IDLE"
                warn("已放弃本次录制")

            elif key == "\n":  # Enter — 停止拖动 + 自动归位 + 保存 CSV
                _key_feedback = "Enter → 停止示教，归位中..."
                arm.rm_stop_drag_teach()

                traj_txt = str(episode_dir / "arm_trajectory.txt") if episode_dir else None
                if traj_txt:
                    ret_sv = arm.rm_save_trajectory(traj_txt)
                    if isinstance(ret_sv, tuple) and ret_sv[0] == 0:
                        info(f"控制器轨迹已保存: {traj_txt} ({ret_sv[1]} 点)")

                if csv_file:
                    csv_file.close()
                    csv_file = None
                csv_writer = None

                ok(f"示教结束，共录制 {frame_count} 帧  ({time.perf_counter()-record_start_t:.1f}s)")
                record_duration_s = time.perf_counter() - record_start_t

                # 自动归位到本次录制起点
                if episode_home is not None:
                    warn("自动归位到录制起点...")
                    ok_home = move_to_home(arm, episode_home, speed=args.home_speed)
                    if ok_home:
                        state = State.IDLE
                        pending_replay = True
                        last_event = f"✓ 已归位到起点，共 {frame_count} 帧，按 [r] 回放"
                    else:
                        state = State.IDLE
                        last_event = "✗ 归位失败"
                else:
                    state = State.IDLE
                    pending_replay = True
                    last_event = f"✓ 录制完成，共 {frame_count} 帧，按 [r] 回放"

            else:
                # Read arm state during recording
                arm_joints, arm_pose_euler, _ = read_arm_obs(arm)

                if arm_joints is not None:
                    arm_pose = arm_pose_euler
                    t_rel = time.perf_counter() - record_start_t

                    if csv_writer:
                        csv_writer.writerow([
                            f"{t_rel:.4f}",
                            *[f"{j:.4f}" for j in arm_joints],
                            *[f"{v:.6f}" for v in arm_pose_euler],
                            0,   # gripper fixed to 0
                        ])

                    frame_count += 1

        elif state == State.REPLAYING:
            if replay_done_event.is_set():
                state = State.REVIEW
                last_event = f"✓ 进入 REVIEW — 回放完成，采集 {len(episode_frames)} 帧，[s] 保存 [d] 丢弃"
            elif key == "b":
                _key_feedback = "b → 已中止回放"
                replay_stop_flag[0] = True
                replay_done_event.wait(timeout=1.0)
                arm.rm_set_arm_slow_stop()
                state = State.REVIEW
                last_event = f"! 回放已中止 → REVIEW，已采集 {len(episode_frames)} 帧，[s] 保存 [d] 丢弃"
                warn("回放已中止")
            else:
                arm_joints, arm_pose_euler, force6 = read_arm_obs(arm)
                if arm_joints is not None:
                    arm_pose = arm_pose_euler
                    cam_frames = read_cameras(cameras)
                    obs = build_observation(arm_joints, arm_pose_euler, force6, cam_frames, 0, arm=arm)
                    episode_frames.append(obs)

                    _replay_frame_idx += 1
                    # 每隔 N 帧保存一张截图到 episode 目录
                    if (args.save_cam_frames > 0 and _cam_frames_dir is not None
                            and _replay_frame_idx % args.save_cam_frames == 0):
                        for name, img in cam_frames.items():
                            frame_path = _cam_frames_dir / f"frame_{_replay_frame_idx:06d}_{name}.jpg"
                            cv2.imwrite(str(frame_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        elif state == State.REVIEW:
            if key == "s":
                _key_feedback = "s → 保存到 LeRobot"
                # Save CSV meta.json
                if episode_dir:
                    meta = {
                        "timestamp":            episode_dir.name,
                        "tool_frame_pose":      {"z_m": 0.2044, "x_m": 0.0, "y_m": 0.0,
                                                 "rx_rad": 0.0, "ry_rad": 0.0, "rz_rad": 0.0},
                        "tool_payload_kg":      0.5892,
                        "tool_com_m":           [-0.0129, 0.0002, 0.0926],
                        "drag_mode":            args.drag_mode,
                        "record_freq_hz":       args.freq,
                        "episode_home_deg":     episode_home,
                        "frame_count":          frame_count,
                        "duration_s":           round(record_duration_s, 3),
                        "replay_frame_count":   len(episode_frames),
                        "task":                 args.task,
                        "state_units":          {"joints": "deg", "xyz": "m", "euler_xyz": "rad",
                                                 "quaternion": "wxyz", "gripper": "normalized"},
                        "action_units":         {"dxyz": "m", "deuler_xyz": "rad",
                                                 "gripper": "normalized"},
                        "cameras":              camera_names,
                        "sdk_version":          "1.1.5",
                    }
                    (episode_dir / "meta.json").write_text(
                        json.dumps(meta, indent=2, ensure_ascii=False)
                    )
                    ok(f"CSV 数据已保存: {episode_dir}")

                if dataset is None:
                    err("LeRobot dataset 未初始化，只保存了 CSV")
                    last_event = "! CSV 已保存，dataset 未初始化"
                    episode_idx += 1
                    episode_frames = []
                    state = State.IDLE
                    episode_dir = None
                elif not episode_frames:
                    err("episode_frames 为空，没有采集到回放数据")
                    last_event = "! CSV 已保存，回放数据为空"
                    episode_idx += 1
                    episode_frames = []
                    state = State.IDLE
                    episode_dir = None
                else:
                    frames_copy = list(episode_frames)
                    ep_num = episode_idx
                    episode_idx += 1
                    n_frames = len(frames_copy)

                    _save_queue.put({
                        "frames":      frames_copy,
                        "task":        args.task,
                        "ep_num":      ep_num + 1,
                        "n_frames":    n_frames,
                        "episode_dir": str(episode_dir) if episode_dir else None,
                    })
                    _bg_save["queue_len"] = _save_queue.qsize()
                    qlen = _bg_save["queue_len"]
                    last_event = (
                        f"→ 已加入保存队列 (ep{ep_num+1}, {n_frames} 帧)"
                        + (f"，队列 {qlen} 条等待" if qlen > 1 else "")
                        + "，已回到 IDLE"
                    )

                episode_frames = []
                state = State.IDLE
                episode_dir = None
                tmp_csv_path = None

            elif key == "d":
                _key_feedback = "d → 已丢弃数据"
                if episode_dir and episode_dir.exists():
                    shutil.rmtree(episode_dir)
                    ok("数据已丢弃")
                    episode_dir = None
                episode_frames = []
                state = State.IDLE
                tmp_csv_path = None
                last_event = "! 数据已丢弃 → 回到 IDLE"

            elif key == "r":
                _key_feedback = "r → 再次回放"
                warn("归位中，请确保路径无障碍物...")
                ok_home = move_to_home(arm, episode_home, speed=args.home_speed)
                if ok_home:
                    episode_frames = []
                    _replay_frame_idx = 0
                    if args.save_cam_frames > 0 and episode_dir:
                        _cam_frames_dir = episode_dir / "cam_frames"
                        _cam_frames_dir.mkdir(parents=True, exist_ok=True)
                    state = State.REPLAYING
                    replay_stop_flag = [False]
                    replay_done_event = threading.Event()
                    if tmp_csv_path and tmp_csv_path.exists():
                        joints_r, grippers_r, t_rels_r = load_recording(tmp_csv_path)
                        last_event = f"✓ 再次回放 — 共 {len(joints_r)} 帧"

                        def _do_replay3():
                            _replay_gripper[0] = 0
                            replay_csv(arm, gripper, joints_r, grippers_r, t_rels_r,
                                       stop_flag=replay_stop_flag,
                                       resample_dt=args.resample_dt,
                                       smooth_window=args.smooth_window,
                                       initial_gripper=0,
                                       gripper_feedback=_replay_gripper)
                            replay_done_event.set()

                        threading.Thread(target=_do_replay3, daemon=True).start()
                    else:
                        err("找不到录制文件")
                        last_event = "✗ 找不到 CSV，无法回放"
                        state = State.REVIEW
                else:
                    last_event = "✗ 归位失败"

        # ── HUD state read (outside RECORDING/REPLAYING) ───────────────────
        if state not in (State.RECORDING, State.REPLAYING):
            ret_s = arm.rm_get_current_arm_state()
            if ret_s[0] == 0:
                arm_joints = list(ret_s[1]["joint"][:6])
                arm_pose   = list(ret_s[1]["pose"][:6])

        elapsed = (time.perf_counter() - record_start_t) if state == State.RECORDING else 0.0

        render_hud(
            state=state,
            arm_pose=arm_pose,
            arm_joints=arm_joints,
            frame_count=frame_count if state == State.RECORDING else 0,
            elapsed_s=elapsed,
            freq_actual=freq_actual,
            episode_idx=episode_idx,
            total_episodes=args.num_episodes,
            first_hud=first_hud,
            cameras_ok=cameras_ok,
            last_event=last_event,
            pending_replay=pending_replay,
            key_feedback=_key_feedback,
        )
        _key_feedback = ""
        first_hud = False

        elapsed_loop = time.perf_counter() - t0
        sleep_t = dt - elapsed_loop
        if sleep_t > 0:
            time.sleep(sleep_t)
        freq_actual = 1.0 / max(time.perf_counter() - t0, 1e-6)

    # ── Teardown ───────────────────────────────────────────────────────────
    print()
    if state == State.RECORDING:
        warn("录制中退出，停止拖动示教...")
        arm.rm_stop_drag_teach()
        if csv_file:
            csv_file.close()
    if state == State.REPLAYING:
        replay_stop_flag[0] = True
        arm.rm_set_arm_slow_stop()
        replay_done_event.wait(timeout=2.0)

    close_cameras(cameras)
    camera_cleanup_pending[0] = False

    if _save_worker_thread.is_alive():
        _save_queue.put(None)
        info("等待后台保存队列全部完成；不会因固定超时强制退出...")
        wait_started = time.monotonic()
        last_report = wait_started
        while _save_worker_thread.is_alive():
            _save_worker_thread.join(timeout=1.0)
            now = time.monotonic()
            if _save_worker_thread.is_alive() and now - last_report >= 10.0:
                elapsed = int(now - wait_started)
                info(
                    f"后台仍在保存: ep{_bg_save['episode']}，"
                    f"排队 {_save_queue.qsize()} 条，已等待 {elapsed}s"
                )
                last_report = now

    if dataset is not None:
        try:
            dataset.finalize()
            ok(f"Dataset 已保存，共 {dataset.num_episodes} 条")
        except Exception as exc:
            warn(f"Dataset finalize 失败: {exc}")

    info("断开机械臂...")
    arm.rm_delete_robot_arm()
    arm_cleanup_pending[0] = False
    _restore_keyboard()
    ok("已退出")


if __name__ == "__main__":
    main()
