"""
Shared utilities for drag_teach_record_keyboard.py.

Covers:
  - tool-frame setup
  - arm home-move
  - CANFD joint replay from a recording.csv
  - ANSI terminal helpers
"""
from __future__ import annotations

import bisect
import csv
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

# ── optional SDK import ───────────────────────────────────────────────────────
try:
    from Robotic_Arm.rm_robot_interface import RoboticArm  # noqa: F401
    _sdk_ok = True
except ImportError:
    _sdk_ok = False

# ── ANSI ──────────────────────────────────────────────────────────────────────
_R      = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_GRAY   = "\033[90m"
_BLUE   = "\033[34m"


def c(text: object, code: str) -> str:
    return f"{code}{text}{_R}"


def ok(msg: str)   -> None: print(f"  {c('✓', _GREEN)}  {msg}")
def warn(msg: str) -> None: print(f"  {c('!', _YELLOW)}  {msg}")
def err(msg: str)  -> None: print(f"  {c('✗', _RED)}  {msg}")
def info(msg: str) -> None: print(f"  {c('·', _CYAN)}  {msg}")


def banner(title: str) -> None:
    print("\n" + "─" * 64)
    print(f"  {c(title, _BOLD)}")
    print("─" * 64)


# ── Tool frame constants ──────────────────────────────────────────────────────
TOOL_FRAME_NAME = "tcp_tip"   # ≤12 chars, stored in controller

# Flange → fingertip-centre offset (m / rad, Euler XYZ).
# Z = 204.45 mm from STL (base_link.STL, origin = flange face).
# X=Y=0 because the two fingertips are symmetric on ±X.
# Rotational offsets (rx/ry/rz) need tactile calibration; zero for now.
TOOL_FRAME_POSE = {
    "x": 0.0, "y": 0.0, "z": 0.2044,
    "rx": 0.0, "ry": 0.0, "rz": 0.0,
}
TOOL_PAYLOAD = 0.5892                        # kg，当前末端夹具称重 589.2 g
TOOL_COM     = (-0.0129, 0.0002, 0.0926)     # m  质心尚未重测，暂留甲方 URDF 值
BARE_FLANGE_FRAME_NAME = "flange_zero"


def create_eco63_kinematics():
    """Create an ECO63 local solver independent from the live robot handle."""
    try:
        from Robotic_Arm.rm_robot_interface import (
            Algo,
            rm_force_type_e,
            rm_robot_arm_model_e,
        )
        solver = Algo(
            rm_robot_arm_model_e.RM_MODEL_ECO_63_E,
            rm_force_type_e.RM_MODEL_RM_B_E,
        )
    except Exception as exc:
        err(f"创建 ECO63 独立运动学求解器失败: {exc}")
        return None
    ok("独立运动学求解器已初始化: ECO63")
    return solver


def setup_tool_frame(arm, kinematics=None) -> bool:
    """Write TOOL_FRAME_POSE into the controller and activate it."""
    try:
        from Robotic_Arm.rm_ctypes_wrap import (
            rm_frame_t, rm_pose_t, rm_position_t, rm_euler_t,
        )
    except ImportError:
        warn("rm_ctypes_wrap not available — skipping tool-frame setup.")
        return False

    frame = rm_frame_t()
    frame.frame_name = TOOL_FRAME_NAME.encode()

    pose = rm_pose_t()
    pos  = rm_position_t()
    pos.x, pos.y, pos.z = TOOL_FRAME_POSE["x"], TOOL_FRAME_POSE["y"], TOOL_FRAME_POSE["z"]
    euler = rm_euler_t()
    euler.rx, euler.ry, euler.rz = TOOL_FRAME_POSE["rx"], TOOL_FRAME_POSE["ry"], TOOL_FRAME_POSE["rz"]
    pose.position = pos
    pose.euler    = euler
    frame.pose    = pose

    frame.payload = TOOL_PAYLOAD
    frame.x = TOOL_COM[0]
    frame.y = TOOL_COM[1]
    frame.z = TOOL_COM[2]

    ret = arm.rm_set_manual_tool_frame(frame)
    if ret != 0:
        warn(f"rm_set_manual_tool_frame code {ret}，尝试直接切换已有工具坐标系…")

    ret = arm.rm_change_tool_frame(TOOL_FRAME_NAME)
    if ret != 0:
        err(f"rm_change_tool_frame failed (code {ret})")
        return False
    if kinematics is not None:
        kinematics.rm_algo_set_toolframe(frame)

    ret2 = arm.rm_get_current_tool_frame()
    if ret2[0] == 0:
        ok(f"工具坐标系已设置: {TOOL_FRAME_NAME}  z={TOOL_FRAME_POSE['z']*1000:.1f} mm")
    else:
        warn("工具坐标系验证读取失败，已继续")
    return True


def setup_bare_flange_frame(
    arm,
    kinematics=None,
    payload_kg: float = 0.0,
    com_m=(0.0, 0.0, 0.0),
) -> bool:
    """Activate a zero-TCP frame with an optional physical flange payload."""
    payload_kg = float(payload_kg)
    com_m = tuple(float(value) for value in com_m)
    if (
        not math.isfinite(payload_kg)
        or payload_kg < 0
        or len(com_m) != 3
        or not all(math.isfinite(v) for v in com_m)
    ):
        raise ValueError("payload_kg/com_m must be finite and payload cannot be negative")
    try:
        from Robotic_Arm.rm_ctypes_wrap import (
            rm_euler_t,
            rm_frame_t,
            rm_pose_t,
            rm_position_t,
        )
    except ImportError:
        warn("rm_ctypes_wrap not available — skipping bare-flange setup.")
        return False

    frame = rm_frame_t()
    frame.frame_name = BARE_FLANGE_FRAME_NAME.encode()
    pose = rm_pose_t()
    pose.position = rm_position_t(0.0, 0.0, 0.0)
    pose.euler = rm_euler_t(0.0, 0.0, 0.0)
    frame.pose = pose
    frame.payload = payload_kg
    frame.x, frame.y, frame.z = com_m

    ret = arm.rm_set_manual_tool_frame(frame)
    if ret != 0:
        ret = arm.rm_update_tool_frame(frame)
    if ret != 0:
        err(f"设置空法兰工具坐标系失败 (code {ret})")
        return False
    ret = arm.rm_change_tool_frame(BARE_FLANGE_FRAME_NAME)
    if ret != 0:
        err(f"切换空法兰工具坐标系失败 (code {ret})")
        return False
    algo = kinematics if kinematics is not None else arm
    if hasattr(algo, "rm_algo_set_toolframe"):
        algo.rm_algo_set_toolframe(frame)
    ok(
        "工具坐标系已设置: TCP偏移=0 mm，"
        f"负载={payload_kg:.3f} kg，"
        f"质心={tuple(round(value * 1000.0, 1) for value in com_m)} mm"
    )
    return True


# ── Arm home-move ─────────────────────────────────────────────────────────────

def move_to_home(arm, home_joints: list[float], speed: int = 20, timeout_s: float = 60.0) -> bool:
    """Move arm to home_joints (degrees) via joint-space motion.

    Uses non-blocking movej and polls. Blocking movej can hang forever if the
    controller is in error / still in drag teach.
    """
    home_joints = [float(j) for j in home_joints]
    info(f"归位目标 (deg): {[round(j, 2) for j in home_joints]}  speed={speed}%")
    sys.stdout.flush()
    ret = arm.rm_movej(home_joints, v=speed, r=0, connect=0, block=0)
    if ret != 0:
        err(f"rm_movej 归位失败 (code {ret})。若刚报过错，先在示教器清黄灯再试。")
        return False
    info(f"已下发归位，最多等 {timeout_s:.0f}s。Ctrl-C 中止。")
    sys.stdout.flush()
    deadline = time.time() + max(5.0, float(timeout_s))
    last_print = 0.0
    try:
        while time.time() < deadline:
            ret_s = arm.rm_get_current_arm_state()
            if isinstance(ret_s, tuple) and ret_s[0] == 0:
                joints = [float(v) for v in list(ret_s[1]["joint"][:6])]
                err_max = max(abs(a - b) for a, b in zip(joints, home_joints))
                now = time.time()
                if now - last_print > 1.5:
                    info(f"当前关节 {_fmt_joints(joints)}  误差 {err_max:.1f}°")
                    sys.stdout.flush()
                    last_print = now
                if err_max < 1.5:
                    ok("已到达起点")
                    return True
            time.sleep(0.2)
    except KeyboardInterrupt:
        warn("归位被中断，缓停")
        try:
            arm.rm_set_arm_slow_stop()
        except Exception:
            pass
        return False
    err("归位超时。控制器可能仍在报错/拖动未停，没有死等。")
    try:
        arm.rm_set_arm_slow_stop()
    except Exception:
        pass
    return False


def _fmt_joints(joints: list[float]) -> str:
    return "[" + ", ".join(f"{v:.1f}" for v in joints) + "]"


# ── CSV recording helpers ─────────────────────────────────────────────────────
CSV_HEADER = [
    "t_rel",
    "j0", "j1", "j2", "j3", "j4", "j5",
    "x", "y", "z", "rx", "ry", "rz",
    "gripper_pos",
]


def load_recording(csv_path: str | Path) -> tuple[list, list, list]:
    """
    Load recording.csv.
    Returns (joints_list, gripper_list, t_rel_list).
      joints_list[i]  = [j0..j5] in degrees
      gripper_list[i] = int 0-255
      t_rel_list[i]   = float seconds
    """
    joints, grippers, t_rels = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            joints.append([float(row[f"j{k}"]) for k in range(6)])
            grippers.append(int(float(row["gripper_pos"])))
            t_rels.append(float(row["t_rel"]))
    return joints, grippers, t_rels


# ── CANFD joint replay ────────────────────────────────────────────────────────

def _smooth_and_resample(
    joints: list, grippers: list, t_rels: list,
    dt: float = 0.010, smooth_window: int = 11,
) -> tuple[list, list, list]:
    """均匀重采样 + 滑动平均，消除不规律时间戳导致的速度突变和编码器噪声。"""
    n = len(joints)
    if n < 2:
        return joints, grippers, t_rels

    n_joints = len(joints[0])

    # Step 1: 均匀重采样（线性插值）
    t0, t1 = t_rels[0], t_rels[-1]
    new_t = []
    t = t0
    while t <= t1 + 1e-9:
        new_t.append(t)
        t += dt

    resampled = []
    new_grippers = []
    for nt in new_t:
        pos = bisect.bisect_right(t_rels, nt)
        i0 = max(0, pos - 1)
        i1 = min(pos, n - 1)
        span = t_rels[i1] - t_rels[i0]
        alpha = max(0.0, min(1.0, (nt - t_rels[i0]) / span if span > 1e-9 else 0.0))
        resampled.append([joints[i0][j] + (joints[i1][j] - joints[i0][j]) * alpha
                          for j in range(n_joints)])
        new_grippers.append(grippers[i0])

    # Step 2: 均匀时间轴上滑动平均
    hw = smooth_window // 2
    n2 = len(resampled)
    smoothed = []
    for i in range(n2):
        lo, hi = max(0, i - hw), min(n2, i + hw + 1)
        cnt = hi - lo
        smoothed.append([sum(resampled[k][j] for k in range(lo, hi)) / cnt
                         for j in range(n_joints)])

    new_t_rel = [i * dt for i in range(n2)]
    info(f"预处理: {n} 帧 → {n2} 帧  dt={dt*1000:.0f}ms  smooth={smooth_window}")
    return smoothed, new_grippers, new_t_rel


def replay_csv(arm, gripper, joints: list, grippers: list, t_rels: list,
               speed: float = 1.0, stop_flag: list | None = None,
               resample_dt: float = 0.010, smooth_window: int = 11,
               initial_gripper: int | None = None,
               gripper_feedback: list | None = None) -> None:
    """
    Replay a recording via rm_movej_canfd.
    stop_flag:        mutable [bool], set True from another thread to abort.
    speed:            playback speed multiplier (1.0 = real-time).
    gripper_feedback: optional mutable [int] — updated with last commanded
                      gripper position each frame.
    """
    if stop_flag is None:
        stop_flag = [False]

    n = len(joints)
    if n == 0:
        warn("回放数据为空")
        return

    if smooth_window > 1 or resample_dt > 0:
        joints, grippers, t_rels = _smooth_and_resample(
            joints, grippers, t_rels,
            dt=resample_dt if resample_dt > 0 else (t_rels[-1] - t_rels[0]) / max(n - 1, 1),
            smooth_window=max(1, smooth_window),
        )
        n = len(joints)

    info(f"开始回放: {n} 帧  speed={speed}x")
    last_g = initial_gripper if initial_gripper is not None else (grippers[0] if grippers else -1)
    t_start = time.perf_counter()

    for i in range(n):
        if stop_flag[0]:
            info("回放被中止")
            break

        arm.rm_movej_canfd(joints[i], follow=True)

        if gripper is not None and grippers[i] != last_g:
            try:
                gripper.move_to(grippers[i], speed=0x80, force=0x80, wait=False)
                last_g = grippers[i]
            except Exception as exc:
                warn(f"回放夹爪指令失败 (帧{i}→{grippers[i]}): {exc}")
        if gripper_feedback is not None and grippers:
            gripper_feedback[0] = grippers[i] if grippers else last_g

        target_t = t_start + t_rels[i] / speed
        gap = target_t - time.perf_counter()
        if gap > 0.002:
            time.sleep(gap - 0.001)
        while time.perf_counter() < target_t:
            pass

    if not stop_flag[0] and n > 0:
        arm.rm_movej_canfd(joints[-1], follow=True)
    ok("回放完成")
