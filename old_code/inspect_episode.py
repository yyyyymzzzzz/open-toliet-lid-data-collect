#!/usr/bin/env python3
"""
inspect_episode.py — open_toilet_lid 数据检查工具

扫描 episodes/ 下所有录制目录，交互式选择 episode 后查看：
  · 录制摘要（帧数、时长、拖动模式等）
  · 关节角 / 末端位姿极值统计
  · 末端轨迹 3D 图（导出 PNG）
  · 关节角曲线 6 轴图（导出 PNG）
  · 相机截图导航
  · 导出 CSV 分析摘要

用法：
  python inspect_episode.py
  python inspect_episode.py --episode 20260810_203532       # 跳过菜单直接加载
  python inspect_episode.py --episode 20260810_203532 --export-plot  # 导出图表
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

EPISODES_DIR = Path(__file__).parent / "episodes"

ANSI_BOLD   = "\033[1m"
ANSI_GREEN  = "\033[32m"
ANSI_CYAN   = "\033[36m"
ANSI_YELLOW = "\033[33m"
ANSI_RED    = "\033[31m"
ANSI_RESET  = "\033[0m"


def c(text: str, code: str) -> str:
    return f"{code}{text}{ANSI_RESET}"


def banner(text: str) -> None:
    print(f"\n{c('─' * 80, ANSI_CYAN)}")
    print(f"{c(text, ANSI_BOLD)}")
    print(f"{c('─' * 80, ANSI_CYAN)}")


# ── Episode discovery ─────────────────────────────────────────────────────────

def discover_episodes(base: Path) -> list[Path]:
    """Return sorted list of valid episode directories under *base*."""
    episodes = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and p.name != "datasets" and (p / "recording.csv").exists():
            episodes.append(p)
    return episodes


def load_meta(ep_dir: Path) -> dict:
    meta_path = ep_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def load_csv(ep_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load recording.csv → (joints[N×6], pose[N×6], t[N], gripper[N]).

    CSV columns: t_rel, j0..j5, x,y,z, rx,ry,rz, gripper_pos
    """
    with open(ep_dir / "recording.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    data = np.array([[float(v) for v in row] for row in rows])
    t       = data[:, 0]
    joints  = data[:, 1:7]    # j0..j5  (deg)
    pose    = data[:, 7:13]   # x,y,z,rx,ry,rz  (m, rad)
    gripper = data[:, 13]
    return joints, pose, t, gripper


# ── Interactive picker ────────────────────────────────────────────────────────

def pick_episode(episodes: list[Path]) -> Path | None:
    """Interactive picker with arrow-key / j-k navigation."""
    n = len(episodes)
    if n == 0:
        print(f"{c('✗  没有找到任何 episode', ANSI_RED)}")
        return None
    if n == 1:
        return episodes[0]

    idx = 0
    try:
        while True:
            _redraw_menu(episodes, idx)
            key = _read_key_raw()

            if key == "\x1b":
                return None
            if key in ("\n", "\r"):  # Enter in raw mode = '\r'
                return episodes[idx]
            if key in ("\x1b[A", "k") and idx > 0:
                idx -= 1
            if key in ("\x1b[B", "j") and idx < n - 1:
                idx += 1
            if key in ("q", "Q"):
                return None
    except (KeyboardInterrupt, EOFError):
        print(f"\n{c('已取消', ANSI_YELLOW)}")
        return None


def _redraw_menu(episodes: list[Path], selected: int) -> None:
    sys.stdout.write("\033[2J\033[H")
    banner("可用 Episode 列表")
    print(f"  {c('↑↓ 或 j/k 选择   Enter 确认   q/ESC 退出', ANSI_YELLOW)}\n")
    for i, ep in enumerate(episodes):
        meta = load_meta(ep)
        frames = meta.get("frame_count", len(episodes))
        dur = meta.get("duration_s", "?")
        dur_s = f"{float(dur):.1f}s" if isinstance(dur, (int, float)) else str(dur)

        prefix = f"{c('▶', ANSI_GREEN)} " if i == selected else "  "
        bold   = ANSI_BOLD if i == selected else ""
        reset  = ANSI_RESET if i == selected else ""
        print(f"{prefix}{bold}{ep.name:20s}  {c(str(frames)+' 帧', ANSI_CYAN)}  {dur_s}{reset}")
    print()


def _read_key_raw(timeout: float = 1.0) -> str:
    """Read a single keystroke from stdin in raw mode.  *timeout* in seconds.

    Ctrl+C is translated to a ``KeyboardInterrupt`` so that the process remains
    interruptible even while the terminal is in raw mode.
    """
    import select, termios, tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            ch = os.read(fd, 1)
            if ch == b"\x03":                     # Ctrl+C
                raise KeyboardInterrupt
            if ch == b"\x1b":
                r2, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r2:
                    rest = os.read(fd, 5)
                    return "\x1b" + rest.decode(errors="replace")
                return "\x1b"
            return ch.decode(errors="replace")
        return ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── Statistics ────────────────────────────────────────────────────────────────

def print_stats(joints: np.ndarray, pose: np.ndarray, t: np.ndarray,
                gripper: np.ndarray, meta: dict) -> None:
    banner("关节角统计 (deg)")
    jnames = ["J0", "J1", "J2", "J3", "J4", "J5"]
    print(f"{'轴':>4s}  {'最小':>10s}  {'最大':>10s}  {'行程':>10s}  {'均值':>10s}")
    print(f"{'':>4s}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
    for i, name in enumerate(jnames):
        col = joints[:, i]
        print(f"{name:>4s}  {col.min():10.2f}  {col.max():10.2f}  "
              f"{np.ptp(col):10.2f}  {col.mean():10.2f}")

    banner("末端位姿统计")
    pos_labels = ["X (m)", "Y (m)", "Z (m)"]
    rot_labels = ["RX (rad)", "RY (rad)", "RZ (rad)"]
    all_labels = pos_labels + rot_labels
    print(f"{'':>10s}  {'最小':>10s}  {'最大':>10s}  {'行程':>10s}  {'均值':>10s}")
    print(f"{'':>10s}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
    for i, lbl in enumerate(all_labels):
        col = pose[:, i]
        print(f"{lbl:>10s}  {col.min():10.4f}  {col.max():10.4f}  "
              f"{np.ptp(col):10.4f}  {col.mean():10.4f}")

    duration = t[-1] - t[0]
    frame_count = meta.get("frame_count", len(joints))
    print(f"\n  总帧数: {frame_count}  录制时长: {duration:.1f}s  "
          f"频率: {len(joints)/duration:.1f} Hz")
    print(f"  拖动模式: {meta.get('drag_mode', '?')}  夹爪: 固定=0")

    # Trajectory span summary
    xyz_span = np.ptp(pose[:, :3], axis=0)
    print(f"  末端位移:  Δx={xyz_span[0]:.3f}m  Δy={xyz_span[1]:.3f}m  "
          f"Δz={xyz_span[2]:.3f}m  总行程={np.linalg.norm(xyz_span):.3f}m")


# ── ASCII trajectory plot ─────────────────────────────────────────────────────

def ascii_trajectory(pose: np.ndarray) -> None:
    """Simple ASCII side-view of XY and XZ trajectories."""
    banner("末端轨迹 XY / XZ (ASCII)")
    xy = pose[:, :2]
    xz = pose[:, [0, 2]]

    def _render(data: np.ndarray, title: str, w: int = 60, h: int = 20) -> None:
        xmin, xmax = data[:, 0].min(), data[:, 0].max()
        ymin, ymax = data[:, 1].min(), data[:, 1].max()
        xr = xmax - xmin or 1e-6
        yr = ymax - ymin or 1e-6

        canvas = [[" "] * w for _ in range(h)]
        for px, py in data:
            col = int((px - xmin) / xr * (w - 1))
            row = int((1 - (py - ymin) / yr) * (h - 1))
            row = max(0, min(h - 1, row))
            col = max(0, min(w - 1, col))
            canvas[row][col] = "·"

        print(f"\n  {title}")
        print(f"  X: [{xmin:.3f} .. {xmax:.3f}] m")
        print(f"  Y/Z: [{ymin:.3f} .. {ymax:.3f}] m")
        print(f"  ┌{'─' * w}┐")
        for row in canvas:
            print(f"  │{''.join(row)}│")
        print(f"  └{'─' * w}┘")

    _render(xy, "XY 平面 (俯视)")
    _render(xz, "XZ 平面 (侧视)")


# ── Camera frames ─────────────────────────────────────────────────────────────

def show_cam_frames(ep_dir: Path) -> None:
    cam_dir = ep_dir / "cam_frames"
    if not cam_dir.exists():
        print(f"\n  {c('无相机截图数据', ANSI_YELLOW)}")
        return

    frames: dict[str, list[str]] = {}
    for f in sorted(cam_dir.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            parts = f.stem.split("_")
            if len(parts) >= 3:
                cam_name = "_".join(parts[2:])
                frames.setdefault(cam_name, []).append(f.name)

    banner("相机截图")
    for cam_name, flist in sorted(frames.items()):
        fnums = []
        for fn in flist:
            try:
                fnums.append(int(fn.split("_")[1]))
            except (IndexError, ValueError):
                pass
        fnums.sort()
        step = fnums[1] - fnums[0] if len(fnums) >= 2 else 0
        print(f"  {c(cam_name, ANSI_CYAN):20s}  {len(flist)} 张  "
              f"(每 {step} 帧一张)  "
              f"{fnums[0]:06d} → {fnums[-1]:06d}")

    print(f"\n  图片目录: {cam_dir}")
    print(f"  查看所有图片: {c(f'ls {cam_dir}', ANSI_GREEN)}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_trajectory_3d(pose: np.ndarray, ep_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"{c('⚠  matplotlib 未安装', ANSI_YELLOW)}")
        return None

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = pose[:, 0], pose[:, 1], pose[:, 2]
    colors = np.linspace(0, 1, len(x))
    sc = ax.scatter(x, y, z, c=colors, cmap="viridis", s=6, alpha=0.8)
    ax.plot(x, y, z, linewidth=0.6, color="gray", alpha=0.4)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"End-Effector Trajectory — {ep_dir.name}")
    fig.colorbar(sc, ax=ax, shrink=0.5, label="Frame index")
    out = ep_dir / "trajectory_3d.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_joints(joints: np.ndarray, t: np.ndarray, ep_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    jnames = ["J0", "J1", "J2", "J3", "J4", "J5"]
    for i, ax in enumerate(axes.flat):
        ax.plot(t, joints[:, i], linewidth=0.5)
        ax.set_ylabel(f"{jnames[i]} (deg)")
        ax.grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")
    fig.suptitle(f"Joint Angles — {ep_dir.name}", fontsize=14)
    fig.tight_layout()
    out = ep_dir / "joints.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── CSV export ────────────────────────────────────────────────────────────────

def export_summary_csv(ep_dir: Path, joints: np.ndarray, pose: np.ndarray) -> None:
    summary = ep_dir / "analysis.csv"
    names = ["j0","j1","j2","j3","j4","j5","x","y","z","rx","ry","rz"]
    data = np.hstack([joints, pose])
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + names)
        w.writerow(["min"]    + [f"{v:.4f}" for v in data.min(axis=0)])
        w.writerow(["max"]    + [f"{v:.4f}" for v in data.max(axis=0)])
        w.writerow(["range"]  + [f"{v:.4f}" for v in np.ptp(data, axis=0)])
        w.writerow(["mean"]   + [f"{v:.4f}" for v in data.mean(axis=0)])
        w.writerow(["std"]    + [f"{v:.4f}" for v in data.std(axis=0)])
    print(f"{c('✓', ANSI_GREEN)} 分析摘要已导出: {summary}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="open_toilet_lid 数据检查工具")
    ap.add_argument("--episode",     default="",  help="直接指定 episode 目录名")
    ap.add_argument("--export-plot", action="store_true", help="导出 PNG 图表")
    ap.add_argument("--export-csv",  action="store_true", help="导出 CSV 摘要")
    ap.add_argument("--ascii-plot",  action="store_true", help="ASCII 轨迹图")
    ap.add_argument("--base-dir",    default=str(EPISODES_DIR),
                    help=f"episodes 根目录 (默认: {EPISODES_DIR})")
    args = ap.parse_args()

    base = Path(args.base_dir)
    if not base.exists():
        print(f"{c(f'✗  目录不存在: {base}', ANSI_RED)}")
        sys.exit(1)

    episodes = discover_episodes(base)
    if args.episode:
        ep_dir = base / args.episode
        if not ep_dir.exists() or not (ep_dir / "recording.csv").exists():
            print(f"{c(f'✗  episode 不存在或缺少 recording.csv: {ep_dir}', ANSI_RED)}")
            sys.exit(1)
    else:
        ep_dir = pick_episode(episodes)
        if ep_dir is None:
            print(f"{c('已取消', ANSI_YELLOW)}")
            return

    meta   = load_meta(ep_dir)
    joints, pose, t, gripper = load_csv(ep_dir)

    # ── Summary header ────────────────────────────────────────────────────
    banner(f"Episode: {c(ep_dir.name, ANSI_GREEN)}")
    print(f"  路径:         {ep_dir}")
    print(f"  录制帧数:     {len(joints)}  (recorded: {meta.get('frame_count', '?')})")
    print(f"  录制时长:     {meta.get('duration_s', t[-1]-t[0] if len(t)>0 else 0):.1f}s")
    print(f"  拖动模式:     {meta.get('drag_mode', '?')}")
    home = meta.get("episode_home_deg")
    if home:
        print(f"  home 关节:    [{', '.join(f'{v:.1f}' for v in home)}]")

    # ── Stats ─────────────────────────────────────────────────────────────
    print_stats(joints, pose, t, gripper, meta)

    # ── ASCII trajectory ──────────────────────────────────────────────────
    if args.ascii_plot:
        ascii_trajectory(pose)

    # ── Camera frames ─────────────────────────────────────────────────────
    show_cam_frames(ep_dir)

    # ── Export ────────────────────────────────────────────────────────────
    if args.export_plot:
        print()
        p1 = plot_trajectory_3d(pose, ep_dir)
        if p1:
            print(f"{c('✓', ANSI_GREEN)} 末端轨迹 3D: {p1}")
        p2 = plot_joints(joints, t, ep_dir)
        if p2:
            print(f"{c('✓', ANSI_GREEN)} 关节角曲线:  {p2}")

    if args.export_csv:
        export_summary_csv(ep_dir, joints, pose)

    print()


if __name__ == "__main__":
    main()
