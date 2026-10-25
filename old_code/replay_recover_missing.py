#!/usr/bin/env python3
"""Safely replay missing demonstrations and append them to a recovered dataset.

The next source trajectory is selected from the dataset's current episode count,
so restarting this program resumes without duplicating already-saved episodes.
Every episode is saved synchronously and immediately finalized.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import shutil
import signal
import sys
import termios
import threading
import time
import tty
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
from lerobot.configs.video import VideoEncoderConfig
from lerobot.datasets import LeRobotDataset
import pyarrow as pa
import pyarrow.parquet as pq

from collect_open_toilet_lid import (
    build_cameras,
    build_observation,
    close_cameras,
    read_arm_obs,
    read_cameras,
    save_lerobot_episode,
)
from drag_teach_utils import (
    banner,
    err,
    info,
    load_recording,
    move_to_home,
    ok,
    replay_csv,
    setup_tool_frame,
    warn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("episodes_backup_before_recovery_20260822"))
    parser.add_argument("--dataset-root", type=Path, default=Path("episodes_recovered/datasets"))
    parser.add_argument("--repo-id", default="local/open_toilet_lid_v1")
    parser.add_argument("--target-episodes", type=int, default=100)
    parser.add_argument("--task", default="Open the toilet lid")
    parser.add_argument("--ip", default="192.168.1.18")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--freq", type=int, default=30)
    parser.add_argument("--home-speed", type=int, default=60)
    parser.add_argument("--resample-dt", type=float, default=0.010)
    parser.add_argument("--smooth-window", type=int, default=11)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--cam-fps", type=int, default=30)
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--sn-top", default="CP0P76300026")
    parser.add_argument("--sn-bottom", default="CP0N7630000A")
    parser.add_argument("--sn-wrist", default="CP0N763000HE")
    parser.add_argument("--allow-partial-cameras", action="store_true")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.home_speed <= 100:
        parser.error("--home-speed must be in 1..100")
    if args.freq <= 0 or args.target_episodes <= 0:
        parser.error("--freq and --target-episodes must be positive")
    return args


def raw_episode_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name[:8].isdigit() and (path / "recording.csv").is_file()
    )


def current_episode_count(dataset_root: Path) -> int:
    return int(json.loads((dataset_root / "meta" / "info.json").read_text())["total_episodes"])


def normalize_episode_metadata_schemas(dataset_root: Path) -> int:
    """Make all episode parquet fragments use the richest available schema.

    The recovered first 73 rows intentionally contain only routing columns,
    while newly recorded rows also contain per-episode stats. Hugging Face
    Datasets requires every parquet fragment to expose identical columns.
    Missing stats in recovered rows are represented by typed null arrays.
    """
    paths = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if len(paths) < 2:
        return 0
    schemas = {path: pq.read_schema(path) for path in paths}
    template = max(schemas.values(), key=lambda schema: len(schema.names))
    rewritten = 0
    for path, schema in schemas.items():
        if schema.names == template.names and all(
            schema.field(name).type == template.field(name).type for name in template.names
        ):
            continue
        table = pq.read_table(path)
        arrays = []
        for field in template:
            if field.name in table.column_names:
                column = table[field.name]
                if column.type != field.type:
                    column = column.cast(field.type)
                arrays.append(column)
            else:
                arrays.append(pa.nulls(table.num_rows, type=field.type))
        normalized = pa.Table.from_arrays(arrays, schema=template)
        backup = path.with_name(f"{path.name}.before_schema_fix")
        if not backup.exists():
            shutil.copy2(path, backup)
        temporary = path.with_name(f"{path.name}.schema_fix_tmp")
        pq.write_table(normalized, temporary, compression="snappy", use_dictionary=True)
        os.replace(temporary, path)
        rewritten += 1
    return rewritten


def resume_dataset(args: argparse.Namespace) -> LeRobotDataset:
    return LeRobotDataset.resume(
        repo_id=args.repo_id,
        root=args.dataset_root,
        force_cache_sync=False,
        video_backend="pyav",
        image_writer_processes=0,
        image_writer_threads=4,
        camera_encoder=VideoEncoderConfig(
            vcodec=args.vcodec,
            pix_fmt="yuv420p",
            preset="ultrafast",
        ),
    )


class CBreak:
    def __enter__(self):
        self.settings = termios.tcgetattr(sys.stdin)
        # Drop stale keys (especially an Enter typed out of habit after r/s),
        # so they cannot confirm the next physical movement prompt.
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *_):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

    @staticmethod
    def key() -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def capture_replay(
    arm,
    cameras: dict,
    csv_path: Path,
    args: argparse.Namespace,
) -> tuple[list[dict], bool, str | None]:
    joints, grippers, t_rels = load_recording(csv_path)
    stop_flag = [False]
    done = threading.Event()
    replay_error: list[str] = []
    frames: list[dict] = []

    def run_replay() -> None:
        try:
            replay_csv(
                arm,
                None,
                joints,
                grippers,
                t_rels,
                stop_flag=stop_flag,
                resample_dt=args.resample_dt,
                smooth_window=args.smooth_window,
                initial_gripper=0,
                gripper_feedback=[0],
            )
        except Exception as exc:
            replay_error.append(str(exc))
        finally:
            done.set()

    worker = threading.Thread(target=run_replay, daemon=False)
    worker.start()
    aborted = False
    dt = 1.0 / args.freq
    next_tick = time.perf_counter()
    with CBreak():
        while not done.is_set():
            key = CBreak.key()
            if key in ("b", "B", "q", "Q", "\x1b"):
                aborted = True
                stop_flag[0] = True
                arm.rm_set_arm_slow_stop()
                warn("收到中止键，机械臂减速停止")
                break

            joints_now, pose_now, force_now = read_arm_obs(arm)
            if joints_now is not None and pose_now is not None and force_now is not None:
                camera_frames = read_cameras(cameras)
                if len(camera_frames) == len(cameras):
                    frames.append(
                        build_observation(
                            joints_now,
                            pose_now,
                            force_now,
                            camera_frames,
                            gripper_pos=0,
                            arm=arm,
                        )
                    )
            next_tick += dt
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()

    if aborted:
        stop_flag[0] = True
    worker.join(timeout=5.0)
    if worker.is_alive():
        arm.rm_set_arm_slow_stop()
        return frames, True, "replay thread did not stop within 5 seconds"
    return frames, aborted, replay_error[0] if replay_error else None


def prompt_choice(message: str, allowed: set[str], enter_value: str | None = None) -> str:
    """Read one key in cbreak mode; r/s/q take effect without Enter."""
    print(message, end="", flush=True)
    with CBreak():
        while True:
            value = sys.stdin.read(1).lower()
            if value in ("\r", "\n") and enter_value is not None:
                print()
                return enter_value
            if value in allowed:
                print(value)
                return value


def main() -> None:
    args = parse_args()
    args.raw_root = args.raw_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    sources = raw_episode_dirs(args.raw_root)
    repaired = normalize_episode_metadata_schemas(args.dataset_root)
    if repaired:
        warn(f"已统一 {repaired} 个 Episode 元数据文件的 schema")
    completed = current_episode_count(args.dataset_root)
    if completed > args.target_episodes:
        raise RuntimeError(f"dataset already has {completed} episodes, target is {args.target_episodes}")
    if len(sources) < args.target_episodes:
        raise RuntimeError(f"only {len(sources)} source trajectories, need {args.target_episodes}")

    banner("Open Toilet Lid — 缺失 Episode 安全回放恢复")
    info(f"dataset 当前: {completed}/{args.target_episodes}")
    info(f"待回放: source #{completed + 1} .. #{args.target_episodes}")
    if args.dry_run:
        for index in range(completed, args.target_episodes):
            print(f"dataset ep{index:03d} <- {sources[index].name}")
        return
    if not sys.stdin.isatty():
        raise RuntimeError("physical replay requires an interactive terminal")

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    cameras: dict = {}
    interrupted = [False]

    def emergency_stop(_signum, _frame) -> None:
        interrupted[0] = True
        try:
            arm.rm_set_arm_slow_stop()
        except Exception:
            pass

    signal.signal(signal.SIGINT, emergency_stop)
    signal.signal(signal.SIGTERM, emergency_stop)

    try:
        info(f"连接机械臂 {args.ip}:{args.port} ...")
        arm.rm_create_robot_arm(args.ip, args.port)
        state = arm.rm_get_current_arm_state()
        if state[0] != 0:
            raise RuntimeError(f"机械臂连接失败: code {state[0]}")
        setup_tool_frame(arm)
        cameras = build_cameras(args)
        expected = 3
        if len(cameras) != expected:
            raise RuntimeError(f"相机必须全部在线: expected {expected}, got {len(cameras)}")
        ok("机械臂和三路相机已就绪")

        while not interrupted[0]:
            episode_index = current_episode_count(args.dataset_root)
            if episode_index >= args.target_episodes:
                break
            source = sources[episode_index]
            source_meta = json.loads((source / "meta.json").read_text())
            home = source_meta.get("episode_home_deg")
            if not home or len(home) != 6:
                joints, _, _ = load_recording(source / "recording.csv")
                home = joints[0]

            banner(f"准备恢复 Episode {episode_index + 1}/{args.target_episodes}")
            info(f"示教来源: {source.name}")
            info(f"原示教帧数: {source_meta.get('frame_count')}")
            choice = prompt_choice(
                "确认机械臂回位路径无障碍后按 Enter；输入 q 保存进度并退出: ",
                {"q"},
                enter_value="go",
            )
            if choice == "q":
                break
            if not move_to_home(arm, home, speed=args.home_speed):
                err("归位失败，本条未回放")
                continue

            choice = prompt_choice(
                "请关闭/复位马桶盖并离开机械臂运动范围；输入 r 开始回放，q 退出: ",
                {"r", "q"},
            )
            if choice == "q":
                break

            info("开始回放并采集；按 b 可随时中止")
            frames, aborted, replay_error = capture_replay(
                arm, cameras, source / "recording.csv", args
            )
            if replay_error:
                err(f"回放异常: {replay_error}")
            if aborted or replay_error or len(frames) < 2:
                warn(f"本次回放未保存（采集 {len(frames)} 帧）")
                continue

            ok(f"回放完成，采集 {len(frames)} 帧")
            choice = prompt_choice(
                "输入 s 保存本条，r 放弃本次并重新回放，q 放弃本次并退出: ",
                {"s", "r", "q"},
            )
            if choice == "q":
                break
            if choice == "r":
                continue

            dataset = resume_dataset(args)
            try:
                if dataset.num_episodes != episode_index:
                    raise RuntimeError(
                        f"保存前 episode 数变化: expected {episode_index}, got {dataset.num_episodes}"
                    )
                success, message = save_lerobot_episode(
                    frames,
                    dataset,
                    args.task,
                    progress_cb=lambda text: info(f"保存: {text}"),
                    episode_dir=source,
                    dataset_root=args.dataset_root,
                )
                if not success:
                    raise RuntimeError(message)
            finally:
                # This writes both parquet footers before the next physical run.
                dataset.finalize()

            repaired = normalize_episode_metadata_schemas(args.dataset_root)
            if repaired:
                info(f"已统一 {repaired} 个新增 Episode 元数据文件的 schema")

            saved_count = current_episode_count(args.dataset_root)
            if saved_count != episode_index + 1:
                raise RuntimeError(
                    f"保存后 episode 数异常: expected {episode_index + 1}, got {saved_count}"
                )
            ok(f"Episode {episode_index + 1} 已完整落盘并 finalize ({saved_count}/{args.target_episodes})")

        completed = current_episode_count(args.dataset_root)
        if completed == args.target_episodes:
            banner(f"恢复完成：{completed} episodes")
        else:
            banner(f"安全退出：已保存 {completed}/{args.target_episodes} episodes，可再次运行续传")
    finally:
        try:
            arm.rm_set_arm_slow_stop()
        except Exception:
            pass
        close_cameras(cameras)
        try:
            arm.rm_delete_robot_arm()
        except Exception:
            pass


if __name__ == "__main__":
    main()
