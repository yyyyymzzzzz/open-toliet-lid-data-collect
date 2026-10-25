#!/usr/bin/env python3
"""将采集失败时保全的 PNG 帧重建为 LeRobot dataset。"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from collect_open_toilet_lid import (  # noqa: E402
    build_dataset as build_lerobot_dataset,
    build_observation,
    compute_episode_actions,
)
from drag_teach_utils import banner, err, info, ok, warn  # noqa: E402


def recording_csv(ep_dir: Path) -> Path | None:
    """返回 episode 的轨迹 CSV，优先使用 recording.csv。"""
    for name in ("recording.csv", "recording_fixed.csv"):
        path = ep_dir / name
        if path.is_file():
            return path
    return None


def load_recording(path: Path) -> dict[str, list]:
    data = {"joints": [], "pose": [], "gripper": [], "t_rel": []}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "t_rel", "j0", "j1", "j2", "j3", "j4", "j5",
            "x", "y", "z", "rx", "ry", "rz", "gripper_pos",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV 缺少字段: {sorted(missing)}")
        for row in reader:
            data["joints"].append([float(row[f"j{i}"]) for i in range(6)])
            data["pose"].append([float(row[k]) for k in ("x", "y", "z", "rx", "ry", "rz")])
            data["gripper"].append(int(float(row["gripper_pos"])))
            data["t_rel"].append(float(row["t_rel"]))
    if not data["joints"]:
        raise ValueError(f"CSV 没有数据行: {path}")
    return data


def camera_dirs(dump_dir: Path) -> list[Path]:
    return sorted(path for path in dump_dir.iterdir() if path.is_dir())


def frame_index(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"无法从文件名解析帧号: {path.name}") from exc


def load_camera_frames(cam_dir: Path) -> dict[int, np.ndarray]:
    frames: dict[int, np.ndarray] = {}
    for path in sorted(cam_dir.glob("frame_*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            warn(f"跳过无法读取的图片: {path}")
            continue
        frames[frame_index(path)] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return frames


def discover_pending(base_dir: Path) -> list[Path]:
    pending = []
    for ep_dir in sorted(base_dir.iterdir()):
        if ep_dir.name == "datasets" or not ep_dir.is_dir():
            continue
        if (ep_dir / "frames_dump").is_dir() and recording_csv(ep_dir) is not None:
            pending.append(ep_dir)
    return pending


def inspect_frame_schema(ep_dirs: list[Path]) -> tuple[list[str], int, int]:
    camera_names: set[str] = set()
    expected_names: set[str] | None = None
    shape: tuple[int, int] | None = None
    for ep_dir in ep_dirs:
        dirs = camera_dirs(ep_dir / "frames_dump")
        current_names = {cam_dir.name for cam_dir in dirs}
        if expected_names is None:
            expected_names = current_names
        elif current_names != expected_names:
            raise ValueError(
                f"相机集合不一致: {ep_dir.name} 为 {sorted(current_names)}，"
                f"期望 {sorted(expected_names)}"
            )
        for cam_dir in dirs:
            camera_names.add(cam_dir.name)
            first = next(iter(sorted(cam_dir.glob("frame_*.png"))), None)
            if first is None:
                continue
            image = cv2.imread(str(first), cv2.IMREAD_COLOR)
            if image is None:
                continue
            current = image.shape[:2]
            if shape is None:
                shape = current
            elif current != shape:
                raise ValueError(f"图像分辨率不一致: {first} 为 {current}，期望 {shape}")
    if not camera_names or shape is None:
        raise ValueError("frames_dump 中没有有效 PNG 帧")
    return sorted(camera_names), shape[1], shape[0]


def build_one(ep_dir: Path, dataset, task: str, remove_dump: bool) -> bool:
    csv_path = recording_csv(ep_dir)
    if csv_path is None:
        err(f"缺少轨迹 CSV: {ep_dir}")
        return False

    recording = load_recording(csv_path)
    dump_dir = ep_dir / "frames_dump"
    names = [path.name for path in camera_dirs(dump_dir)]
    frames_by_camera = {name: load_camera_frames(dump_dir / name) for name in names}
    if not frames_by_camera or any(not frames for frames in frames_by_camera.values()):
        err("至少一路相机没有有效帧")
        return False

    common_indices = set.intersection(*(set(frames) for frames in frames_by_camera.values()))
    indices = sorted(index for index in common_indices if index < len(recording["joints"]))
    if len(indices) < 2:
        err(f"可对齐帧数过少: {len(indices)}")
        return False

    union_count = len(set.union(*(set(frames) for frames in frames_by_camera.values())))
    if len(indices) != union_count:
        warn(f"仅使用三路相机与 CSV 共同存在的帧: {len(indices)}/{union_count}")

    observations = []
    for index in indices:
        cam_frames = {name: frames[index] for name, frames in frames_by_camera.items()}
        observations.append(
            build_observation(
                recording["joints"][index],
                recording["pose"][index],
                [0.0] * 6,
                cam_frames,
                recording["gripper"][index],
                arm=None,
            )
        )

    actions = compute_episode_actions(observations)
    try:
        for observation, action in zip(observations, actions):
            frame = {key: value for key, value in observation.items() if not key.startswith("_")}
            frame["action"] = action
            frame["task"] = task
            dataset.add_frame(frame)
        dataset.save_episode(parallel_encoding=False)
    except Exception:
        try:
            dataset.clear_episode_buffer(delete_images=True)
        except Exception:
            pass
        raise

    ok(f"{ep_dir.name}: 已重建 {len(observations)} 帧")
    if remove_dump:
        shutil.rmtree(dump_dir)
        info(f"已删除备份帧: {dump_dir}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="episodes")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--repo-id", default="local/open_toilet_lid_v1")
    parser.add_argument("--task", default="Open the toilet lid")
    parser.add_argument("--freq", type=int, default=30)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remove-frames-dump", action="store_true")
    args = parser.parse_args()
    if args.freq <= 0:
        parser.error("--freq 必须大于 0")
    return args


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    if not base_dir.is_dir():
        err(f"episodes 目录不存在: {base_dir}")
        sys.exit(1)

    pending = discover_pending(base_dir)
    if not pending:
        ok("没有待重建的 frames_dump")
        return

    banner("离线重建 Open Toilet Lid 数据集")
    for ep_dir in pending:
        info(ep_dir.name)
    if args.dry_run:
        return

    try:
        names, width, height = inspect_frame_schema(pending)
    except ValueError as exc:
        err(str(exc))
        sys.exit(1)

    dataset_args = SimpleNamespace(
        repo_id=args.repo_id,
        freq=args.freq,
        vcodec=args.vcodec,
        cam_height=height,
        cam_width=width,
        dataset_root=args.dataset_root or str(base_dir / "datasets"),
    )
    info(f"相机: {names}; 分辨率: {width}x{height}")
    info(f"dataset: {dataset_args.dataset_root}")
    dataset = build_lerobot_dataset(dataset_args, names)
    if dataset is None:
        sys.exit(1)

    succeeded = failed = 0
    for ep_dir in pending:
        try:
            if build_one(ep_dir, dataset, args.task, args.remove_frames_dump):
                succeeded += 1
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            err(f"{ep_dir.name}: {exc}")
            traceback.print_exc()
    try:
        dataset.finalize()
    except Exception as exc:
        warn(f"dataset finalize 失败: {exc}")
        failed += 1
    banner(f"完成: {succeeded} 成功, {failed} 失败")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
