#!/usr/bin/env python3
"""Rebuild the first finalized episodes after an interrupted LeRobot finalize.

The original concatenated MP4 files are copied byte-for-byte.  Numeric state is
reconstructed from each demonstration CSV and resampled to the exact replay
frame count stored in meta.json.  Force is set to zero because it was only
present in the unfinalized parquet stream.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import av
import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.feature_utils import get_hf_features_from_features


CAMERAS = ("cam_top", "cam_bottom", "cam_wrist")
TASK = "Open the toilet lid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=73)
    return parser.parse_args()


def raw_episode_dirs(raw_root: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_root.iterdir()
        if path.is_dir() and path.name[:8].isdigit() and (path / "recording.csv").is_file()
    )


def load_csv(path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    required = ["t_rel", *[f"j{i}" for i in range(6)], "x", "y", "z", "rx", "ry", "rz"]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return {name: frame[name].to_numpy(dtype=np.float64) for name in required}


def resample_state(csv_data: dict[str, np.ndarray], length: int) -> np.ndarray:
    source_t = csv_data["t_rel"] - csv_data["t_rel"][0]
    if len(source_t) < 2 or source_t[-1] <= 0:
        raise ValueError("recording CSV has no usable time span")
    target_t = np.linspace(0.0, float(source_t[-1]), length)

    joints = np.column_stack(
        [np.interp(target_t, source_t, csv_data[f"j{i}"]) for i in range(6)]
    )
    xyz = np.column_stack(
        [np.interp(target_t, source_t, csv_data[name]) for name in ("x", "y", "z")]
    )
    euler_source = np.unwrap(
        np.column_stack([csv_data[name] for name in ("rx", "ry", "rz")]), axis=0
    )
    euler = np.column_stack(
        [np.interp(target_t, source_t, euler_source[:, i]) for i in range(3)]
    )
    xyzw = Rotation.from_euler("xyz", euler, degrees=False).as_quat()
    quat_wxyz = xyzw[:, [3, 0, 1, 2]]
    gripper = np.zeros((length, 1), dtype=np.float64)
    return np.column_stack([joints, xyz, euler, quat_wxyz, gripper]).astype(np.float32)


def compute_actions(state: np.ndarray) -> np.ndarray:
    actions = np.zeros((len(state), 7), dtype=np.float32)
    if len(state) < 2:
        return actions
    actions[:-1, :3] = state[1:, 6:9] - state[:-1, 6:9]
    current = Rotation.from_quat(state[:-1, [13, 14, 15, 12]])
    following = Rotation.from_quat(state[1:, [13, 14, 15, 12]])
    actions[:-1, 3:6] = (following * current.inv()).as_euler("xyz").astype(np.float32)
    return actions


def ffprobe_frames(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
            "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return int(result.stdout.strip())


def video_timestamps(path: Path) -> tuple[np.ndarray, float]:
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        timestamps = np.asarray(
            [float(frame.pts * stream.time_base) for frame in container.decode(stream)],
            dtype=np.float64,
        )
        duration = float(stream.duration * stream.time_base)
        return timestamps, duration
    finally:
        container.close()


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    source = args.source_dataset.resolve()
    output = args.output_dataset.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    episodes = raw_episode_dirs(raw_root)[: args.episodes]
    if len(episodes) != args.episodes:
        raise RuntimeError(f"expected {args.episodes} raw episodes, found {len(episodes)}")

    lengths: list[int] = []
    for path in episodes:
        meta = json.loads((path / "meta.json").read_text())
        length = int(meta["replay_frame_count"])
        if length < 2:
            raise ValueError(f"invalid replay_frame_count in {path}")
        lengths.append(length)
    total_frames = sum(lengths)

    info = json.loads((source / "meta" / "info.json").read_text())
    if int(info["total_episodes"]) != args.episodes or int(info["total_frames"]) != total_frames:
        raise RuntimeError(
            f"metadata mismatch: info={info['total_episodes']}/{info['total_frames']}, "
            f"raw={args.episodes}/{total_frames}"
        )

    video_sources = {
        camera: source / "videos" / f"observation.images.{camera}" / "chunk-000" / "file-000.mp4"
        for camera in CAMERAS
    }
    camera_timestamps: dict[str, np.ndarray] = {}
    camera_durations: dict[str, float] = {}
    for camera, path in video_sources.items():
        count = ffprobe_frames(path)
        if count != total_frames:
            raise RuntimeError(f"{camera}: video has {count} frames, expected {total_frames}")
        camera_timestamps[camera], camera_durations[camera] = video_timestamps(path)
    reference_timestamps = camera_timestamps[CAMERAS[0]]
    for camera in CAMERAS[1:]:
        if not np.allclose(camera_timestamps[camera], reference_timestamps, atol=1e-9, rtol=0):
            raise RuntimeError(f"{camera}: frame timestamps differ from {CAMERAS[0]}")

    (output / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (output / "data" / "chunk-000").mkdir(parents=True)
    for camera in CAMERAS:
        destination = output / "videos" / f"observation.images.{camera}" / "chunk-000"
        destination.mkdir(parents=True)
        shutil.copy2(video_sources[camera], destination / "file-000.mp4")
    shutil.copy2(source / "meta" / "tasks.parquet", output / "meta" / "tasks.parquet")

    # A low limit makes every newly replayed episode start a separate video file,
    # so a later interruption cannot damage the recovered file-000.mp4.
    info["video_files_size_in_mb"] = 1
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))

    features = {
        key: {**feature, "shape": tuple(feature["shape"])}
        for key, feature in info["features"].items()
    }
    hf_features = get_hf_features_from_features(features)
    data_path = output / "data" / "chunk-000" / "file-000.parquet"
    parquet_writer: pq.ParquetWriter | None = None
    episode_rows: list[dict] = []
    all_numeric: dict[str, list[np.ndarray]] = {key: [] for key in hf_features}
    global_index = 0
    video_start_index = 0

    try:
        for episode_index, (episode_dir, length) in enumerate(zip(episodes, lengths, strict=True)):
            state = resample_state(load_csv(episode_dir / "recording.csv"), length)
            action = compute_actions(state)
            frame_index = np.arange(length, dtype=np.int64)
            values = {
                "observation.state": state,
                "observation.force": np.zeros((length, 6), dtype=np.float32),
                "action": action,
                "timestamp": (frame_index / int(info["fps"])).astype(np.float32),
                "frame_index": frame_index,
                "episode_index": np.full(length, episode_index, dtype=np.int64),
                "index": np.arange(global_index, global_index + length, dtype=np.int64),
                "task_index": np.zeros(length, dtype=np.int64),
            }
            hf_dataset = datasets.Dataset.from_dict(values, features=hf_features, split="train")
            table = hf_dataset.with_format("arrow")[:]
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(
                    data_path, schema=table.schema, compression="snappy", use_dictionary=True
                )
            parquet_writer.write_table(table)
            for key, value in values.items():
                all_numeric[key].append(value)

            row = {
                "episode_index": episode_index,
                "tasks": [TASK],
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": global_index,
                "dataset_to_index": global_index + length,
            }
            for camera in CAMERAS:
                prefix = f"videos/observation.images.{camera}"
                timestamps = camera_timestamps[camera]
                from_timestamp = float(timestamps[video_start_index])
                next_index = video_start_index + length
                to_timestamp = (
                    float(timestamps[next_index])
                    if next_index < len(timestamps)
                    else camera_durations[camera]
                )
                row[f"{prefix}/chunk_index"] = 0
                row[f"{prefix}/file_index"] = 0
                row[f"{prefix}/from_timestamp"] = from_timestamp
                row[f"{prefix}/to_timestamp"] = to_timestamp
            row["meta/episodes/chunk_index"] = 0
            row["meta/episodes/file_index"] = 0
            episode_rows.append(row)
            global_index += length
            video_start_index += length
            print(f"[{episode_index + 1:02d}/{args.episodes}] {episode_dir.name}: {length} frames")
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    pd.DataFrame(episode_rows).to_parquet(
        output / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        index=False,
    )

    numeric_arrays = {key: np.concatenate(parts, axis=0) for key, parts in all_numeric.items()}
    numeric_stats = compute_episode_stats(numeric_arrays, features)
    old_stats = json.loads((source / "meta" / "stats.json").read_text())
    recovered_stats = {
        **{key: value for key, value in old_stats.items() if key.startswith("observation.images.")},
        **serializable(numeric_stats),
    }
    (output / "meta" / "stats.json").write_text(
        json.dumps(recovered_stats, indent=2, ensure_ascii=False)
    )

    report = {
        "status": "recovered_first_episodes",
        "episodes": args.episodes,
        "frames": total_frames,
        "video_handling": "copied byte-for-byte from interrupted dataset",
        "state_handling": "resampled from demonstration recording.csv to replay_frame_count",
        "force_handling": "filled with zeros; original force existed only in unfinalized parquet",
        "source_dataset": str(source),
        "raw_root": str(raw_root),
    }
    (output / "RECOVERY_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Recovered {args.episodes} episodes / {total_frames} frames into {output}")


if __name__ == "__main__":
    main()
