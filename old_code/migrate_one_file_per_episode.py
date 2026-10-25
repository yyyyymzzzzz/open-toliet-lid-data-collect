#!/usr/bin/env python3
"""Migrate a LeRobot v3 dataset to one data/meta/video file per episode.

The source dataset is never modified. H.264 video packets are stream-copied,
not re-encoded, and packet payload hashes are compared after segmentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


CAMERAS = ("cam_top", "cam_bottom", "cam_wrist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def ffprobe_frames(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return int(result.stdout.strip())


def packet_hashes(path: Path, lengths: list[int]) -> list[str]:
    """Hash H.264 packet payloads, split at the supplied episode lengths."""
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        result: list[str] = []
        episode = 0
        count = 0
        digest = hashlib.sha256()
        for packet in container.demux(stream):
            if packet.dts is None or packet.size == 0:
                continue
            if episode >= len(lengths):
                raise RuntimeError(f"too many video packets in {path}")
            digest.update(bytes(packet))
            count += 1
            if count == lengths[episode]:
                result.append(digest.hexdigest())
                episode += 1
                count = 0
                digest = hashlib.sha256()
        if episode != len(lengths) or count:
            raise RuntimeError(
                f"packet count mismatch in {path}: completed {episode}/{len(lengths)}, remainder {count}"
            )
        return result
    finally:
        container.close()


def first_frame_timestamp(path: Path) -> float:
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        frame = next(container.decode(stream))
        return float(frame.pts * stream.time_base)
    finally:
        container.close()


def load_episode_metadata(root: Path) -> pa.Table:
    paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError("no episode metadata parquet files")
    table = pa.concat_tables([pq.read_table(path) for path in paths], promote_options="default")
    order = pc.sort_indices(table, sort_keys=[("episode_index", "ascending")])
    return pc.take(table, order)


def set_scalar(table: pa.Table, name: str, value) -> pa.Table:
    index = table.column_names.index(name)
    field = table.schema.field(name)
    return table.set_column(index, field, pa.array([value], type=field.type))


def split_video_group(source: Path, lengths: list[int], destination_dir: Path) -> list[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    if len(lengths) == 1:
        output = destination_dir / "segment-000.mp4"
        shutil.copy2(source, output)
        return [output]

    boundaries = np.cumsum(lengths[:-1]).tolist()
    pattern = destination_dir / "segment-%03d.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-map", "0:v:0", "-c:v", "copy", "-f", "segment",
        "-segment_frames", ",".join(map(str, boundaries)),
        "-reset_timestamps", "1", str(pattern),
    ]
    subprocess.run(command, check=True)
    outputs = sorted(destination_dir.glob("segment-*.mp4"))
    if len(outputs) != len(lengths):
        raise RuntimeError(f"{source}: ffmpeg produced {len(outputs)} segments, expected {len(lengths)}")
    return outputs


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    building = output.with_name(f"{output.name}.building")
    if output.exists() or building.exists():
        raise FileExistsError(f"output or build directory already exists: {output}, {building}")

    info = json.loads((source / "meta" / "info.json").read_text())
    total_episodes = int(info["total_episodes"])
    total_frames = int(info["total_frames"])
    if total_episodes <= 0:
        raise ValueError("source dataset has no episodes")
    meta = load_episode_metadata(source)
    if meta.num_rows != total_episodes:
        raise RuntimeError(f"metadata rows {meta.num_rows} != episodes {total_episodes}")
    rows = meta.to_pylist()
    if [row["episode_index"] for row in rows] != list(range(total_episodes)):
        raise RuntimeError("episode indices are not contiguous")
    if sum(int(row["length"]) for row in rows) != total_frames:
        raise RuntimeError("episode lengths do not sum to total_frames")

    try:
        (building / "data" / "chunk-000").mkdir(parents=True)
        (building / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

        # Copy immutable global metadata first. File-size limits remain small so
        # future appends continue the one-file-per-episode organization.
        new_info = dict(info)
        new_info["data_files_size_in_mb"] = 1
        new_info["video_files_size_in_mb"] = 1
        (building / "meta" / "info.json").write_text(
            json.dumps(new_info, indent=2, ensure_ascii=False)
        )
        for name in ("tasks.parquet", "stats.json"):
            shutil.copy2(source / "meta" / name, building / "meta" / name)
        for name in ("RECOVERY_REPORT.json", "FINAL_VALIDATION.json"):
            if (source / name).is_file():
                shutil.copy2(source / name, building / name)

        # Video migration and lossless packet verification.
        video_validation: dict[str, dict] = {}
        for camera in CAMERAS:
            video_key = f"observation.images.{camera}"
            metadata_prefix = f"videos/{video_key}"
            destination = building / "videos" / video_key / "chunk-000"
            destination.mkdir(parents=True)
            groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
            for row in rows:
                groups[
                    (
                        row[f"{metadata_prefix}/chunk_index"],
                        row[f"{metadata_prefix}/file_index"],
                    )
                ].append(row)

            migrated = 0
            for (chunk_index, file_index), group in sorted(groups.items()):
                group.sort(key=lambda row: row["episode_index"])
                source_video = (
                    source / "videos" / video_key / f"chunk-{chunk_index:03d}"
                    / f"file-{file_index:03d}.mp4"
                )
                lengths = [int(row["length"]) for row in group]
                original_hashes = packet_hashes(source_video, lengths)
                temp_dir = building / ".segments" / camera / f"{chunk_index:03d}-{file_index:03d}"
                segments = split_video_group(source_video, lengths, temp_dir)
                for row, length, expected_hash, segment in zip(
                    group, lengths, original_hashes, segments, strict=True
                ):
                    episode_index = int(row["episode_index"])
                    actual_frames = ffprobe_frames(segment)
                    if actual_frames != length:
                        raise RuntimeError(
                            f"{camera} ep{episode_index}: {actual_frames} frames != {length}"
                        )
                    actual_hash = packet_hashes(segment, [length])[0]
                    if actual_hash != expected_hash:
                        raise RuntimeError(f"{camera} ep{episode_index}: H.264 packet hash mismatch")
                    first_pts = first_frame_timestamp(segment)
                    if abs(first_pts) > 1e-6:
                        raise RuntimeError(f"{camera} ep{episode_index}: first PTS is {first_pts}, expected 0")
                    final_path = destination / f"file-{episode_index:03d}.mp4"
                    shutil.move(segment, final_path)
                    migrated += 1
            shutil.rmtree(building / ".segments" / camera)
            video_validation[camera] = {
                "files": migrated,
                "frames": total_frames,
                "packet_hash_match": True,
                "reencoded": False,
            }
            print(f"VIDEO {camera}: {migrated} files / {total_frames} frames / packet hashes PASS")
        segments_root = building / ".segments"
        if segments_root.exists():
            shutil.rmtree(segments_root)

        # One data parquet and one metadata parquet per episode.
        for row_index, row in enumerate(rows):
            episode_index = int(row["episode_index"])
            data_source = (
                source / "data" / f"chunk-{int(row['data/chunk_index']):03d}"
                / f"file-{int(row['data/file_index']):03d}.parquet"
            )
            episode_data = pq.read_table(
                data_source,
                filters=[("episode_index", "=", episode_index)],
            )
            if episode_data.num_rows != int(row["length"]):
                raise RuntimeError(
                    f"ep{episode_index}: data rows {episode_data.num_rows} != {row['length']}"
                )
            pq.write_table(
                episode_data,
                building / "data" / "chunk-000" / f"file-{episode_index:03d}.parquet",
                compression="snappy",
                use_dictionary=True,
            )

            episode_meta = meta.slice(row_index, 1)
            for name, value in (
                ("data/chunk_index", 0),
                ("data/file_index", episode_index),
                ("meta/episodes/chunk_index", 0),
                ("meta/episodes/file_index", episode_index),
            ):
                episode_meta = set_scalar(episode_meta, name, value)
            for camera in CAMERAS:
                prefix = f"videos/observation.images.{camera}"
                for name, value in (
                    (f"{prefix}/chunk_index", 0),
                    (f"{prefix}/file_index", episode_index),
                    (f"{prefix}/from_timestamp", 0.0),
                    (f"{prefix}/to_timestamp", int(row["length"]) / int(info["fps"])),
                ):
                    episode_meta = set_scalar(episode_meta, name, value)
            pq.write_table(
                episode_meta,
                building / "meta" / "episodes" / "chunk-000" / f"file-{episode_index:03d}.parquet",
                compression="snappy",
                use_dictionary=True,
            )
            if (episode_index + 1) % 10 == 0 or episode_index + 1 == total_episodes:
                print(f"PARQUET {episode_index + 1}/{total_episodes}")

        migration_report = {
            "status": "complete",
            "layout": "one data parquet, one meta parquet, and three MP4 files per episode",
            "source": str(source),
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "video_validation": video_validation,
            "source_modified": False,
        }
        (building / "MIGRATION_REPORT.json").write_text(
            json.dumps(migration_report, indent=2, ensure_ascii=False)
        )
        os.rename(building, output)
        print(f"MIGRATION COMPLETE: {output}")
    except Exception:
        print(f"MIGRATION FAILED; partial build preserved at: {building}")
        raise


if __name__ == "__main__":
    main()
