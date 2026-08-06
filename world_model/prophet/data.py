import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio.v3 as iio
import numpy as np


@dataclass(frozen=True)
class DatasetSummary:
    root: Path
    clips: int
    metadata: int
    action_frames: int
    manifest_entries: int
    manifest_coverage: int
    unlisted_clips: int


def write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.asarray(frames, dtype=np.uint8), fps=fps, codec="libx264", pixelformat="yuv420p")


def write_metadata(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def write_manifest(root: Path, entries: list[dict[str, str]]) -> None:
    (root / "dataset.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")


def video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def validate_processed_dataset(root: str | Path, sample_limit: int = 32) -> DatasetSummary:
    dataset_root = Path(root).expanduser().resolve()
    clips = sorted((dataset_root / "clips").glob("*.mp4"))
    metadata = sorted((dataset_root / "metadata").glob("*.npz"))
    action_frames = sorted((dataset_root / "blacks").glob("*_black.mp4")) if (dataset_root / "blacks").is_dir() else []
    clip_stems = {path.stem for path in clips}
    metadata_stems = {path.stem for path in metadata}
    if clip_stems != metadata_stems:
        missing_video = sorted(metadata_stems - clip_stems)[:10]
        missing_metadata = sorted(clip_stems - metadata_stems)[:10]
        raise ValueError(f"Unpaired samples: missing_video={missing_video}, missing_metadata={missing_metadata}")

    manifest_path = dataset_root / "dataset.json"
    entries = json.loads(manifest_path.read_text()) if manifest_path.is_file() else []
    manifest_stems = set()
    for entry in entries:
        stem = Path(entry["media_path"]).stem
        manifest_stems.add(stem)
        if stem not in clip_stems:
            raise ValueError(f"Manifest references a missing clip: {entry['media_path']}")

    for metadata_path in metadata[:sample_limit]:
        clip_path = dataset_root / "clips" / f"{metadata_path.stem}.mp4"
        with np.load(metadata_path) as annotation:
            expected_frames = int(annotation["clip_len"]) if "clip_len" in annotation else None
        actual_frames = video_frame_count(clip_path)
        if expected_frames is not None and actual_frames != expected_frames:
            raise ValueError(f"Frame mismatch for {metadata_path.stem}: video={actual_frames}, metadata={expected_frames}")

    manifest_coverage = len(manifest_stems & clip_stems)
    return DatasetSummary(
        dataset_root,
        len(clips),
        len(metadata),
        len(action_frames),
        len(entries),
        manifest_coverage,
        len(clip_stems - manifest_stems),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a processed Prophet dataset")
    parser.add_argument("root", type=Path)
    parser.add_argument("--sample-limit", type=int, default=32)
    args = parser.parse_args()
    summary = validate_processed_dataset(args.root, args.sample_limit)
    print(json.dumps({
        "root": str(summary.root),
        "clips": summary.clips,
        "metadata": summary.metadata,
        "action_frames": summary.action_frames,
        "manifest_entries": summary.manifest_entries,
        "manifest_coverage": summary.manifest_coverage,
        "unlisted_clips": summary.unlisted_clips,
    }, indent=2))


if __name__ == "__main__":
    main()
