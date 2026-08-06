import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from prophet.data import write_manifest, write_metadata, write_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LIBERO HDF5 demonstrations to Prophet clips")
    parser.add_argument("--input", type=Path, required=True, help="HDF5 file or directory containing LIBERO demos")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def axis_angle_to_xyzw(axis_angles: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(axis_angles).as_quat().astype(np.float32)


def pad_single_arm(values: np.ndarray, trailing_shape: tuple[int, ...]) -> np.ndarray:
    padded = np.zeros((len(values), 2, *trailing_shape), dtype=values.dtype)
    padded[:, 0] = values
    return padded


def task_caption(path: Path) -> str:
    name = path.stem.split("_demo", 1)[0]
    return name.replace("_", " ")


def iter_hdf5_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.hdf5"))


def source_identifier(input_path: Path, hdf5_path: Path) -> str:
    source = hdf5_path.name if input_path.is_file() else hdf5_path.relative_to(input_path).as_posix()
    return hashlib.sha1(source.encode()).hexdigest()[:10]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, str]] = []
    sample_count = 0

    for hdf5_path in iter_hdf5_files(args.input):
        source_id = source_identifier(args.input, hdf5_path)
        with h5py.File(hdf5_path, "r") as handle:
            for demo_name in sorted(handle["data"], key=lambda name: int(name.rsplit("_", 1)[-1])):
                demo = handle["data"][demo_name]
                frames = np.asarray(demo["obs/agentview_rgb"])
                positions = np.asarray(demo["obs/ee_pos"], dtype=np.float32)
                orientations = axis_angle_to_xyzw(np.asarray(demo["obs/ee_ori"], dtype=np.float32))
                grippers = np.asarray(demo["obs/gripper_states"], dtype=np.float32)
                actions = np.asarray(demo["actions"], dtype=np.float32)
                if len(frames) < args.clip_length:
                    continue

                for start in range(0, len(frames) - args.clip_length + 1, args.stride):
                    stem = f"agentview_{source_id}_{demo_name}_clip_{start // args.stride:05d}"
                    clip_path = args.output / "clips" / f"{stem}.mp4"
                    metadata_path = args.output / "metadata" / f"{stem}.npz"
                    if not args.overwrite and clip_path.exists() and metadata_path.exists():
                        entries.append({"caption": task_caption(hdf5_path), "media_path": f"clips/{clip_path.name}"})
                        continue

                    stop = start + args.clip_length
                    clip_frames = frames[start:stop, :, ::-1]
                    clip_positions = pad_single_arm(positions[start:stop], (3,))
                    clip_orientations = pad_single_arm(orientations[start:stop], (4,))
                    clip_orientations[:, 1, 3] = 1
                    clip_grippers = pad_single_arm(grippers[start:stop], (2,))
                    clip_actions = pad_single_arm(actions[start:stop], (7,))
                    write_video(clip_path, clip_frames, args.fps)
                    write_metadata(
                        metadata_path,
                        end_position=clip_positions,
                        end_orientation=clip_orientations,
                        effector_position=clip_grippers,
                        actions=clip_actions,
                        start_frame=np.asarray(start),
                        clip_len=np.asarray(args.clip_length),
                        fps=np.asarray(args.fps),
                    )
                    entries.append({"caption": task_caption(hdf5_path), "media_path": f"clips/{clip_path.name}"})
                    sample_count += 1
                    if args.limit is not None and sample_count >= args.limit:
                        write_manifest(args.output, entries)
                        return

    write_manifest(args.output, entries)


if __name__ == "__main__":
    main()
