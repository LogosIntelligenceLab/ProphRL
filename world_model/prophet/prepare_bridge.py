import argparse
import hashlib
from pathlib import Path

import numpy as np

from prophet.data import write_manifest, write_metadata, write_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BridgeData V2 RLDS episodes to Prophet clips")
    parser.add_argument("--input", type=Path, required=True, help="TFDS builder directory, usually bridge_orig/1.0.0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train+val")
    parser.add_argument("--image-key", default="image_0")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--minimum-frames", type=int, default=21)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def decode_text(value: object) -> str:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def pad_single_arm(values: np.ndarray) -> np.ndarray:
    padded = np.zeros((len(values), 2, 7), dtype=np.float32)
    padded[:, 0] = values
    return padded


def episode_identifier(episode: object) -> str:
    metadata = episode["episode_metadata"]
    episode_id = int(metadata["episode_id"].numpy())
    file_path = decode_text(metadata["file_path"])
    return hashlib.sha1(f"{file_path}:{episode_id}".encode()).hexdigest()[:10]


def main() -> None:
    args = parse_args()
    try:
        import tensorflow_datasets as tfds
    except ImportError as error:
        raise RuntimeError("Bridge conversion requires tensorflow-datasets; install the bridge-data extra") from error

    builder = tfds.builder_from_directory(builder_dir=str(args.input))
    dataset = builder.as_dataset(split=args.split, shuffle_files=False)
    args.output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, str]] = []

    for episode in dataset:
        steps = list(tfds.as_numpy(episode["steps"]))
        if len(steps) < args.minimum_frames:
            continue
        frames = np.stack([step["observation"][args.image_key] for step in steps])
        actions = pad_single_arm(np.stack([step["action"] for step in steps]).astype(np.float32))
        states = pad_single_arm(np.stack([step["observation"]["state"] for step in steps]).astype(np.float32))
        caption = decode_text(steps[0]["language_instruction"])
        stem = f"bridge_orig_ep{episode_identifier(episode)}_image_primary_clip_00000"
        clip_path = args.output / "clips" / f"{stem}.mp4"
        metadata_path = args.output / "metadata" / f"{stem}.npz"
        if args.overwrite or not (clip_path.exists() and metadata_path.exists()):
            write_video(clip_path, frames, args.fps)
            write_metadata(
                metadata_path,
                action=actions,
                obs_state=states,
                start_frame=np.asarray(0),
                clip_len=np.asarray(len(frames)),
                fps=np.asarray(args.fps),
            )
        entries.append({"caption": caption, "media_path": f"clips/{clip_path.name}"})
        if args.limit is not None and len(entries) >= args.limit:
            break

    write_manifest(args.output, entries)


if __name__ == "__main__":
    main()
