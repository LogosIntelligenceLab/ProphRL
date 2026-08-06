#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path

from prophet.checkpoints import resolve_checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a model-only checkpoint bundle for Prophet fine-tuning")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = resolve_checkpoint_path(args.checkpoint)
    model_dir = args.output / "checkpoints" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / source.name
    if destination.exists() or destination.is_symlink():
        if not args.force:
            raise FileExistsError(f"Destination already exists: {destination}")
        destination.unlink()
    if args.mode == "copy":
        shutil.copy2(source, destination)
    elif args.mode == "hardlink":
        os.link(source, destination)
    else:
        destination.symlink_to(source)
    (args.output / "checkpoints" / "latest_checkpoint.txt").write_text(f"{destination.name}\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
