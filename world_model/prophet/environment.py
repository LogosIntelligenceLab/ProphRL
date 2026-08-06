#!/usr/bin/env python3

import argparse
import importlib.util
import platform

import torch

REQUIRED_MODULES = (
    "apex",
    "attrs",
    "cv2",
    "decord",
    "flash_attn",
    "hydra",
    "mediapy",
    "megatron",
    "scipy",
    "transformer_engine",
    "wandb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the Prophet runtime environment")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Visible GPUs: {torch.cuda.device_count()}")
    if missing:
        raise RuntimeError(f"Missing required modules: {', '.join(missing)}")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but no GPU is visible")
    print("Environment check passed")


if __name__ == "__main__":
    main()
