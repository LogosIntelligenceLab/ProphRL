#!/usr/bin/env python3

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPOSITORIES = {
    "nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned": {
        "revision": "1d2b0a920e76ca913f216427abe5b97633806143",
        "allow_patterns": ["model-480p-4fps.pt"],
    },
    "nvidia/Cosmos-Predict2-2B-Video2World": {
        "revision": "f50c09f5d8ab133a90cac3f4886a6471e9ba3f18",
        "allow_patterns": ["tokenizer/*"],
    },
    "nvidia/Cosmos-Guardrail1": {
        "revision": "d6d4bfa899a71454a700907664f3e88f503950cf",
    },
    "meta-llama/Llama-Guard-3-8B": {
        "revision": "7327bd9f6efbbe6101dc6cc4736302b3cbb6e425",
        "ignore_patterns": ["original/*"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the NVIDIA base assets required by Prophet")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for repo_id, options in REPOSITORIES.items():
        local_dir = args.checkpoint_dir / repo_id
        print(f"{repo_id} -> {local_dir}")
        if args.dry_run:
            continue
        snapshot_download(repo_id=repo_id, local_dir=local_dir, **options)


if __name__ == "__main__":
    main()
