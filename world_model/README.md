# Prophet World Model

This directory is the standalone world-model component of ProphRL. It contains the single-view Prophet model, full-parameter BRIDGE and LIBERO fine-tuning, checkpoint validation, and rollout generation.

The package intentionally retains the internal `cosmos_predict2` and `imaginaire` module names because the released checkpoints were trained with those state-dict and configuration paths. New release utilities live in the smaller `prophet` package.

## Contents

```text
world_model/
├── prophet/                 # Clean action, checkpoint, and rollout utilities
├── examples/                # Ready-to-run inference demo
├── cosmos_predict2/         # Minimal inherited Video2World training/runtime core
├── imaginaire/              # Minimal trainer and distributed utilities
├── scripts/
│   ├── finetune_bridge.sh
│   ├── finetune_libero.sh
│   ├── create_checkpoint_bundle.py
│   └── download_base_models.py
├── docs/                    # Data preparation and format
└── release/                 # Model card, weight license, and checksums
```

## Environment

The code was validated with:

- Python 3.10.18
- PyTorch 2.6.0 + CUDA 12.6
- flash-attn 2.6.3
- Transformer Engine 1.13.0
- Megatron-Core 0.10.0

### Use an existing Cosmos environment

If a compatible Cosmos-Predict2 environment already exists:

```bash
conda activate cosmos-predict2
cd world_model
python -m pip uninstall -y cosmos-predict2
python -m pip install -e . --no-deps
prophet-env-check
```

Add `--require-cuda` on a GPU node:

```bash
prophet-env-check --require-cuda
```

### Create a new environment

```bash
conda create -n prophrl-world-model python=3.10 -y
conda activate prophrl-world-model
python -m pip install --upgrade pip uv

cd world_model
uv sync --extra cu126 --active --inexact
git clone https://github.com/NVIDIA/apex.git /tmp/apex
python -m pip install -v --no-build-isolation /tmp/apex
python -m pip install -e . --no-deps
prophet-env-check --require-cuda
```

The pinned Python dependencies and CUDA 12.6 package index are defined in `pyproject.toml` and `uv.lock`. NVIDIA Apex is installed separately because it is built against the active PyTorch/CUDA environment.

## NVIDIA Base Assets

Prophet uses the 2B action-conditioned Cosmos initialization and the Cosmos Video2World tokenizer. Log in to Hugging Face, accept the required NVIDIA and guardrail model terms, then run:

```bash
python scripts/download_base_models.py
```

The expected files include:

```text
checkpoints/nvidia/
├── Cosmos-Predict2-2B-Sample-Action-Conditioned/model-480p-4fps.pt
└── Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth
```

## Prophet Checkpoints

The public model repository is [`Fleurrr/Prophet-World-Model`](https://huggingface.co/Fleurrr/Prophet-World-Model) on Hugging Face. Download individual artifacts with:

```bash
hf download Fleurrr/Prophet-World-Model prophet-pretrained.pt --local-dir checkpoints/prophet
hf download Fleurrr/Prophet-World-Model prophet-bridge.pt --local-dir checkpoints/prophet
hf download Fleurrr/Prophet-World-Model prophet-libero.pt --local-dir checkpoints/prophet
```

The release contains three model-only artifacts:

| Artifact | Training stage |
| --- | --- |
| [`prophet-pretrained.pt`](https://huggingface.co/Fleurrr/Prophet-World-Model/resolve/main/prophet-pretrained.pt?download=true) | pretrained Prophet initialization |
| [`prophet-bridge.pt`](https://huggingface.co/Fleurrr/Prophet-World-Model/resolve/main/prophet-bridge.pt?download=true) | full-parameter BRIDGE adaptation |
| [`prophet-libero.pt`](https://huggingface.co/Fleurrr/Prophet-World-Model/resolve/main/prophet-libero.pt?download=true) | full-parameter LIBERO adaptation |

The pretrained artifact is the initialization used by the released BRIDGE and LIBERO variants. Exact file sizes and SHA-256 checksums are in `release/weights.json`.

Validate one or more downloaded checkpoints without allocating the 2B model:

```bash
prophet-checkpoint \
  /path/to/prophet-pretrained.pt \
  /path/to/prophet-bridge.pt \
  /path/to/prophet-libero.pt \
  --manifest release/weights.json
```

Add `--verify-sha256` for a full integrity check. The validator accepts either a bare model file or a run-style directory, deserializes the state dict, verifies Prophet-specific tensor shapes, and checks artifact sizes against the release manifest.

## Data Layout

BRIDGE and LIBERO fine-tuning expect processed roots with matching video and annotation stems:

```text
dataset_root/
├── clips/example.mp4
├── metadata/example.npz
├── blacks/example_black.mp4   # optional projected action frames
├── dataset.json               # BRIDGE task metadata when used
└── task_info.json             # optional cached task mapping
```

See `docs/DATA.md` for action conventions and annotation keys.

The release includes `prophet-prepare-bridge`, `prophet-prepare-libero`, and `prophet-validate-data` for converting official BRIDGE RLDS and LIBERO HDF5 demonstrations into this layout.

## Fine-tuning

Both public downstream recipes use **full-parameter FSDP fine-tuning** (`train_architecture="base"`), not PEFT/LoRA adaptation. The defaults are 30,000 steps and batch size 24 per process. Some inherited Cosmos configuration fields contain `lora` in their names, but those fields do not change the public recipes into LoRA training.

The trainer expects a run-style directory rather than a bare model file. Build it once from the pretrained artifact:

```bash
python scripts/create_checkpoint_bundle.py \
  --checkpoint /path/to/prophet-pretrained.pt \
  --output checkpoints/prophet-pretrained

export PROPHRL_PRETRAIN_CKPT=$PWD/checkpoints/prophet-pretrained
```

Use `--mode symlink` for a local zero-copy bundle or `--mode hardlink` when the source and destination share a filesystem.

### BRIDGE

```bash
export PROPHRL_BRIDGE_DATA=/path/to/processed/bridge
GPUS=8 ./scripts/finetune_bridge.sh
```

### LIBERO

```bash
export PROPHRL_LIBERO_DATA=/path/to/processed/libero
GPUS=8 ./scripts/finetune_libero.sh
```

For multi-node training, additionally set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`. Hydra overrides may be appended to either command.

Training outputs are written under `checkpoints/prophrl_world_model/` by default. Set `IMAGINAIRE_OUTPUT_ROOT=/path/to/output` to place run directories elsewhere.

To verify initialization and one optimizer update without writing a multi-gigabyte checkpoint:

```bash
GPUS=2 ./scripts/finetune_libero.sh \
  trainer.max_iter=1 \
  trainer.run_validation=false \
  trainer.logging_iter=1 \
  checkpoint.save_iter=0 \
  dataloader_train.batch_size=1 \
  dataloader_train.num_workers=0
```

The same overrides work with `finetune_bridge.sh`. `checkpoint.save_iter=0` disables checkpoint writing and is intended for smoke tests only.

## World-Model Rollout

### Automatic-download demo

`prophet-demo` downloads the selected checkpoint from `Fleurrr/Prophet-World-Model` and then invokes the standalone rollout pipeline. NVIDIA base assets must already exist under the directory passed to `--base-checkpoints`.

LIBERO example:

```bash
prophet-demo \
  --variant libero \
  --input /path/to/libero/clips/example.mp4 \
  --actions /path/to/libero/metadata/example.npz \
  --base-checkpoints checkpoints \
  --output outputs/libero_demo.mp4
```

BRIDGE example:

```bash
prophet-demo \
  --variant bridge \
  --input /path/to/bridge/clips/example.mp4 \
  --actions /path/to/bridge/metadata/example.npz \
  --base-checkpoints checkpoints \
  --output outputs/bridge_demo.mp4
```

The equivalent shell wrapper is:

```bash
./examples/inference_demo.sh libero /path/to/input.mp4 /path/to/actions.npz checkpoints
```

Add `--preflight` to `prophet-demo` to validate the checkpoint and inputs without allocating the model. Use `--checkpoint /path/to/local.pt` to skip the Hugging Face download.

### Direct rollout CLI

`prophet-rollout` accepts either an image or video as the initial observation and a JSON/NPZ action annotation. History conditioning is enabled by default.

If the NVIDIA base assets are stored outside `world_model/checkpoints`, point the inherited Cosmos runtime to their parent directory:

```bash
export COSMOS_PREDICT2_ARGS="--checkpoints /path/to/checkpoints"
```

For a processed LIBERO sample, use the servo-action conversion and its paired projected action frames:

```bash
prophet-rollout \
  --checkpoint /path/to/prophet-libero.pt \
  --input /path/to/conditioning.mp4 \
  --actions /path/to/actions.npz \
  --action-format libero-servo \
  --action-frames \
  --output outputs/libero_rollout.mp4 \
  --autoregressive
```

On a CPU-only login node, validate all rollout inputs without allocating the model:

```bash
prophet-rollout \
  --checkpoint /path/to/prophet-libero.pt \
  --input /path/to/conditioning.mp4 \
  --actions /path/to/actions.npz \
  --preflight
```

Use `--no-history` for ablation and `--action-frames` when the paired projected action-frame video exists under `blacks/`. Use `--num-sampling-steps` to override the default 35 denoising steps. Prophet checkpoints are fixed to a 20-step action chunk, corresponding to a 280-dimensional dual-arm conditioning vector.

`--checkpoint` may also point to a run-style directory containing `checkpoints/latest_checkpoint.txt` and `checkpoints/model/<name>.pt`.

Supported action conversions are:

- `--action-format libero-servo`: reproduces LIBERO fine-tuning preprocessing, including the `0.05` motion scale and gripper remapping;
- `--action-format pose-delta`: converts `end_position`, `end_orientation`, and `effector_position` into local delta actions;
- `--action-format precomputed`: consumes `actions` or `action` as already normalized 7D/14D actions;
- `--action-format auto`: selects LIBERO servo conversion for `[T, 2, 7]` `actions` arrays and otherwise uses precomputed or pose-delta annotations.

Single-arm 7D actions are zero-padded to the shared dual-arm 14D representation.

## Release Boundary and License

The source code is Apache-2.0. Prophet weights are derivative NVIDIA Cosmos models governed by `release/NVIDIA_OPEN_MODEL_LICENSE.md`. Built on NVIDIA Cosmos.
