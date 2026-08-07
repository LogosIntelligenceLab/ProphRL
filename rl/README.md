# ProphRL Reinforcement Learning

**Reinforcing Action Policies by Prophesying**

ProphRL is an online reinforcement learning framework for vision-language-action (VLA) policies. It uses the action-conditioned **Prophet** world model as an interactive visual simulator: the VLA predicts action chunks, Prophet generates the subsequent robot video, a frozen VLM reward model judges task completion, and **Flow-action-GRPO (FA-GRPO)** with **FlowScale (FS)** updates the flow-based action policy.

This directory contains the reinforcement learning component of ProphRL, including the Pi05, VLA-Adapter, and OpenVLA-OFT-Flow policy backends, the Cosmos/Prophet rollout interface, VLM reward evaluation, FSDP training, and checkpoint merging. Prophet pretraining, fine-tuning, and standalone inference code live in the sibling [`../world_model`](../world_model/) directory. By default, training loads its source through `VAE_FOLDER=../world_model` and its DiT checkpoint through `WORLD_MODEL_PATH=../world_model/checkpoints/prophet/prophet-bridge.pt`.

See the paper [Reinforcing Action Policies by Prophesying](https://arxiv.org/abs/2511.20633) and the [Logos Robotics Group / ProphRL project page](https://LogosRoboticsGroup.github.io/ProphRL).

Unless noted otherwise, run all commands below from the repository's `rl/` directory. This document uses `PROPHRL_ROOT` for the repository root and `RL_ROOT` and `WORLD_MODEL_ROOT` for the two component directories.

## Method Overview

A single training update follows this data flow:

~~~text
initial observation + language instruction
        |
        v
VLA policy -- action chunk --> Prophet world model -- future video --> VLM reward
    ^                                                                  |
    +---------------- FA-GRPO + FlowScale policy update ----------------+
~~~

- **Prophet** is a history-aware, action-conditioned video world model that generates closed-loop future trajectories from the current observation and multi-step actions.
- **FA-GRPO** first aggregates action log-probabilities over internal flow steps and then constructs the PPO ratio at the environment-action level, aligning optimization with the action chunks that are actually executed.
- **FlowScale (FS)** reweights advantages using the noise scale at each flow step. It suppresses excessively large score gradients at low-noise steps without changing the policy's sampling distribution.
- **VLM reward** uniformly samples rollout frames and asks Qwen2.5-VL to classify the trajectory as `Success` or `Failure`. The default configuration samples five judgments per trajectory and assigns a positive reward when at least three votes indicate success.

## FlowScale Formulation

Let `std` be the noise standard deviation of the flow schedule at outer decision step `s` and internal flow step `k`:

$$
\sigma_{s,k}^{2}=\operatorname{std}_{s,k}^{2},
\qquad
\widetilde{w}_{s,k}=(\sigma_{s,k}^{2}+\varepsilon)^p.
$$

FlowScale applies the **normalize-mix-clip** weighting from Eq. (22) of the paper:

$$
w_{s,k}=\operatorname{clip}\!\left(
\alpha+(1-\alpha)
\frac{\widetilde{w}_{s,k}}
{\frac{1}{K}\sum_{j=1}^{K}\widetilde{w}_{s,j}},
w_{\min},w_{\max}
\right),
\qquad
w_{s,k}=\operatorname{stopgrad}(w_{s,k}).
$$

The default configuration uses `p=0.5`, `alpha=0.3`, `w_min=0.3`, and `w_max=2.0`. With `p=0.5`, the unnormalized weight is approximately proportional to the noise standard deviation. Before clipping, normalization gives the weights a mean of 1 over the `K` flow steps, uniform mixing prevents excessive concentration, and clipping bounds the effective scale. The same environment-action advantage is broadcast to every flow step and multiplied by detached `w`:

$$
\widehat{A}^{\mathrm{FS}}_{s,k,c}=w_{s,k}\widehat{A}_{s,c}.
$$

### Source of `std`

The `std` value comes from the SDE sampling process in the VLA action head; it is not an uncertainty estimate predicted by another network. Each policy call uses `K=10` flow steps with `\Delta t=-1/K` and the following fixed noise schedule:

~~~text
[1.0000, 0.9601, 0.9133, 0.8577, 0.7904,
 0.7073, 0.6022, 0.4649, 0.2780, 0.0089, 0.0000]
~~~

At flow time `t`, the implementation selects `sigma = sigmas[K * (1 - t)]`. The SDE noise scale and the transition standard deviation are:

$$
\gamma_{s,k}=0.7\sqrt{\frac{\sigma_{s,k}}{1-\sigma'_{s,k}}},
\qquad
\operatorname{std}_{s,k}=\max\!\left(\sqrt{|\Delta t|}\,\gamma_{s,k},10^{-6}\right).
$$

For the first flow step, where `sigma=1`, the denominator uses `sigma_max=sigmas[1]` to avoid division by zero. Other steps use `sigma'=sigma`. This `std` is used both for sampling

$$
x_{k+1}=\mu_k+\operatorname{std}_{s,k}\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I),
$$

and for recomputing the Gaussian log-probability. All three backends reshape it to `[B, S, K, 1, 1]`, share it across the action-chunk and action-dimension axes, and pass it from the actor worker to FlowScale.

The core computation is:

~~~python
index = (K * (1 - t)).long()
sigma = sigmas[index]
sigma_for_denominator = torch.where(sigma == 1, sigmas[1], sigma)
std_dev_t = torch.sqrt(sigma / (1 - sigma_for_denominator)) * 0.7
std = (math.sqrt(1.0 / K) * std_dev_t).clamp_min(1e-6)
std = std[..., None, None]  # [B, S, K, 1, 1]
~~~

The corresponding implementation is in [`verl_vla/workers/actor/dp_rob.py`](verl_vla/workers/actor/dp_rob.py):

~~~python
variance = std.float().squeeze(-1).squeeze(-1).square()
flow_weights = (variance + 1e-6).pow(self.config.fs_p)
flow_weights /= flow_weights.mean(dim=2, keepdim=True) + 1e-6
flow_weights = self.config.fs_alpha + (1.0 - self.config.fs_alpha) * flow_weights
flow_weights = flow_weights.clamp(self.config.fs_w_min, self.config.fs_w_max).detach()
flow_weights = flow_weights[..., None].expand(
    batch_size, trajectory_steps, flow_steps, action_chunks
)
advantages = advantages[:, :, None, :].expand_as(flow_weights)
advantages = advantages * flow_weights
~~~

FlowScale weights are broadcast to every flow step as stop-gradient coefficients that rescale the policy gradient. The unified `kl_fs` training branch computes the PPO ratio, trajectory mask, and reference-policy KL.

The `kl_fs` objective adds the log-probability KL against a frozen reference policy to the FlowScale-weighted policy loss:

$$
L=L_{\mathrm{FA\text{-}GRPO+FS}}+
\beta\,\mathbb{E}_{M}\!\left[\log\pi_\theta(a\mid o)-\log\pi_{\mathrm{ref}}(a\mid o)\right].
$$

## Repository Layout

~~~text
ProphRL/
├── rl/
│   ├── verl_vla/
│   │   ├── trainer/                # Ray/FSDP trainer, GRPO, and reward manager
│   │   ├── workers/                # actor, reference, rollout, and checkpoint workers
│   │   ├── environment/            # LIBERO environment integration
│   │   └── utils/
│   │       ├── vla_utils/          # Pi05, VLA-Adapter, and OpenVLA-OFT-Flow
│   │       └── wm_utils/           # Prophet/Cosmos world-model interface
│   ├── examples/exp/
│   │   ├── Pi05_exp/               # Bridge carrot/stack training and merging
│   │   ├── VLA-Adapter_exp/        # Bridge carrot training and merging
│   │   └── OpenVLAOFT_exp/         # Bridge carrot training and merging
│   ├── data/bridge/                 # Bridge manifest, normalization statistics, and initial frames
│   ├── .env.example                # local path-variable template
│   ├── align.json                  # Ray runtime environment
│   └── pyproject.toml              # ProphRL package and base dependencies
└── world_model/                    # Prophet source, training, inference, and checkpoints
~~~

The RL component retains selected upstream module names and package structures for compatibility with existing Pi05, VLA-Adapter, OpenVLA-OFT, and FSDP checkpoints. Prophet's internal `cosmos_predict2`, `imaginaire`, and Megatron-Core packages are provided by the sibling `../world_model` directory.

## Environment

### Training Environment

Prophet/Cosmos has the strictest dependency constraints. Create its CUDA 12.6 environment first, then install ProphRL. The world-model component pins these core versions:

| Component | Version |
| --- | --- |
| Python | 3.10 |
| PyTorch / TorchVision | 2.6.0 / 0.21.0 |
| CUDA wheels | 12.6 |
| flash-attn | 2.6.3 |
| Transformer Engine | 1.13 |
| Megatron-Core | 0.10.0 |
| Transformers | 4.51.3 |
| NumPy | 1.26.4 |

From the ProphRL repository root, enter `rl/`, create the environment from the pinned dependencies in the sibling `world_model/` directory, and then install the RL package:

~~~bash
cd rl
export RL_ROOT="$(pwd)"
export PROPHRL_ROOT="$(cd .. && pwd)"
export WORLD_MODEL_ROOT="${PROPHRL_ROOT}/world_model"

conda create -n prophrl python=3.10 -y
conda activate prophrl
python -m pip install --upgrade pip uv

cd "${WORLD_MODEL_ROOT}"
uv sync --extra cu126 --active --inexact
python -m pip install -e . --no-deps

cd "${RL_ROOT}"
python -m pip install -e ".[train,pi05,dev]"
~~~

The `train` extra provides the shared FSDP, rollout, VLM client, TensorFlow image-processing, VLA-Adapter, and OpenVLA-OFT-Flow runtime dependencies. The `pi05` extra adds the JAX/Flax, LeRobot, OpenPI client, Orbax, and tokenizer dependencies required by the vendored OpenPI implementation. The default Pi05 Bridge recipe requires both extras. The `pi05` extra can be omitted when running only VLA-Adapter or OpenVLA-OFT-Flow.

`uv sync` uses the CUDA 12.6 package index defined by `world_model/`. If NVIDIA Apex is required in your environment, build it against the active environment after PyTorch is installed:

~~~bash
git clone "https://github.com/NVIDIA/apex.git" "/tmp/apex"
python -m pip install -v --no-build-isolation "/tmp/apex"
~~~

Use a separate environment for the vLLM reward server to prevent its Torch, Ray, and Transformers dependencies from affecting the training environment. The training process accesses the reward server through an OpenAI-compatible HTTP API.

### Installation Verification

Activate the training environment, set the world-model path, and run:

~~~bash
export RL_ROOT="$(pwd)"
export PROPHRL_ROOT="$(cd .. && pwd)"
export WORLD_MODEL_ROOT="${PROPHRL_ROOT}/world_model"
export VAE_FOLDER="${WORLD_MODEL_ROOT}"
export PYTHONPATH="${VAE_FOLDER}${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'PY'
import torch
import cosmos_predict2
import imaginaire
from megatron.core import parallel_state

assert torch.cuda.is_available()
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("gpus:", torch.cuda.device_count(), torch.cuda.get_device_name(0))
print("cosmos:", cosmos_predict2.__file__)
print("imaginaire:", imaginaire.__file__)
PY
~~~

For Pi05, verify the actual OpenPI policy/configuration import chain instead of importing only an empty package entry point:

~~~bash
python - <<'PY'
import flax
import jax
import lerobot
import openpi_client
import orbax.checkpoint
import tensorflow as tf
from verl_vla.utils.vla_utils.pi05 import openpi as _openpi
from openpi.policies import policy_config
from openpi.training import config as openpi_config
from verl_vla.workers.actor.dp_rob import RobDataParallelPPOActor
from verl_vla.workers.rollout.rob_rollout import RobHFRollout

assert openpi_config.get_config("pi05_bridge").name == "pi05_bridge"
print("jax:", jax.__version__, "flax:", flax.__version__, "tensorflow:", tf.__version__)
print("pi05-entry-import-ok", policy_config.__file__)
PY
~~~

If `openpi_client` comes from a local checkout instead of a wheel, add its `src` directory to `PYTHONPATH`. Every Ray node must see the same Python environment and world-model source. The training scripts add `VAE_FOLDER` to `PYTHONPATH`.

### Cosmos Tokenizer

The Cosmos Video2World tokenizer is downloaded automatically from NVIDIA's Hugging Face repository and cached locally; it does not need to be placed under `VAE_FOLDER`:

~~~bash
HF_ENDPOINT=https://huggingface.co hf auth login
~~~

Before the first run, accept the NVIDIA Open Model License and access terms for `nvidia/Cosmos-Predict2-2B-Video2World` on Hugging Face. The downloader connects directly to the official endpoint by default and ignores third-party `HF_ENDPOINT` mirrors configured in the shell. The default settings are:

~~~text
repo_id:  nvidia/Cosmos-Predict2-2B-Video2World
filename: tokenizer/tokenizer.pth
revision: f50c09f5d8ab133a90cac3f4886a6471e9ba3f18
endpoint: https://huggingface.co
~~~

Hugging Face manages the download location. Set `HF_HOME` to change the cache root. For offline execution, set `actor_rollout_ref.world_model.tokenizer_path` to an existing `tokenizer.pth`, or set `actor_rollout_ref.world_model.tokenizer_local_files_only=true` after preparing the cache.

## Training Assets

Pi05 Bridge reinforcement learning requires three independent assets. The following table lists both their Hugging Face locations and their expected paths after download:

| Asset | Purpose | Hugging Face repository | Remote path | Local path | Status |
| --- | --- | --- | --- | --- | --- |
| Pi05 Bridge SFT checkpoint | initial actor and reference-policy weights | [`Fleurrr/OpenPI05-Bridge-RL`](https://huggingface.co/Fleurrr/OpenPI05-Bridge-RL) | `model.safetensors`, `metadata.pt` | `checkpoints/policy/pi05-bridge/` | Published |
| Prophet Bridge checkpoint | action-conditioned world-model rollout | [`Fleurrr/Prophet-World-Model`](https://huggingface.co/Fleurrr/Prophet-World-Model/tree/main) | `prophet-bridge.pt` | `../world_model/checkpoints/prophet/prophet-bridge.pt` | Published |
| Bridge RL data | initial observations and task instructions | [`Fleurrr/Bridge-RL-Data`](https://huggingface.co/datasets/Fleurrr/Bridge-RL-Data) | `dataset.jsonl`, `images/**` | `data/bridge/` | Published |

Notes:

- The Pi05 Bridge SFT checkpoint contains Pi05 weights already supervised-fine-tuned on Bridge. It is not an actor checkpoint produced by ProphRL training.
- The Prophet Bridge checkpoint is the public model-only checkpoint for the Bridge world model and is loaded through `WORLD_MODEL_PATH`.
- The Bridge RL dataset provides the processed manifest and initial-observation images. Its `dataset.jsonl` uses sharded paths of the form `images/<first two hash characters>/<filename>`, which must resolve relative to `data/bridge/`. The Pi05 `norm_stats.json` file is included in `rl/data/bridge/` and is not supplied by this Hugging Face dataset.

Download all training assets from `rl/`:

~~~bash
export RL_ROOT="$(pwd)"
export PROPHRL_ROOT="$(cd .. && pwd)"
export WORLD_MODEL_ROOT="${PROPHRL_ROOT}/world_model"

hf download "Fleurrr/OpenPI05-Bridge-RL" \
    "model.safetensors" "metadata.pt" \
    --local-dir "${RL_ROOT}/checkpoints/policy/pi05-bridge"

hf download "Fleurrr/Prophet-World-Model" "prophet-bridge.pt" \
    --local-dir "${WORLD_MODEL_ROOT}/checkpoints/prophet"

hf download "Fleurrr/Bridge-RL-Data" \
    --repo-type dataset \
    --local-dir "${RL_ROOT}/data/bridge"
~~~

## Prophet Weights

ProphRL Bridge training uses `prophet-bridge.pt` from [Fleurrr/Prophet-World-Model](https://huggingface.co/Fleurrr/Prophet-World-Model/tree/main). This is the public model-only Bridge checkpoint. The training scripts load it from `../world_model/checkpoints/prophet/prophet-bridge.pt` by default.

Download it from `rl/`:

~~~bash
export RL_ROOT="$(pwd)"
export PROPHRL_ROOT="$(cd .. && pwd)"
export WORLD_MODEL_ROOT="${PROPHRL_ROOT}/world_model"
mkdir -p "${WORLD_MODEL_ROOT}/checkpoints/prophet"
hf download "Fleurrr/Prophet-World-Model" "prophet-bridge.pt" \
    --local-dir "${WORLD_MODEL_ROOT}/checkpoints/prophet"

sha256sum "${WORLD_MODEL_ROOT}/checkpoints/prophet/prophet-bridge.pt"
~~~

The expected file size is `4,115,563,141` bytes, and its SHA-256 digest is:

~~~text
acbc0b0806bfbc1f90724152f16b492b54aea5f62be629ea8601a8ba6582434e
~~~

The repository also publishes `prophet-pretrained.pt` and `prophet-libero.pt`. Use `prophet-bridge.pt` for the current Bridge training entry point. The Cosmos tokenizer is still downloaded from NVIDIA as described above and cannot be replaced by a Prophet DiT checkpoint.

## Model, Data, and Path Configuration

For the default Pi05 Bridge recipe, the final directory layout should be:

~~~text
ProphRL/
├── rl/
│   ├── checkpoints/policy/pi05-bridge/
│   │   ├── model.safetensors
│   │   └── metadata.pt
│   └── data/bridge/
│       ├── dataset.jsonl
│       ├── norm_stats.json
│       └── images/
│           ├── 00/
│           ├── 01/
│           └── ...
└── world_model/
    ├── cosmos_predict2/
    ├── imaginaire/
    └── checkpoints/prophet/prophet-bridge.pt
~~~

Each path is used as follows:

| Asset | Path relative to the ProphRL root | Training parameter |
| --- | --- | --- |
| Pi05 Bridge SFT checkpoint | `rl/checkpoints/policy/pi05-bridge` | `SFT_MODEL_PATH` |
| Prophet Bridge checkpoint | `world_model/checkpoints/prophet/prophet-bridge.pt` | `WORLD_MODEL_PATH` |
| Bridge rollout data | `rl/data/bridge` | `BRIDGE_DATA_DIR` |
| Prophet world-model source | `world_model` | `VAE_FOLDER` |

Copy and load the environment template:

~~~bash
cp ".env.example" ".env"
# Edit .env, then load it into the current shell.
set -a
source ".env"
set +a
~~~

Paths in the template are relative to `rl/`:

~~~dotenv
SFT_MODEL_PATH=checkpoints/policy/pi05-bridge
BRIDGE_DATA_DIR=data/bridge
WORLD_MODEL_PATH=../world_model/checkpoints/prophet/prophet-bridge.pt
VAE_FOLDER=../world_model
BRIDGE_LEROBOT_REPO_ID=
~~~

The training scripts resolve these paths from `rl/`. When `BRIDGE_LEROBOT_REPO_ID` is empty, Pi05 reads `norm_stats.json` from `BRIDGE_DATA_DIR`. If a custom `BRIDGE_DATA_DIR` contains only the rollout manifest and images, set `BRIDGE_LEROBOT_REPO_ID=data/bridge` to reuse the repository's bundled normalization statistics. In this code path, the value identifies a dataset/assets location that resolves `norm_stats.json`; it does not require a remote Hugging Face lookup. For VLA-Adapter or OpenVLA-OFT-Flow, point `SFT_MODEL_PATH` to the checkpoint directory for the selected backend.

`.env.example` also defines paths used only by optional upstream workflows: `PRISMATIC_DATA_ROOT` for Prismatic pretraining data, `DROID_RLDS_DATA_DIR` for complete DROID RLDS fine-tuning, `PRIME_MODEL_PATH` and `PRIME_REF_PATH` for the PRIME reward model, and `GENERATION_*` for generic generation jobs. The default Bridge RL recipe does not read these values. Configure them only when enabling the corresponding workflow. The codebase does not provide machine-specific absolute paths as defaults.

Main variables:

| Variable | Meaning |
| --- | --- |
| `SFT_MODEL_PATH` | initial checkpoint for Pi05, VLA-Adapter, or OpenVLA-OFT-Flow |
| `BRIDGE_DATA_DIR` | root of the processed Bridge world-model rollout data |
| `BRIDGE_LEROBOT_REPO_ID` | optional dataset/assets location containing Pi05 normalization statistics; defaults to `BRIDGE_DATA_DIR` |
| `PALIGEMMA_TOKENIZER_PATH` | optional local PaliGemma tokenizer file |
| `WORLD_MODEL_PATH` | Prophet/Cosmos DiT checkpoint (`.pt`) |
| `VAE_FOLDER` | Prophet world-model source directory, added to `PYTHONPATH` by the training scripts |
| `PI05_BASE_TORCH_PATH` | optional Pi05 base PyTorch weights |
| `OUTPUT_DIR` | root for logs, rollouts, and checkpoints; defaults to `work_dirs` |
| `PRISMATIC_DATA_ROOT` | optional Prismatic pretraining-data directory; defaults to `data/prismatic-vlms` |
| `DROID_RLDS_DATA_DIR` | optional complete DROID RLDS dataset directory |
| `PRIME_MODEL_PATH` | optional PRIME reward checkpoint or Hugging Face model ID |
| `PRIME_REF_PATH` | checkpoint or model ID for the frozen reference used with `PRIME_MODEL_PATH` |
| `GENERATION_INPUT_PATH` | optional input file for generic generation jobs |
| `GENERATION_OUTPUT_PATH` | optional generic-generation output file; defaults to `outputs/generation.parquet` |
| `GENERATION_MODEL_PATH` | optional generic-generation model directory or Hugging Face model ID |
| `REFERENCE_VIDEO_DIR` | optional successful-example video directory for reference-aware reward |

For Pi05 Bridge, `SFT_MODEL_PATH` must point to a training checkpoint containing `model.safetensors` and `metadata.pt`. If the checkpoint does not contain normalization statistics, the code falls back to `BRIDGE_LEROBOT_REPO_ID`, which defaults to the bundled `data/bridge/norm_stats.json`.

Store Bridge data under:

~~~text
data/bridge/
├── dataset.jsonl
├── norm_stats.json
└── images/                         # sharded by the first two filename-hash characters
    ├── 00/
    ├── 01/
    └── ...
~~~

The official `dataset.jsonl` stores `init_state_path` relative to `data/bridge/` and uses the same sharded layout as the Hugging Face dataset, for example `images/0a/PutSpoonOnTableClothInScene_0a0bf6fa.png`. The `hf download` command above preserves this layout. The repository includes the manifest and normalization statistics, but the images referenced by the manifest must still be downloaded before training.

Custom preprocessed data may instead use a flat `images/<filename>` layout because the loader resolves paths directly from the manifest. The manifest and image directory must come from the same dataset; do not mix flat and sharded versions. The following check reads the `BRIDGE_DATA_DIR` actually used for training:

~~~bash
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ.get("BRIDGE_DATA_DIR", "data/bridge")).expanduser()
rows = [json.loads(line) for line in (root / "dataset.jsonl").read_text().splitlines() if line]
missing = [path for row in rows for path in row["init_state_path"] if not (root / path).is_file()]
if missing:
    raise FileNotFoundError(f"Missing {len(missing)} Bridge images; first: {missing[0]}")
print(f"bridge-data-ok: {len(rows)} records from {root.resolve()}")
PY
~~~

If a custom `BRIDGE_DATA_DIR` does not contain `norm_stats.json`, point Pi05 normalization statistics to the repository's bundled data directory:

~~~bash
export BRIDGE_DATA_DIR="/path/to/bridge_for_rl"
export BRIDGE_LEROBOT_REPO_ID="${RL_ROOT}/data/bridge"
~~~

Do not commit models, datasets, videos, `.env`, or `WANDB_API_KEY` to Git. Public data manifests and checksum metadata are the exceptions.

## VLM Reward Service

The training scripts use `reward_model.type=vlm_serve` and connect to `http://localhost:18901/v1` with the served model name `judge`. Start the server in a separate reward environment before training:

~~~bash
vllm serve "Qwen/Qwen2.5-VL-72B-Instruct" \
    --port 18901 \
    --gpu-memory-utilization 0.35 \
    --cpu-offload-gb 4 \
    --max-model-len 8192 \
    --tensor-parallel-size 8 \
    --served-model-name judge \
    --enforce-eager \
    --disable-log-requests
~~~

Verify it from another terminal:

~~~bash
curl http://localhost:18901/v1/models
~~~

The Bridge reward prompt requires the final response `\box{Success}` or `\box{Failure}`. Each trajectory uses 30 frames with `vote_n=5` and `vote_m=3`. Unparseable responses count as failures. The training reward comes entirely from this VLM judgment: `return_env_score=False` prevents ground-truth environment success from being used as the optimization signal.

On an 8 x 48 GiB GPU configuration, the reward service uses approximately 17.3 GiB per GPU. For other GPUs or longer inputs, adjust `gpu-memory-utilization` according to KV-cache headroom and peak training usage.

## Training

### Training Scripts

| Backend | Scenario | Script |
| --- | --- | --- |
| Pi05 | Bridge carrot | `examples/exp/Pi05_exp/train/exp1-pi05_wm_fs_bridge_carrot.sh` |
| Pi05 | Bridge stack | `examples/exp/Pi05_exp/train/exp1-pi05_wm_fs_bridge_stack.sh` |
| VLA-Adapter | Bridge carrot | `examples/exp/VLA-Adapter_exp/train/wm/exp1-vla_adapter_wm_fs_bridge_carrot.sh` |
| OpenVLA-OFT-Flow | Bridge carrot | `examples/exp/OpenVLAOFT_exp/train/exp1-openvla_oft_wm_fs_bridge_carrot.sh` |

First perform a configuration-only check without loading models or allocating GPUs:

~~~bash
export RL_ROOT="$(pwd)"
export PROPHRL_ROOT="$(cd .. && pwd)"
export WORLD_MODEL_ROOT="${PROPHRL_ROOT}/world_model"
export SFT_MODEL_PATH="${RL_ROOT}/checkpoints/policy/pi05-bridge"
export WORLD_MODEL_PATH="${WORLD_MODEL_ROOT}/checkpoints/prophet/prophet-bridge.pt"
export BRIDGE_DATA_DIR="${RL_ROOT}/data/bridge"
export VAE_FOLDER="${WORLD_MODEL_ROOT}"

DRY_RUN=1 bash "examples/exp/Pi05_exp/train/exp1-pi05_wm_fs_bridge_carrot.sh"
~~~

To validate the entry point with existing custom preprocessed data, override `BRIDGE_DATA_DIR` as described above before running `DRY_RUN`. The script accepts absolute paths and passes them unchanged to `data.libero_raw_data_dir`.

The output should contain the expected backend, scene, checkpoint, world-model path, and `kl_loss_type=kl_fs`. The four `fs_*` values come from `verl_vla/trainer/config/ppo_trainer.yaml` by default and can also be appended as Hydra overrides. After validation, start training:

~~~bash
NUM_GPUS=8 NUM_NODES=1 \
bash "examples/exp/Pi05_exp/train/exp1-pi05_wm_fs_bridge_carrot.sh"
~~~

Append Hydra overrides directly to the command. For example, disable W&B and change checkpoint frequency:

~~~bash
bash examples/exp/Pi05_exp/train/exp1-pi05_wm_fs_bridge_carrot.sh \
    trainer.wandb_mode=disabled \
    trainer.save_freq=20
~~~

Use `PROJECT_NAME`, `EXPERIMENT_NAME`, `OUTPUT_DIR`, `CKPT_PATH`, `DATASET_NAME`, `NUM_GPUS`, `NUM_NODES`, and `RUNTIME_ENV_PATH` to override runtime settings. The default recipe performs full-parameter FSDP updates and enables history-conditioned Prophet rollout, 10-step world-model sampling, GRPO, FlowScale, and KL regularization.

## World-Model Rollout and Video

Training does not require a separate rollout CLI. At each outer step, `RobHFRollout` invokes `CosmosWorldModel`: the policy produces an action chunk, the world model generates the next video segment, the final frame becomes the next policy observation, and earlier frames enter Prophet's history buffer.

The external interface always accepts **one real RGB image** with current-observation shape `[B, H, W, 3]`; the dataset does not need to provide a second camera view. The Bridge checkpoint retains a two-view internal structure, so the adapter automatically constructs a black dummy view and zero actions for that view. Only the generated frames from the real view are returned. Enabling `use_history=true` does not change the caller's input format because the rollout wrapper maintains previous generated frames in memory.

The `step_images` sent to the VLM reward model are this closed-loop world-model trajectory. Every 10 global steps, including step 0, the reward manager automatically saves a batch rollout grid to:

~~~text
${CKPT_PATH}/train_rollouts/<global_step>_rand<id>.mp4
~~~

Videos use 10 FPS, with one batch sample per grid row. They are both the direct reward input and a way to inspect Prophet's action consistency, temporal continuity, and visual quality. `CosmosWorldModel.save_video_grid(...)` can also write a rollout of shape `[B, F, H, W, C]` explicitly to MP4.

## Merging Checkpoints

Training produces sharded FSDP checkpoints. The first argument below is the actor-checkpoint directory, and the second is the merged `.pt` file:

~~~bash
bash examples/exp/Pi05_exp/merge_ckpts/merge_bridge.sh \
    work_dirs/pi05/actor/global_step_9 \
    work_dirs/merged/pi05-step9.pt

bash examples/exp/VLA-Adapter_exp/merge_ckpts/merge_bridge.sh \
    work_dirs/vla-adapter/actor/global_step_9 \
    work_dirs/merged/vla-adapter-step9.pt

bash examples/exp/OpenVLAOFT_exp/merge_ckpts/merge_bridge.sh \
    work_dirs/openvla-oft/actor/global_step_9 \
    work_dirs/merged/openvla-oft-step9.pt
~~~

Set `DRY_RUN=1` to inspect the arguments first. `SFT_MODEL_PATH` must match the training backend. Merging requires the same backend dependencies and an FSDP-capable GPU environment.

## License

Original code is released under the [MIT License](LICENSE). `verl_vla/` contains derived code from projects including veRL, OpenPI, OpenVLA-OFT, VLA-Adapter, and vLLM; their respective licenses and copyright notices are retained. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES](LICENSES/). Prophet/Cosmos models and weights are also subject to their upstream NVIDIA licenses.
