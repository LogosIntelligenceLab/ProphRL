#!/usr/bin/env bash
set -euo pipefail

LOAD_MODEL_PATH="${1:-${LOAD_MODEL_PATH:-}}"
MERGED_OUT_PATH="${2:-${MERGED_OUT_PATH:-}}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

: "${SFT_MODEL_PATH:?Set SFT_MODEL_PATH to the OpenVLA-OFT-Flow base checkpoint}"
: "${LOAD_MODEL_PATH:?Pass the FSDP checkpoint directory as argument 1 or set LOAD_MODEL_PATH}"
: "${MERGED_OUT_PATH:?Pass the merged output file as argument 2 or set MERGED_OUT_PATH}"
BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR:-${REPO_ROOT}/data/bridge}"
[[ "${SFT_MODEL_PATH}" = /* ]] || SFT_MODEL_PATH="${REPO_ROOT}/${SFT_MODEL_PATH}"
[[ "${BRIDGE_DATA_DIR}" = /* ]] || BRIDGE_DATA_DIR="${REPO_ROOT}/${BRIDGE_DATA_DIR}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export ACCELERATE_USE_FSDP="${ACCELERATE_USE_FSDP:-true}"
export FSDP_CPU_RAM_EFFICIENT_LOADING="${FSDP_CPU_RAM_EFFICIENT_LOADING:-true}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export BRIDGE_LEROBOT_REPO_ID="${BRIDGE_LEROBOT_REPO_ID:-${BRIDGE_DATA_DIR}}"

PROJECT_NAME="${PROJECT_NAME:-SimpleVLA-RL}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-merge-openvla-oft-flow-bridge-checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-work_dirs}"
CKPT_PATH="${CKPT_PATH:-${OUTPUT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
DATASET_NAME="${DATASET_NAME:-bridge_orig}"
DATASET_PATH="${BRIDGE_DATA_DIR}"
VLA_NAME="openvla-oft-flow"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_NODES="${NUM_NODES:-1}"
ALIGN_PATH="${RUNTIME_ENV_PATH:-${REPO_ROOT}/align.json}"
META_PATH="${META_PATH:-}"

run_merge() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '%s\n' "$@"
        return
    fi
    HYDRA_FULL_ERROR=1 "$@" 2>&1 | tee -a "${EXPERIMENT_NAME}.log"
}

run_merge python -m verl_vla.trainer.main_ppo \
    "data.task_suite_name=${DATASET_NAME}" \
    "data.libero_raw_data_dir=${DATASET_PATH}" \
    data.num_trials_per_task=50 \
    data.n_samples=8 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.1 \
    data.accuracy_upper_bound=0.9 \
    data.oversample_factor=1 \
    data.train_batch_size=32 \
    data.val_batch_size=496 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    "actor_rollout_ref.model.path=${SFT_MODEL_PATH}" \
    "actor_rollout_ref.model.load_ckpt=${LOAD_MODEL_PATH}" \
    actor_rollout_ref.model.merge_save_ckpt_only=True \
    "actor_rollout_ref.model.merged_out=${MERGED_OUT_PATH}" \
    "actor_rollout_ref.model.meta_path=${META_PATH}" \
    "actor_rollout_ref.model.vla=${VLA_NAME}" \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=20 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.dlogp_clamp=True \
    actor_rollout_ref.actor.dlogp_clamp_max=4.0 \
    actor_rollout_ref.actor.dlogp_clamp_min=-4.0 \
    actor_rollout_ref.actor.kl_loss_type=kl_fs \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    algorithm.kl_ctrl.kl_coef=0.04 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    "actor_rollout_ref.actor.unnorm_key=${DATASET_NAME}" \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.val_micro_batch_size=8 \
    actor_rollout_ref.rollout.temperature=1.6 \
    "actor_rollout_ref.rollout.experiment_name=${EXPERIMENT_NAME}" \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    "actor_rollout_ref.rollout.unnorm_key=${DATASET_NAME}" \
    actor_rollout_ref.rollout.model_family=openvla \
    "actor_rollout_ref.rollout.task_suite_name=${DATASET_NAME}" \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    "actor_rollout_ref.rollout.pretrained_checkpoint=${SFT_MODEL_PATH}" \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    "actor_rollout_ref.rollout.n_gpus_per_node=${NUM_GPUS}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    "actor_rollout_ref.ref.vla=${VLA_NAME}" \
    "actor_rollout_ref.ref.unnorm_key=${DATASET_NAME}" \
    "trainer.logger=[console,tensorboard]" \
    "trainer.project_name=${PROJECT_NAME}" \
    "trainer.experiment_name=${EXPERIMENT_NAME}" \
    "trainer.default_local_dir=${CKPT_PATH}" \
    "trainer.n_gpus_per_node=${NUM_GPUS}" \
    "trainer.nnodes=${NUM_NODES}" \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=100 \
    trainer.val_only=True \
    ray_init.num_cpus=32 \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    algorithm.whiten=False \
    "trainer.runtime_env=${ALIGN_PATH}" \
    trainer.wandb_mode=online \
    trainer.val_before_train=True
