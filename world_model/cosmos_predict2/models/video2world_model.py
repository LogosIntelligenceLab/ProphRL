# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import math
import random
from collections.abc import Mapping
from typing import Any

from torchvision.io import write_video
import os

import attrs
import torch
from einops import rearrange
from megatron.core import parallel_state
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_predict2.conditioner import DataType, TextCondition
from cosmos_predict2.configs.base.config_video2world import (
    Video2WorldPipelineConfig,
    get_cosmos_predict2_video2world_pipeline,
)
from cosmos_predict2.networks.model_weights_stats import WeightTrainingStat
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.utils.checkpointer import non_strict_load_model
from cosmos_predict2.utils.optim_instantiate import get_base_scheduler
from cosmos_predict2.utils.torch_future import clip_grad_norm_
from cosmos_predict2.utils.context_parallel import cat_outputs_cp, split_inputs_cp
from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
from imaginaire.lazy_config import LazyDict, instantiate
from imaginaire.model import ImaginaireModel
from imaginaire.utils import log, misc

def save_videos_tv(videos, out_dir, fps=10, prefix="vid", codec="h264"):
    os.makedirs(out_dir, exist_ok=True)
    assert videos.ndim == 5 and videos.shape[1] == 3, "shape 必须是 (B, 3, T, H, W)"
    B, C, T, H, W = videos.shape

    for b in range(B):
        vid = videos[b].permute(1, 2, 3, 0).contiguous().to(torch.uint8).cpu()
        path = os.path.join(out_dir, f"{prefix}_{b:03d}.mp4")
        write_video(path, vid, fps=fps, video_codec=codec, options={"crf": "18", "preset": "veryfast"})
        print("saved:", path)

def _normalize_key(k: str,
                   strip_prefixes=("module.",),
                   remove_substrings=("_checkpoint_wrapped_module.",)) -> str:
    for p in strip_prefixes:
        if k.startswith(p):
            k = k[len(p):]
    for s in remove_substrings:
        k = k.replace(s, "")
    return k

def filter_state_dict_by_shape(model: torch.nn.Module,
                               state_dict: dict,
                               *,
                               strip_prefixes=("module.",),
                               remove_substrings=("_checkpoint_wrapped_module.",)):
    sd = state_dict
    for top in ("state_dict", "model", "module"):
        if isinstance(sd, dict) and top in sd and isinstance(sd[top], dict):
            sd = sd[top]

    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())

    filtered = OrderedDict()
    skipped_mismatch = []
    skipped_missing = []

    for raw_k, v in sd.items():
        k = _normalize_key(raw_k, strip_prefixes, remove_substrings)
        if k not in model_keys:
            skipped_missing.append(raw_k)
            continue
        tgt = model_sd[k]
        if v.shape != tgt.shape:
            skipped_mismatch.append((k, tuple(v.shape), tuple(tgt.shape)))
            continue
        if v.dtype != tgt.dtype:
            v = v.to(dtype=tgt.dtype)
        if v.device != tgt.device:
            v = v.to(device=tgt.device)
        filtered[k] = v

    return filtered, skipped_mismatch, skipped_missing

@attrs.define(slots=False)
class Predict2ModelManagerConfig:
    # Local path, use it in fast debug run
    dit_path: str = get_cosmos_predict2_video2world_checkpoint(model_size="2B")
    # For inference
    text_encoder_path: str = ""  # not used in training.


@attrs.define(slots=False)
class Predict2Video2WorldModelConfig:
    train_architecture: str = "base"
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_target_modules: str = "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"
    init_lora_weights: bool = True

    precision: str = "bfloat16"

    def __attrs_post_init__(self):
        """Validate LoRA configuration after initialization."""
        if self.train_architecture == "lora":
            if self.lora_rank <= 0:
                raise ValueError(f"LoRA rank must be positive, got {self.lora_rank}")
            if self.lora_alpha <= 0:
                raise ValueError(f"LoRA alpha must be positive, got {self.lora_alpha}")
            if not self.lora_target_modules.strip():
                raise ValueError("LoRA target_modules cannot be empty")

            # Warn about potentially inefficient configurations
            if self.lora_rank > 64:
                log.warning(f"High LoRA rank ({self.lora_rank}) may reduce training efficiency")
            if self.lora_alpha != self.lora_rank:
                log.info(
                    f"LoRA alpha ({self.lora_alpha}) != rank ({self.lora_rank}), scaling factor: {self.lora_alpha / self.lora_rank}"
                )

    input_video_key: str = "video"
    input_image_key: str = "images"
    loss_reduce: str = "mean"
    loss_scale: float = 10.0

    adjust_video_noise: bool = True

    # This is used for the original way to load models
    model_manager_config: Predict2ModelManagerConfig = Predict2ModelManagerConfig()  # noqa: RUF009
    # This is a new way to load models
    pipe_config: Video2WorldPipelineConfig = get_cosmos_predict2_video2world_pipeline(model_size="2B")  # noqa: RUF009
    # debug flag
    debug_without_randomness: bool = False
    fsdp_shard_size: int = 0  # 0 means not using fsdp, -1 means set to world size
    # High sigma strategy
    high_sigma_ratio: float = 0.0


class Predict2Video2WorldModel(ImaginaireModel):
    def __init__(self, config: Predict2Video2WorldModelConfig):
        super().__init__()

        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        self.device = torch.device("cuda")

        # 1. set data keys and data information
        self.setup_data_key()

        # 4. Set up loss options, including loss masking, loss reduce and loss scaling
        self.loss_reduce = getattr(config, "loss_reduce", "mean")
        assert self.loss_reduce in ["mean", "sum"]
        self.loss_scale = getattr(config, "loss_scale", 1.0)
        log.critical(f"Using {self.loss_reduce} loss reduce with loss scale {self.loss_scale}")
        if self.config.adjust_video_noise:
            self.video_noise_multiplier = math.sqrt(self.config.pipe_config.state_t)
        else:
            self.video_noise_multiplier = 1.0

        # 7. training states
        if parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

        # New way to init pipe
        self.pipe = Video2WorldPipeline.from_config(
            config.pipe_config,
            dit_path=config.model_manager_config.dit_path,
        )

        self.freeze_parameters()
        if config.train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.dit,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_target_modules=config.lora_target_modules,
                init_lora_weights=config.init_lora_weights,
            )
            if self.pipe.dit_ema:
                self.add_lora_to_model(
                    self.pipe.dit_ema,
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_target_modules=config.lora_target_modules,
                    init_lora_weights=config.init_lora_weights,
                )
            # Enhanced LoRA logging
            self._log_lora_statistics()
        else:
            self.pipe.denoising_model().requires_grad_(True)
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Print the number in billions, or in the format of 1,000,000,000
        log.info(
            f"Total parameters: {total_params / 1e9:.2f}B, Frozen parameters: {frozen_params:,}, Trainable parameters: {trainable_params:,}"
        )

        if config.fsdp_shard_size != 0 and torch.distributed.is_initialized():
            if config.fsdp_shard_size == -1:
                fsdp_shard_size = torch.distributed.get_world_size()
                replica_group_size = 1
            else:
                fsdp_shard_size = min(config.fsdp_shard_size, torch.distributed.get_world_size())
                replica_group_size = torch.distributed.get_world_size() // fsdp_shard_size
            dp_mesh = init_device_mesh(
                "cuda", (replica_group_size, fsdp_shard_size), mesh_dim_names=("replicate", "shard")
            )
            log.info(f"Using FSDP with shard size {fsdp_shard_size} | device mesh: {dp_mesh}")
            self.pipe.apply_fsdp(dp_mesh)
        else:
            log.info("FSDP (Fully Sharded Data Parallel) is disabled.")

    # New function, added for i4 adaption
    @property
    def net(self) -> torch.nn.Module:
        return self.pipe.dit

    # New function, added for i4 adaption
    @property
    def net_ema(self) -> torch.nn.Module:
        return self.pipe.dit_ema

    # New function, added for i4 adaption
    def init_optimizer_scheduler(
        self,
        optimizer_config: LazyDict[torch.optim.Optimizer],
        scheduler_config: LazyDict[torch.optim.lr_scheduler.LRScheduler],
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Creates the optimizer and scheduler for the model.

        Args:
            config_model (ModelConfig): The config object for the model.

        Returns:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
        """
        optimizer = instantiate(optimizer_config, model=self.net)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        return optimizer, scheduler

    # ------------------------ training hooks ------------------------
    def on_before_zero_grad(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        """
        update the net_ema
        """
        del scheduler, optimizer

        if self.config.pipe_config.ema.enabled:
            # calculate beta for EMA update
            ema_beta = self.ema_beta(iteration)
            self.pipe.dit_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)

    # New function, added for i4 adaption
    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if self.config.pipe_config.ema.enabled:
            self.net_ema.to(dtype=torch.float32)
        for module in [self.net, self.pipe.tokenizer]:
            if module is not None:
                module.to(memory_format=memory_format, **self.tensor_kwargs)

    def freeze_parameters(self) -> None:
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def add_lora_to_model(
        self,
        model: torch.nn.Module,
        lora_rank: int = 4,
        lora_alpha: int = 4,
        lora_target_modules: str = "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        init_lora_weights: bool = True,
    ) -> None:
        """Add LoRA (Low-Rank Adaptation) adapters to the model.

        This function injects LoRA adapters into specified modules of the model,
        enabling parameter-efficient fine-tuning by training only a small number
        of additional parameters.

        Args:
            model: The PyTorch model to add LoRA adapters to
            lora_rank: The rank of the LoRA adaptation matrices. Higher rank allows
                      more expressiveness but uses more parameters (default: 4)
            lora_alpha: Scaling parameter for LoRA. Controls the magnitude of the
                       LoRA adaptation (default: 4)
            lora_target_modules: Comma-separated string of module names to target
                               for LoRA adaptation (default: attention and MLP layers)
            init_lora_weights: Whether to initialize LoRA weights properly (default: True)

        Raises:
            ImportError: If PEFT library is not installed
            ValueError: If invalid parameters are provided
            RuntimeError: If LoRA injection fails
        """
        try:
            from peft import LoraConfig, inject_adapter_in_model
        except ImportError as e:
            raise ImportError(
                "PEFT library is required for LoRA training. Please install it with: pip install peft"
            ) from e

        # Validate parameters
        if lora_rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {lora_rank}")
        if lora_alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {lora_alpha}")

        target_modules_list = [module.strip() for module in lora_target_modules.split(",")]
        if not target_modules_list:
            raise ValueError("LoRA target_modules cannot be empty")

        # Validate target modules exist in model
        model_module_names = set(name for name, _ in model.named_modules())
        invalid_modules = []
        for target_module in target_modules_list:
            # Check if any module contains this target pattern
            if not any(target_module in module_name for module_name in model_module_names):
                invalid_modules.append(target_module)

        if invalid_modules:
            log.warning(f"Target modules not found in model: {invalid_modules}")

        # Add LoRA to model
        self.lora_alpha = lora_alpha

        log.info(f"Adding LoRA adapters: rank={lora_rank}, alpha={lora_alpha}, targets={target_modules_list}")

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=target_modules_list,
        )

        try:
            model = inject_adapter_in_model(lora_config, model)
        except Exception as e:
            raise RuntimeError(f"Failed to inject LoRA adapters into model: {e}") from e

        # Count and log LoRA parameters
        lora_params = 0
        total_params = 0
        for name, param in model.named_parameters():  # noqa: B007
            total_params += param.numel()
            if param.requires_grad:
                lora_params += param.numel()
                # Upcast LoRA parameters into fp32
                param.data = param.to(torch.float32)

        log.info(
            f"LoRA injection successful: {lora_params:,} trainable parameters out of {total_params:,} total ({100 * lora_params / total_params:.3f}%)"
        )

    def _log_lora_statistics(self) -> None:
        """Log detailed LoRA parameter statistics."""
        lora_params_by_type = {}
        total_lora_params = 0

        for name, param in self.pipe.dit.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                param_type = "lora_A" if "lora_A" in name else "lora_B" if "lora_B" in name else "other_lora"
                if param_type not in lora_params_by_type:
                    lora_params_by_type[param_type] = 0
                lora_params_by_type[param_type] += param.numel()
                total_lora_params += param.numel()

        if total_lora_params > 0:
            log.info("LoRA parameter breakdown:")
            for param_type, count in lora_params_by_type.items():
                log.info(f"  {param_type}: {count:,} parameters")
            log.info(f"  Total LoRA: {total_lora_params:,} parameters")
        else:
            log.warning("No LoRA parameters found in model")

    def setup_data_key(self) -> None:
        self.input_video_key = self.config.input_video_key  # by default it is video key for Video diffusion model
        self.input_image_key = self.config.input_image_key

    def is_image_batch(self, data_batch: dict[str, torch.Tensor]) -> bool:
        """We hanlde two types of data_batch. One comes from a joint_dataloader where "dataset_name" can be used to differenciate image_batch and video_batch.
        Another comes from a dataloader which we by default assumes as video_data for video model training.
        """
        is_image = self.input_image_key in data_batch
        is_video = self.input_video_key in data_batch
        assert is_image != is_video, (
            "Only one of the input_image_key or input_video_key should be present in the data_batch."
        )
        return is_image

    def _update_train_stats(self, data_batch: dict[str, torch.Tensor]) -> None:
        is_image = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image else self.input_video_key
        if isinstance(self.pipe.dit, WeightTrainingStat):
            if is_image:
                self.pipe.dit.accum_image_sample_counter += data_batch[input_key].shape[0] * self.data_parallel_size
            else:
                self.pipe.dit.accum_video_sample_counter += data_batch[input_key].shape[0] * self.data_parallel_size

    def draw_training_sigma_and_epsilon(self, x0_size: torch.Size, condition: Any) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x0_size[0]
        epsilon = torch.randn(x0_size, device="cuda")
        sigma_B = self.pipe.scheduler.sample_sigma(batch_size).to(device="cuda")
        sigma_B_1 = rearrange(sigma_B, "b -> b 1")  # add a dimension for T, all frames share the same sigma
        is_video_batch = condition.data_type == DataType.VIDEO

        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B_1 = sigma_B_1 * multiplier
        if is_video_batch and self.config.high_sigma_ratio > 0:
            # Implement the high sigma strategy LOGUNIFORM200_100000
            LOG_200 = math.log(200)
            LOG_100000 = math.log(100000)
            mask = torch.rand(sigma_B_1.shape, device=sigma_B_1.device) < self.config.high_sigma_ratio
            log_new_sigma = (
                torch.rand(sigma_B_1.shape, device=sigma_B_1.device).type_as(sigma_B_1) * (LOG_100000 - LOG_200)
                + LOG_200
            )
            sigma_B_1 = torch.where(mask, log_new_sigma.exp(), sigma_B_1)
        return sigma_B_1, epsilon

    def get_per_sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sigma (tensor): noise level

        Returns:
            loss weights per sigma noise level
        """
        return (sigma**2 + self.pipe.sigma_data**2) / (sigma * self.pipe.sigma_data) ** 2

    def compute_loss_with_epsilon_and_sigma(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: TextCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute loss givee epsilon and sigma

        This method is responsible for computing loss give epsilon and sigma. It involves:
        1. Adding noise to the input data.
        2. Passing the noisy data through the network to generate predictions.
        3. Computing the loss based on the difference between the predictions and the original data, \
            considering any configured loss weighting.

        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            x0: image/video latent
            condition: text condition
            epsilon: noise
            sigma: noise level

        Returns:
            tuple: A tuple containing four elements:
                - dict: additional data that used to debug / logging / callbacks
                - Tensor 1: kendall loss,
                - Tensor 2: MSE loss,
                - Tensor 3: EDM loss

        Raises:
            AssertionError: If the class is conditional, \
                but no number of classes is specified in the network configuration.

        Notes:
            - The method handles different types of conditioning
            - The method also supports Kendall's loss
        """
        # Get the mean and stand deviation of the marginal probability distribution.
        mean_B_C_T_H_W, std_B_T = x0_B_C_T_H_W, sigma_B_T
        # Generate noisy observations
        xt_B_C_T_H_W = mean_B_C_T_H_W + epsilon_B_C_T_H_W * rearrange(std_B_T, "b t -> b 1 t 1 1")
        # make prediction
        model_pred = self.pipe.denoise(xt_B_C_T_H_W, sigma_B_T, condition)
        # loss weights for different noise levels
        weights_per_sigma_B_T = self.get_per_sigma_loss_weights(sigma=sigma_B_T)
        # extra loss mask for each sample, for example, human faces, hands
        pred_mse_B_C_T_H_W = (x0_B_C_T_H_W - model_pred.x0) ** 2
        edm_loss_B_C_T_H_W = pred_mse_B_C_T_H_W * rearrange(weights_per_sigma_B_T, "b t -> b 1 t 1 1")
        kendall_loss = edm_loss_B_C_T_H_W
        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "sigma": sigma_B_T,
            "weights_per_sigma": weights_per_sigma_B_T,
            "condition": condition,
            "model_pred": model_pred,
            "mse_loss": pred_mse_B_C_T_H_W.mean(),
            "edm_loss": edm_loss_B_C_T_H_W.mean(),
            "edm_loss_per_frame": torch.mean(edm_loss_B_C_T_H_W, dim=[1, 3, 4]),
        }
        output_batch["loss"] = kendall_loss.mean()  # check if this is what we want

        return output_batch, kendall_loss, pred_mse_B_C_T_H_W, edm_loss_B_C_T_H_W

    def _get_data_batch_input(
        self,
        video_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        blacks_batch: torch.Tensor | None,
        history_batch: torch.Tensor | None,
        prompt: str,
        negative_prompt: str = "",
        num_latent_conditional_frames: int = 1,
    ):
        """
        Prepares the input data batch for the diffusion model. (MODIFIED FOR BATCHING)
        ...
        """
        # MODIFIED: Get batch size dynamically from the input tensor
        batch_size, C, T, H, W = video_batch.shape

        data_batch = {
            "dataset_name": "video_data",
            "video": video_batch, # MODIFIED: Use the passed-in batch
            # MODIFIED: Create dummy embeddings with the correct batch size
            "t5_text_embeddings": torch.zeros(batch_size, 512, 1024, dtype=torch.bfloat16).cuda(),
            "fps": torch.randint(16, 32, (batch_size,)),
            "padding_mask": torch.zeros(batch_size, 1, H, W),
            "num_conditional_frames": num_latent_conditional_frames,
            "action": actions_batch, # MODIFIED: Use the passed-in action batch
        }
        if blacks_batch is not None:
            data_batch['blacks'] = blacks_batch # MODIFIED: Use the passed-in blacks batch
        if history_batch is not None:
            data_batch['history'] = history_batch

        # Handle negative prompts for classifier-free guidance
        if negative_prompt:
            # MODIFIED: Expand negative embeddings to match the batch size
            neg_embeds = self.encode_prompt(negative_prompt).to(dtype=self.torch_dtype)
            data_batch["neg_t5_text_embeddings"] = neg_embeds.repeat(batch_size, 1, 1)

        # Move tensors to GPU and convert to bfloat16 if they are floating point
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

        return data_batch

    @torch.no_grad()
    def get_generated_condition(self, data_batch: dict):
        # 1. Get data for generating condition
        history_video_length = data_batch['history_video_length'][0].item()
        num_frames = data_batch['num_frames'][0].item()
        forcing_history = data_batch['history'][:, :, :history_video_length, ...] # (B, 3, 60, H, W)
        # save_videos_tv(forcing_history, './output', prefix='forcing_history')
        if 'history_black' in data_batch.keys():
            forcing_blacks = data_batch['history_black'][:, :, history_video_length:, ...] # (B, 3, 21, H, W)
            # save_videos_tv(forcing_blacks, './output', prefix='forcing_blacks')
        else:
            forcing_blacks = None
        forcing_first_frame = data_batch['history'][:, :, history_video_length, ...].unsqueeze(2) # (B, 3, 1, H, W)
        forcing_action = data_batch['history_action'] # (B, 20, 14)

        # 2. Preprocessing data
        num_latent_conditional_frames = self.pipe.tokenizer.get_latent_num_frames(1)
        forcing_data_batch = self._get_data_batch_input(
            forcing_first_frame,
            forcing_action,
            forcing_blacks,
            forcing_history,
            "",
            "",
            num_latent_conditional_frames=num_latent_conditional_frames,
        )
        self.pipe._normalize_video_databatch_inplace(forcing_data_batch)
        self.pipe._augment_image_dim_inplace(forcing_data_batch)
        if 'blacks' in forcing_data_batch.keys():
            self.pipe._normalize_video_databatch_inplace(forcing_data_batch, input_key='blacks', check_key='is_black_preprocessed')
            if 'is_blacks_latent' not in forcing_data_batch.keys() or not forcing_data_batch['is_blacks_latent']:
                forcing_data_batch['blacks'] = self.pipe.encode(forcing_data_batch['blacks']).contiguous().float()
                forcing_data_batch['is_blacks_latent'] = True
        if 'history' in forcing_data_batch.keys():
            self.pipe._normalize_video_databatch_inplace(forcing_data_batch, input_key='history', check_key='is_history_preprocessed')
            if 'is_history_latent' not in forcing_data_batch.keys() or not forcing_data_batch['is_history_latent']:
                forcing_data_batch['history'] = self.pipe.encode(forcing_data_batch['history']).contiguous().float()
                forcing_data_batch['is_history_latent'] = True
        is_image_batch = self.pipe.is_image_batch(forcing_data_batch)
        input_key = self.pipe.input_image_key if is_image_batch else self.pipe.input_video_key

        # 3. Start generating
        n_sample = forcing_data_batch[input_key].shape[0]
        _T, _H, _W = forcing_data_batch[input_key].shape[-3:]
        state_shape = [
            self.pipe.config.state_ch,
            self.pipe.tokenizer.get_latent_num_frames(_T),
            _H // self.pipe.tokenizer.spatial_compression_factor,
            _W // self.pipe.tokenizer.spatial_compression_factor,
        ]
        x0_fn = self.pipe.get_x0_fn_from_batch(forcing_data_batch, 0, is_negative_prompt=True)
        log.info(f"Starting video generation for a batch of {n_sample}...")
        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),
                torch.float32,
                self.pipe.tensor_kwargs["device"],
                seed=0,
            )
            * self.pipe.scheduler.config.sigma_max
        )
        if self.pipe.dit.is_context_parallel_enabled:
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.pipe.get_context_parallel_group())

        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(35, device=x_sigma_max.device)
        sample = x_sigma_max.to(dtype=torch.float32)
        x0_prev: torch.Tensor | None = None
        for i, _ in enumerate(scheduler.timesteps):
            sigma_t = scheduler.sigmas[i].to(sample.device, dtype=torch.float32)
            sigma_in = sigma_t.repeat(sample.shape[0])
            x0_pred = x0_fn(sample, sigma_in)
            sample, x0_prev = scheduler.step(x0_pred=x0_pred, i=i, sample=sample, x0_prev=x0_prev)
        sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
        sigma_in = sigma_min.repeat(sample.shape[0])
        samples = x0_fn(sample, sigma_in)
        video_batch = self.pipe.decode(samples).detach() # (B, 3, 21, H, W)

        # 4. Updating raw data batch
        video_batch = video_batch.clamp(-1.0, 1.0)
        video_normalized = (video_batch + 1) / 2
        frames = (video_normalized * 255).clamp(0, 255).to(torch.uint8) # (B, 3, 21, H, W)
        # save_videos_tv(frames, './output', prefix='generated_frames')

        del data_batch['history_action']
        if 'history_black' in data_batch.keys():
            del data_batch['history_black']
        # data_batch['first_frame'] = frames[:, :, -1, ...]
        # save_videos_tv(data_batch['video'], './output', prefix='old_video')
        # data_batch['video'] = torch.cat([frames[:, :, -1:, ...], data_batch['video'][:, :, 1:, ...]], 2)
        # save_videos_tv(data_batch['video'], './output', prefix='new_video')
        data_batch['history'] = torch.cat([data_batch['history'][:, :, -history_video_length: -num_frames, ...], frames], 2)
        # save_videos_tv(data_batch['history'], './output', prefix='new_history')
        return data_batch

    def training_step(self, data_batch: dict, data_batch_idx: int) -> tuple[dict, torch.Tensor]:
        self.pipe.device = self.device

        # Self-forcing
        if self.use_self_forcing:
            data_batch = self.get_generated_condition(data_batch)

        # Loss
        self._update_train_stats(data_batch)

        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        _, x0_B_C_T_H_W, condition = self.pipe.get_data_and_condition(data_batch)

        # Sample pertubation noise levels and N(0, 1) noises
        sigma_B_T, epsilon_B_C_T_H_W = self.draw_training_sigma_and_epsilon(x0_B_C_T_H_W.size(), condition)

        # Broadcast and split the input data and condition for model parallelism
        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T = self.pipe.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
        )
        output_batch, kendall_loss, _, _ = self.compute_loss_with_epsilon_and_sigma(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
        )

        if self.loss_reduce == "mean":
            kendall_loss = kendall_loss.mean() * self.loss_scale
        elif self.loss_reduce == "sum":
            kendall_loss = kendall_loss.sum(dim=1).mean() * self.loss_scale
        else:
            raise ValueError(f"Invalid loss_reduce: {self.loss_reduce}")

        return output_batch, kendall_loss

    @torch.no_grad()
    def validation_step(self, data_batch: dict, data_batch_idx: int) -> tuple[dict, torch.Tensor]:
        return self.training_step(data_batch, data_batch_idx)

    # ------------------ Checkpointing ------------------

    def state_dict(self) -> dict[str, Any]:
        # the checkpoint format should be compatible with traditional imaginaire4
        # pipeline contains both net and net_ema
        # checkpoint should be saved/loaded from Model
        # checkpoint should be loadable from pipeline as well - We don't use Model for inference only jobs.

        net_state_dict = self.pipe.dit.state_dict(prefix="net.")
        if self.config.pipe_config.ema.enabled:
            ema_state_dict = self.pipe.dit_ema.state_dict(prefix="net_ema.")
            net_state_dict.update(ema_state_dict)

        # convert DTensor to Tensor
        for key, val in net_state_dict.items():
            if isinstance(val, DTensor):
                # Convert to full tensor
                net_state_dict[key] = val.full_tensor().detach().cpu()
            else:
                net_state_dict[key] = val.detach().cpu()

        return net_state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """
        Loads a state dictionary into the model and optionally its EMA counterpart.
        Different from torch strict=False mode, the method will not raise error for unmatched state shape while raise warning.

        Parameters:e
            state_dict (Mapping[str, Any]): A dictionary containing separate state dictionaries for the model and
                                            potentially for an EMA version of the model under the keys 'model' and 'ema', respectively.
            strict (bool, optional): If True, the method will enforce that the keys in the state dict match exactly
                                    those in the model and EMA model (if applicable). Defaults to True.
            assign (bool, optional): If True and in strict mode, will assign the state dictionary directly rather than
                                    matching keys one-by-one. This is typically used when loading parts of state dicts
                                    or using customized loading procedures. Defaults to False.
        """
        _reg_state_dict = collections.OrderedDict()
        _ema_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net."):
                _reg_state_dict[k.replace("net.", "")] = v
            elif k.startswith("net_ema."):
                _ema_state_dict[k.replace("net_ema.", "")] = v

        state_dict = _reg_state_dict

        if strict:
            reg_results: _IncompatibleKeys = self.pipe.dit.load_state_dict(
                _reg_state_dict, strict=strict, assign=assign
            )

            if self.config.pipe_config.ema.enabled:
                ema_results: _IncompatibleKeys = self.pipe.dit_ema.load_state_dict(
                    _ema_state_dict, strict=strict, assign=assign
                )

            return _IncompatibleKeys(
                missing_keys=reg_results.missing_keys
                + (ema_results.missing_keys if self.config.pipe_config.ema.enabled else []),
                unexpected_keys=reg_results.unexpected_keys
                + (ema_results.unexpected_keys if self.config.pipe_config.ema.enabled else []),
            )
        else:
            log.critical("load model in non-strict mode")
            log.critical(non_strict_load_model(self.pipe.dit, _reg_state_dict), rank0_only=False)
            if self.config.pipe_config.ema.enabled:
                log.critical("load ema model in non-strict mode")
                log.critical(non_strict_load_model(self.pipe.dit_ema, _ema_state_dict), rank0_only=False)

    # ------------------ public methods ------------------
    def ema_beta(self, iteration: int) -> float:
        """
        Calculate the beta value for EMA update.
        weights = weights * beta + (1 - beta) * new_weights

        Args:
            iteration (int): Current iteration number.

        Returns:
            float: The calculated beta value.
        """
        iteration = iteration + self.config.pipe_config.ema.iteration_shift
        if iteration < 1:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.pipe.ema_exp_coefficient + 1)

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: bool | None = None,
    ) -> torch.Tensor:
        return clip_grad_norm_(
            self.net.parameters(),
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )
