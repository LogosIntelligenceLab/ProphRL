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

import math
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from einops import rearrange
from megatron.core import parallel_state
from torch.distributed import get_process_group_ranks
from tqdm import tqdm

from cosmos_predict2.conditioner import DataType, TextCondition
from cosmos_predict2.configs.base.config_video2world import ConditioningStrategy, Video2WorldPipelineConfig
from cosmos_predict2.configs.base.config_multiview_action import (
    MultiviewStatePredPipelineConfig,
)
from cosmos_predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_predict2.models.utils import init_weights_on_device, load_state_dict
from cosmos_predict2.module.denoise_prediction import DenoisePrediction
from cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.schedulers.rectified_flow_scheduler import (
    RectifiedFlowAB2Scheduler,
)
from cosmos_predict2.utils.context_parallel import (
    broadcast_split_tensor,
    cat_outputs_cp,
    split_inputs_cp,
)
from imaginaire.lazy_config import instantiate
from imaginaire.utils import log, misc
from imaginaire.utils.ema import FastEmaModelUpdater

IS_PREPROCESSED_KEY = "is_preprocessed"
_VIDEO_EXTENSIONS = [".mp4"]
NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"
SAMPLE_N_VIEWS_KEY: str = "sample_n_views"

class MultiviewVideo2WorldActionConditionedStatePredPipeline(Video2WorldPipeline):
    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        
    @staticmethod
    def from_config(
        config: MultiviewStatePredPipelineConfig,
        dit_path: str = "",
        use_text_encoder: bool = False,         # not used in action-conditioned generation
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_ema_to_reg: bool = False,
        load_prompt_refiner: bool = False,
    ) -> Any:
        pipe = MultiviewVideo2WorldActionConditionedStatePredPipeline(device=device, torch_dtype=torch_dtype)
        pipe.config = config
        pipe.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        pipe.tensor_kwargs = {"device": device, "dtype": pipe.precision}
        log.warning(f"[MultiviewAction] precision {pipe.precision}")

        # 1. data keys
        pipe.sigma_data = config.sigma_data
        pipe.setup_data_key()

        # 2. scheduler + scaling
        pipe.scheduler = RectifiedFlowAB2Scheduler(
            sigma_min=config.timestamps.t_min,
            sigma_max=config.timestamps.t_max,
            order=config.timestamps.order,
            t_scaling_factor=config.rectified_flow_t_scaling_factor,
        )
        pipe.scaling = RectifiedFlowScaling(pipe.sigma_data, config.rectified_flow_t_scaling_factor)

        # 3. tokenizer
        pipe.tokenizer = instantiate(config.tokenizer)
        assert pipe.tokenizer.latent_ch == pipe.config.state_ch, (
            f"latent_ch {pipe.tokenizer.latent_ch} != state_shape {pipe.config.state_ch}"
        )

        # 4. text encoder
        if use_text_encoder:
            from imaginaire.auxiliary.text_encoder import get_cosmos_text_encoder

            pipe.text_encoder = get_cosmos_text_encoder(config=config.text_encoder, device=device)
        else:
            pipe.text_encoder = None

        # 5. conditioner：ActionBlackHistoryMultiViewStatePredConditioner
        pipe.conditioner = instantiate(config.conditioner)
        assert sum(p.numel() for p in pipe.conditioner.parameters() if p.requires_grad) == 0, (
            "conditioner should not have learnable parameters"
        )

        # prompt refiner / guardrail
        if load_prompt_refiner:
            from cosmos_predict2.auxiliary.cosmos_reason1 import CosmosReason1

            pipe.prompt_refiner = CosmosReason1(
                checkpoint_dir=config.prompt_refiner_config.checkpoint_dir,
                offload_model_to_cpu=config.prompt_refiner_config.offload_model_to_cpu,
                enabled=config.prompt_refiner_config.enabled,
            )

        if config.guardrail_config.enabled:
            from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

            pipe.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                config.guardrail_config.checkpoint_dir, config.guardrail_config.offload_model_to_cpu
            )
            pipe.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
                config.guardrail_config.checkpoint_dir, config.guardrail_config.offload_model_to_cpu
            )
        else:
            pipe.text_guardrail_runner = None
            pipe.video_guardrail_runner = None

        # 6. DiT
        if dit_path:
            log.info(f"[MultiviewAction] Loading DiT from {dit_path}")
        else:
            log.warning("[MultiviewAction] dit_path not provided, initializing DiT with random weights")
        
        dit_config = config.net
        pipe.dit = instantiate(dit_config).eval()  # inference

        if dit_path:
            state_dict = load_state_dict(dit_path)
            prefix_to_load = "net_ema." if load_ema_to_reg else "net."
            state_dict_dit_compatible = {
                key[len(prefix_to_load) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix_to_load)
            }
            if not state_dict_dit_compatible:
                model_keys = set(pipe.dit.state_dict())
                state_dict_dit_compatible = {
                    key: value for key, value in state_dict.items() if key in model_keys
                }
            if not state_dict_dit_compatible:
                alternate_prefix = "net." if prefix_to_load == "net_ema." else "net_ema."
                state_dict_dit_compatible = {
                    key[len(alternate_prefix) :]: value
                    for key, value in state_dict.items()
                    if key.startswith(alternate_prefix)
                }
            if not state_dict_dit_compatible:
                raise ValueError(
                    f"Checkpoint {dit_path} contains neither bare model keys nor net/net_ema-prefixed parameters"
                )

            incompatible = pipe.dit.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
            if incompatible.missing_keys:
                raise RuntimeError(
                    f"Checkpoint {dit_path} is missing {len(incompatible.missing_keys)} model keys: "
                    f"{incompatible.missing_keys[:10]}"
                )

            allowed_history_prefixes = ("history_tokenizer.", "mem_to_k.", "mem_to_v.", "vpb.")
            unexpected = incompatible.unexpected_keys
            if not pipe.dit.use_history:
                unexpected = [key for key in unexpected if not key.startswith(allowed_history_prefixes)]
            if unexpected:
                raise RuntimeError(
                    f"Checkpoint {dit_path} has {len(unexpected)} unexpected model keys: {unexpected[:10]}"
                )
            del state_dict, state_dict_dit_compatible
            log.success(f"[MultiviewAction] Successfully loaded DiT from {dit_path}")
        pipe.dit = pipe.dit.to(device=device, dtype=torch_dtype)

        # 6-2. EMA
        if config.ema.enabled:
            pipe.dit_ema = instantiate(dit_config).eval()
            pipe.dit_ema.requires_grad_(False)

            pipe.dit_ema_worker = FastEmaModelUpdater()
            s = config.ema.rate
            pipe.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()
            pipe.dit_ema_worker.copy_to(src_model=pipe.dit, tgt_model=pipe.dit_ema)

        torch.cuda.empty_cache()

        # 7. dp size
        if parallel_state.is_initialized():
            pipe.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            pipe.data_parallel_size = 1

        return pipe
    
    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        n_views = state.shape[2] // self.tokenizer.get_pixel_num_frames(self.config.state_t)
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if n_views > 4 and cp_size > 1 and n_views <= cp_size:
            return self.encode_cp(state)
        state = rearrange(state, "B C (V T) H W -> (B V) C T H W", V=n_views)
        encoded_state = super().encode(state)
        encoded_state = rearrange(encoded_state, "(B V) C T H W -> B C (V T) H W", V=n_views)
        return encoded_state
    
    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        n_views = latent.shape[2] // self.config.state_t
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if n_views > 4 and cp_size > 1 and n_views <= cp_size:
            return self.decode_cp(latent)
        latent = rearrange(latent, "B C (V T) H W -> (B V) C T H W", V=n_views)
        decoded_state = super().decode(latent)
        decoded_state = rearrange(decoded_state, "(B V) C T H W -> B C (V T) H W", V=n_views)
        return decoded_state
    
    @torch.no_grad()
    def encode_cp(self, state: torch.Tensor) -> torch.Tensor:
        cp_size = len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
        cp_group = parallel_state.get_context_parallel_group()
        n_views = state.shape[2] // self.tokenizer.get_pixel_num_frames(self.config.state_t)
        assert n_views < cp_size, f"n_views must be less than cp_size, got n_views={n_views} and cp_size={cp_size}"
        state_V_B_C_T_H_W = rearrange(state, "B C (V T) H W -> V B C T H W", V=n_views)
        state_input = torch.zeros((cp_size, *state_V_B_C_T_H_W.shape[1:]), **self.tensor_kwargs)
        state_input[0:n_views] = state_V_B_C_T_H_W
        local_state_V_B_C_T_H_W = broadcast_split_tensor(state_input, seq_dim=0, process_group=cp_group)
        local_state = rearrange(local_state_V_B_C_T_H_W, "V B C T H W -> (B V) C T H W")
        encoded_state = super().encode(local_state)
        encoded_state_list = [torch.empty_like(encoded_state) for _ in range(cp_size)]
        dist.all_gather(encoded_state_list, encoded_state, group=cp_group)
        encoded_state = torch.cat(encoded_state_list[0:n_views], dim=2)  # [B, C, V * T, H, W]
        return encoded_state

    @torch.no_grad()
    def decode_cp(self, latent: torch.Tensor) -> torch.Tensor:
        cp_size = len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
        cp_group = parallel_state.get_context_parallel_group()
        log.info(f"[MultiviewAction] latent.shape: {latent.shape}")
        log.info(f"[MultiviewAction] self.config.state_t: {self.config.state_t}")
        n_views = latent.shape[2] // self.config.state_t
        assert n_views < cp_size, f"n_views must be less than cp_size, got n_views={n_views} and cp_size={cp_size}"
        latent_V_B_C_T_H_W = rearrange(latent, "B C (V T) H W -> V B C T H W", V=n_views)
        latent_input = torch.zeros((cp_size, *latent_V_B_C_T_H_W.shape[1:]), **self.tensor_kwargs)
        latent_input[0:n_views] = latent_V_B_C_T_H_W
        local_latent_V_B_C_T_H_W = broadcast_split_tensor(latent_input, seq_dim=0, process_group=cp_group)
        local_latent = rearrange(local_latent_V_B_C_T_H_W, "V B C T H W -> (B V) C T H W")
        decoded_state = super().decode(local_latent)
        decoded_state_list = [torch.empty_like(decoded_state) for _ in range(cp_size)]
        dist.all_gather(decoded_state_list, decoded_state, group=cp_group)
        decoded_state = torch.cat(decoded_state_list[0:n_views], dim=2)  # [B, C, V * T, H, W]
        return decoded_state
    
    def _normalize_video_databatch_inplace(self, data_batch: dict[str, torch.Tensor], input_key: str = None):  # noqa: RUF013
        input_key = self.input_video_key if input_key is None else input_key
        if input_key in data_batch:
            num_video_frames_per_view = self.tokenizer.get_pixel_num_frames(self.config.state_t)
            n_views = data_batch[input_key].shape[2] // num_video_frames_per_view
            data_batch[input_key] = rearrange(data_batch[input_key], "B C (V T) H W -> (B V) C T H W", V=n_views)
            super()._normalize_video_databatch_inplace(data_batch, input_key)
            data_batch[input_key] = rearrange(data_batch[input_key], "(B V) C T H W -> B C (V T) H W", V=n_views)
            
    def _normalize_video_databatch_inplace_black_history(self, data_batch: dict[str, torch.Tensor], input_key: str = None, check_key: str = None) -> None:  # noqa: RUF013
        input_key = self.input_video_key if input_key is None else input_key
        check_key = IS_PREPROCESSED_KEY if check_key is None else check_key
        # only handle video batch
        if input_key in data_batch:
            # Check if the data has already been normalized and avoid re-normalizing
            if check_key in data_batch and data_batch[check_key] is True:
                pass
            else:
                # assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
                data_batch[input_key] = data_batch[input_key].to(torch.uint8)
                data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[check_key] = True

    def _augment_image_dim_inplace(self, data_batch: dict[str, torch.Tensor], input_key: str = None, check_key: str = None) -> None:  # noqa: RUF013
        input_key = self.input_image_key if input_key is None else input_key
        check_key = IS_PREPROCESSED_KEY if check_key is None else check_key
        if input_key in data_batch:
            # Check if the data has already been augmented and avoid re-augmenting
            if check_key in data_batch and data_batch[check_key] is True:
                assert data_batch[input_key].shape[2] == 1, (
                    f"Image data is claimed be augmented while its shape is {data_batch[input_key].shape}"
                )
                return
            else:
                data_batch[input_key] = rearrange(data_batch[input_key], "b c h w -> b c 1 h w").contiguous()
                data_batch[check_key] = True
                
    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, TextCondition]:
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        if 'blacks' in data_batch.keys():
            self._normalize_video_databatch_inplace_black_history(data_batch, input_key='blacks', check_key='is_black_preprocessed')
            if 'is_blacks_latent' not in data_batch.keys() or not data_batch['is_blacks_latent']:
                data_batch['blacks'] = self.encode(data_batch['blacks']).contiguous().float()
                data_batch['is_blacks_latent'] = True
        if 'history' in data_batch.keys():
            self._normalize_video_databatch_inplace_black_history(data_batch, input_key='history', check_key='is_history_preprocessed')
            if 'is_history_latent' not in data_batch.keys() or not data_batch['is_history_latent']:
                data_batch['history'] = self.encode(data_batch['history']).contiguous().float()
                data_batch['is_history_latent'] = True
        is_image_batch = self.is_image_batch(data_batch)

        # Latent state
        raw_state = data_batch[self.input_image_key if is_image_batch else self.input_video_key]
        latent_state = self.encode(raw_state).contiguous().float()

        # Condition
        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        condition = condition.set_video_condition(
            state_t=self.config.state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
        )
        return raw_state, latent_state, condition
            
    def _get_data_batch_input(
        self,
        video: torch.Tensor,            # (B, C, T=V*T_per_view, H, W)
        actions: np.ndarray | torch.Tensor,
        state: torch.Tensor | None,
        blacks: torch.Tensor | None,
        history: torch.Tensor | None,
        latent_view_indices: torch.Tensor | None,
        prompt: str,
        negative_prompt: str = "",
        num_latent_conditional_frames: int = 1,
        n_views: int = 3,
        fps: int = 10,
    ):
        B, C, T, H, W = video.shape
        t5_text_embeddings = torch.zeros(B, n_views * 512, 1024, dtype=self.torch_dtype, device=self.device)
        
        if latent_view_indices is None:
            latent_view_indices_T = torch.repeat_interleave(torch.arange(n_views, device=self.device), self.config.state_t)
            latent_view_indices_B_T = latent_view_indices_T.unsqueeze(0).expand(B, -1)
        else:
            if len(latent_view_indices.shape) == 1:
                latent_view_indices_B_T = latent_view_indices.to(self.device).unsqueeze(0).expand(B, -1)
            elif len(latent_view_indices.shape) == 2:
                latent_view_indices_B_T = latent_view_indices.to(self.device)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        actions = actions.to(device=self.device, dtype=self.torch_dtype)

        data_batch = {
            "sample_n_views": n_views,
            "latent_view_indices_B_T": latent_view_indices_B_T,
            "ref_cam_view_idx_sample_position": torch.tensor([-1], device=self.device),
            "dataset_name": "video_data",
            "video": video,
            "t5_text_embeddings": t5_text_embeddings,
            "fps": fps * torch.ones(B, device=self.device),
            "padding_mask": torch.zeros(B, 1, H, W, device=self.device),
            "num_conditional_frames": num_latent_conditional_frames,
            "action": actions,
        }

        if blacks is not None:
            data_batch["blacks"] = blacks
        if history is not None:
            data_batch["history"] = history
        if state is not None:
            data_batch["state"] = state
        if negative_prompt:
            log.warning("[MultiviewAction] Negative prompt only used for text CFG; here mostly action-cond.")
            neg_t5 = torch.zeros(B, n_views * 512, 1024, dtype=self.torch_dtype, device=self.device)
            neg_t5[:, 0:512] = self.encode_prompt(negative_prompt).to(dtype=self.torch_dtype, device=self.device)
            data_batch["neg_t5_text_embeddings"] = neg_t5

        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                data_batch[k] = v.to(device=self.device, dtype=torch.bfloat16)
        return data_batch
    
    def get_x0_fn_from_batch(
        self,
        data_batch: dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        use_cuda_graphs: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch)
        # override condition with inference mode; num_conditional_frames used Here!
        condition = condition.set_video_condition(
            state_t=self.config.state_t,
            gt_frames=x0,
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=num_conditional_frames,
        )
        uncondition = uncondition.set_video_condition(
            state_t=self.config.state_t,
            gt_frames=x0,
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=num_conditional_frames,
        )
        condition = condition.edit_for_inference(
            is_cfg_conditional=True,
            condition_locations=self.config.condition_locations,
            num_conditional_frames_per_view=num_conditional_frames,
        )
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False,
            condition_locations=self.config.condition_locations,
            num_conditional_frames_per_view=num_conditional_frames,
        )
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if not parallel_state.is_initialized():
            assert not self.dit.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(noise_x, sigma, condition).x0
            uncond_x0 = self.denoise(noise_x, sigma, uncondition).x0
            raw_x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)

            if "guided_image" in data_batch:
                # replacement trick that enables inpainting with base model
                assert "guided_mask" in data_batch, "guided_mask should be in data_batch if guided_image is present"
                guide_image = data_batch["guided_image"]
                guide_mask = data_batch["guided_mask"]
                raw_x0 = guide_mask * guide_image + (1 - guide_mask) * raw_x0
            return raw_x0

        return x0_fn
    
    def broadcast_split_for_model_parallelsim(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: torch.Tensor,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
    ):
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        n_views = x0_B_C_T_H_W.shape[2] // self.config.state_t
        if cp_size > 1 and n_views > 1:
            x0_B_C_T_H_W = rearrange(x0_B_C_T_H_W, "B C (V T) H W -> (B V) C T H W", V=n_views).contiguous()
            if epsilon_B_C_T_H_W is not None:
                epsilon_B_C_T_H_W = rearrange(
                    epsilon_B_C_T_H_W, "B C (V T) H W -> (B V) C T H W", V=n_views
                ).contiguous()
            reshape_sigma_B_T = False
            if sigma_B_T is not None:
                assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
                if sigma_B_T.shape[-1] != 1:
                    assert sigma_B_T.shape[-1] % n_views == 0, (
                        f"sigma_B_T temporal dimension T must either be 1 or a multiple of sample_n_views. Got T={sigma_B_T.shape[-1]} and sample_n_views={n_views}"
                    )
                    sigma_B_T = rearrange(sigma_B_T, "B (V T) -> (B V) T", V=n_views).contiguous()
                    reshape_sigma_B_T = True
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T = super().broadcast_split_for_model_parallelsim(
                x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
            )
            x0_B_C_T_H_W = rearrange(x0_B_C_T_H_W, "(B V) C T H W -> B C (V T) H W", V=n_views)
            if epsilon_B_C_T_H_W is not None:
                epsilon_B_C_T_H_W = rearrange(epsilon_B_C_T_H_W, "(B V) C T H W -> B C (V T) H W", V=n_views)
            if reshape_sigma_B_T:
                sigma_B_T = rearrange(sigma_B_T, "(B V) T -> B (V T)", V=n_views)
        return x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
    
    def denoise(
        self, xt_B_C_T_H_W: torch.Tensor, sigma: torch.Tensor, condition: TextCondition, use_cuda_graphs: bool = False
    ) -> DenoisePrediction:
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (TextCondition): conditional information, generated from self.conditioner
            use_cuda_graphs (bool, optional): Whether to use CUDA Graphs for inference. Defaults to False.

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred).
        """

        if sigma.ndim == 1:
            sigma_B_T = rearrange(sigma, "b -> b 1")
        elif sigma.ndim == 2:
            sigma_B_T = sigma
        else:
            raise ValueError(f"sigma shape {sigma.shape} is not supported")

        sigma_B_1_T_1_1 = rearrange(sigma_B_T, "b t -> b 1 t 1 1")
        # get precondition for the network
        c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = self.scaling(sigma=sigma_B_1_T_1_1)

        net_state_in_B_C_T_H_W = xt_B_C_T_H_W * c_in_B_1_T_1_1

        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(net_state_in_B_C_T_H_W) / self.config.sigma_data
            if not condition.use_video_condition:
                # When using random dropout, we zero out the ground truth frames
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                net_state_in_B_C_T_H_W
            )

            if self.config.conditioning_strategy == str(ConditioningStrategy.FRAME_REPLACE):
                # In case of frame replacement strategy, replace the first few frames of the video with the conditional frames
                # Update the c_noise as the conditional frames are clean and have very low noise

                # Make the first few frames of x_t be the ground truth frames
                net_state_in_B_C_T_H_W = (
                    condition_state_in_B_C_T_H_W * condition_video_mask
                    + net_state_in_B_C_T_H_W * (1 - condition_video_mask)
                )

                # Adjust c_noise for the conditional frames
                sigma_cond_B_1_T_1_1 = torch.ones_like(sigma_B_1_T_1_1) * self.config.sigma_conditional
                _, _, _, c_noise_cond_B_1_T_1_1 = self.scaling(sigma=sigma_cond_B_1_T_1_1)
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                c_noise_B_1_T_1_1 = c_noise_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + c_noise_B_1_T_1_1 * (
                    1 - condition_video_mask_B_1_T_1_1
                )
            elif self.config.conditioning_strategy == str(ConditioningStrategy.CHANNEL_CONCAT):
                # In case of channel concatenation strategy, concatenate the conditional frames in the channel dimension
                condition_state_in_masked_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask
                net_state_in_B_C_T_H_W = torch.cat([net_state_in_B_C_T_H_W, condition_state_in_masked_B_C_T_H_W], dim=1)

        else:
            # In case of image batch, simply concatenate the 0 frames when channel concat strategy is used
            if self.config.conditioning_strategy == str(ConditioningStrategy.CHANNEL_CONCAT):
                net_state_in_B_C_T_H_W = torch.cat(
                    [net_state_in_B_C_T_H_W, torch.zeros_like(net_state_in_B_C_T_H_W)], dim=1
                )

        # forward pass through the network
        net_output_B_C_T_H_W = self.dit(
            x_B_C_T_H_W=net_state_in_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(**self.tensor_kwargs),
            **condition.to_dict(),
            use_cuda_graphs=use_cuda_graphs,
        )
        net_output_B_C_T_H_W = net_output_B_C_T_H_W.float()

        x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W
        if condition.is_video:
            # Set the first few frames to the ground truth frames. This will ensure that the loss is not computed for the first few frames.
            x0_pred_B_C_T_H_W = condition.gt_frames.type_as(
                x0_pred_B_C_T_H_W
            ) * condition_video_mask + x0_pred_B_C_T_H_W * (1 - condition_video_mask)

        # get noise prediction
        eps_pred_B_C_T_H_W = (xt_B_C_T_H_W - x0_pred_B_C_T_H_W) / sigma_B_1_T_1_1

        return DenoisePrediction(x0_pred_B_C_T_H_W, eps_pred_B_C_T_H_W, None)
    
    @torch.no_grad()
    def __call__(
        self,
        video_tensor: torch.Tensor | None,
        input_path: str | None,
        actions: np.ndarray,
        state: np.ndarray | None,
        blacks: np.ndarray | None,
        history: torch.Tensor | None,
        latent_view_indices: torch.Tensor | None,
        prompt: str = "",
        negative_prompt: str = "",
        aspect_ratio: str = "16:9",
        num_conditional_frames: int = 1,
        guidance: float = 0.0,
        n_views: int = 3,
        fps: int = 10,
        num_sampling_step: int = 35,
        seed: int = 0,
        use_cuda_graphs: bool = False,
        return_prompt: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, str] | None:

        # 1. resolution + conditional frame
        width, height = VIDEO_RES_SIZE_INFO[self.config.resolution][aspect_ratio]
        height, width = self.check_resize_height_width(height, width)
        assert num_conditional_frames in [0, 1, 5], "num_conditional_frames must be 0, 1 or 5"
        num_latent_conditional_frames = (
            self.tokenizer.get_latent_num_frames(num_conditional_frames) if num_conditional_frames > 0 else 0
        )

        log.info(f"[MultiviewAction] n_views={n_views}")
        log.info(f"[MultiviewAction] state_t={self.config.state_t}")
        num_video_frames = self.tokenizer.get_pixel_num_frames(self.config.state_t) * n_views
        log.info(f"[MultiviewAction] num_video_frames={num_video_frames}")
        
        # 2. Prepare the in-memory multi-view tensor. ProphRL always provides
        # one real view plus one adapter-created black view; file decoding is
        # intentionally kept outside this inference pipeline.
        if video_tensor is not None:
            vid_input = video_tensor.detach().to(torch.uint8)
            if vid_input.ndim == 4:
                vid_input = vid_input.unsqueeze(0)
            if vid_input.ndim != 5:
                raise ValueError(f"video_tensor must be [C,T,H,W] or [B,C,T,H,W], got {vid_input.shape}")
            if vid_input.shape[2] != num_video_frames:
                raise ValueError(
                    f"video_tensor has {vid_input.shape[2]} frames, expected {num_video_frames} "
                    f"for {n_views} internal views"
                )
            vid_input = rearrange(vid_input, "b c (v t) h w -> b c v t h w", v=n_views)
            vid_input[:, :, :, 1:] = 0
            vid_input = rearrange(vid_input, "b c v t h w -> b c (v t) h w")
        else:
            raise ValueError("Bridge inference requires video_tensor; input_path decoding is not supported")

        # 3. convert blacks / history / state to tensor
        blacks_tensor = None
        history_tensor = None
        state_tensor = None
        if blacks is not None:
            blacks_tensor = torch.from_numpy(blacks).to(device=self.device)
        if history is not None:
            history_tensor = history.to(device=self.device)
        if state is not None:
            state_tensor = torch.from_numpy(state).to(device=self.device)

        # 4. organize to data_batch
        data_batch = self._get_data_batch_input(
            vid_input.to(device=self.device),
            actions=actions,
            state=state_tensor,
            blacks=blacks_tensor,
            history=history_tensor,
            latent_view_indices=latent_view_indices,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_latent_conditional_frames=num_latent_conditional_frames,
            n_views=n_views,
            fps=fps,
        )

        # 5. preprocess multiview video
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)

        # 6. blacks / history
        if "blacks" in data_batch:
            self._normalize_video_databatch_inplace_black_history(data_batch, input_key="blacks")
            if not data_batch.get("is_blacks_latent", False):
                data_batch["blacks"] = self.encode(data_batch["blacks"]).contiguous().float()
                data_batch["is_blacks_latent"] = True

        if "history" in data_batch:
            self._normalize_video_databatch_inplace_black_history(data_batch, input_key='history', check_key='is_history_preprocessed')
            if 'is_history_latent' not in data_batch.keys() or not data_batch['is_history_latent']:
                data_batch["history"] = self.encode(data_batch["history"]).contiguous().float()
                data_batch["is_history_latent"] = True
        
        # 7. shape / x0_fn
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_video_key
        n_sample = data_batch[input_key].shape[0]
        _T, _H, _W = data_batch[input_key].shape[-3:]
        state_shape = [
            self.config.state_ch,
            self.config.state_t * n_views,  # latent = state_t * n_views
            _H // self.tokenizer.spatial_compression_factor,
            _W // self.tokenizer.spatial_compression_factor,
        ]

        x0_fn = self.get_x0_fn_from_batch(
            data_batch,
            guidance=guidance,
            is_negative_prompt=True,
            use_cuda_graphs=use_cuda_graphs,
        )

        log.info("[MultiviewAction] Starting video generation...")

        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            * self.scheduler.config.sigma_max
        )

        # 8. context parallel split
        if self.dit.is_context_parallel_enabled:
            log.info("[MultiviewAction] Splitting input for context parallel.")
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.get_context_parallel_group())

        # 9. Rectified Flow sampling loop
        scheduler = self.scheduler
        scheduler.set_timesteps(num_sampling_step, device=x_sigma_max.device)
        sample = x_sigma_max.to(dtype=torch.float32)
        x0_prev: torch.Tensor | None = None

        for i, _ in enumerate(tqdm(scheduler.timesteps, desc="Generating world", leave=False)):
            sigma_t = scheduler.sigmas[i].to(sample.device, dtype=torch.float32)
            sigma_in = sigma_t.repeat(sample.shape[0])
            x0_pred = x0_fn(sample, sigma_in)
            sample, x0_prev = scheduler.step(
                x0_pred=x0_pred,
                i=i,
                sample=sample,
                x0_prev=x0_prev,
            )

        sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
        sigma_in = sigma_min.repeat(sample.shape[0])
        samples = x0_fn(sample, sigma_in)

        # 10. merge context parallel's chunk
        if self.dit.is_context_parallel_enabled:
            cp_group = self.get_context_parallel_group()
            cp_size = 1 if cp_group is None else cp_group.size()
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=cp_group)
            if n_views > 1:
                samples = rearrange(
                    samples, "B C (c V T) H W -> B C (V c T) H W", c=cp_size, T=self.config.state_t // cp_size
                )

        # 11. decode to image
        video = self.decode(samples)

        # 12. guardrail
        if self.video_guardrail_runner is not None:
            from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

            video = video.clamp(-1.0, 1.0)
            video_normalized = (video + 1) / 2
            video_squeezed = video_normalized.squeeze(0)
            frames = (video_squeezed * 255).clamp(0, 255).to(torch.uint8)
            frames = frames.permute(1, 2, 3, 0).cpu().numpy()

            processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
            if processed_frames is None:
                return None
            log.success("[MultiviewAction] Passed guardrail on generated video")
            processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
            processed_video = processed_video * 2 - 1
            processed_video = processed_video.unsqueeze(0)
            video = processed_video.to(video.device, dtype=video.dtype)

        log.success("[MultiviewAction] Video generation completed successfully")
        if return_prompt:
            return video, prompt
        else:
            return video
