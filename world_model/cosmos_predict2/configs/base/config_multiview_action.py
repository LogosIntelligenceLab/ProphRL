# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Bridge state-prediction configuration used by ProphRL.

The upstream research tree contains several unrelated multiview variants.  This
module intentionally keeps only the 2B/720p/10fps Bridge configuration required
by ``CosMosWorldModel``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

import attrs

from cosmos_predict2.conditioner import (
    ActionBlackHistoryStateMultiviewConditioner,
    BooleanFlag,
    ConditionLocation,
    ReMapkey,
    TextAttr,
)
from cosmos_predict2.configs.base.config_video2world import (
    ConditioningStrategy,
    CosmosGuardrailConfig,
    CosmosReason1Config,
    SolverTimestampConfig,
)
from cosmos_predict2.configs.base.defaults.ema import EMAConfig
from cosmos_predict2.models.text2image_dit import SACConfig
from cosmos_predict2.models.video2world_multiview_action_state_pred_dit_11 import (
    ActionConditionedMultiViewStatePredDiT,
)
from cosmos_predict2.tokenizers.tokenizer import TokenizerInterface
from imaginaire.config import make_freezable
from imaginaire.constants import (
    CHECKPOINTS_DIR,
    COSMOS_REASON1_MODEL_DIR,
    get_cosmos_predict2_video2world_tokenizer,
)
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict


@make_freezable
@attrs.define(slots=False)
class MultiviewStatePredPipelineConfig:
    adjust_video_noise: bool
    conditioner: LazyDict[ActionBlackHistoryStateMultiviewConditioner]
    conditioning_strategy: str
    min_num_conditional_frames_per_view: int
    max_num_conditional_frames_per_view: int
    condition_locations: list[ConditionLocation]
    sigma_conditional: float
    net: LazyDict[ActionConditionedMultiViewStatePredDiT]
    tokenizer: LazyDict[TokenizerInterface]
    prompt_refiner_config: CosmosReason1Config
    guardrail_config: CosmosGuardrailConfig
    precision: str
    rectified_flow_t_scaling_factor: float
    resize_online: bool
    resolution: str
    ema: EMAConfig
    sigma_data: float = 1.0
    state_ch: int = 16
    state_t: int = 6
    text_encoder: Any = None
    input_video_key: str = "video"
    input_image_key: str = "images"
    timestamps: SolverTimestampConfig = attrs.field(
        factory=lambda: SolverTimestampConfig(
            nfe=35,
            t_min=0.01,
            t_max=200.0,
            order=7.0,
            is_forward=False,
        )
    )


_BRIDGE_STATE_PRED_NET = L(ActionConditionedMultiViewStatePredDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    atten_backend="minimal_a2a",
    pos_emb_cls="rope3d",
    pos_emb_learnable=False,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_h_extrapolation_ratio=3.0,
    rope_w_extrapolation_ratio=3.0,
    rope_t_extrapolation_ratio=6.0 / 24.0,
    extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False,
    sac_config=L(SACConfig)(every_n_blocks=1, mode="predict2_2b_720"),
    state_t=6,
    n_cameras_emb=4,
    view_condition_dim=7,
    concat_view_embedding=True,
    action_dim=14 * (21 - 1),
    use_black=True,
    use_history=True,
)


_BRIDGE_STATE_PRED_PIPELINE = MultiviewStatePredPipelineConfig(
    adjust_video_noise=True,
    conditioner=L(ActionBlackHistoryStateMultiviewConditioner)(
        fps=L(ReMapkey)(dropout_rate=0.0, dtype=None, input_key="fps", output_key="fps"),
        padding_mask=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="padding_mask",
            output_key="padding_mask",
        ),
        text=L(TextAttr)(dropout_rate=0.0, input_key=["t5_text_embeddings"]),
        use_video_condition=L(BooleanFlag)(
            dropout_rate=0.0,
            input_key="fps",
            output_key="use_video_condition",
        ),
        view_indices_B_T=L(ReMapkey)(
            input_key="latent_view_indices_B_T",
            output_key="view_indices_B_T",
            dropout_rate=0.0,
            dtype=None,
        ),
        ref_cam_view_idx_sample_position=L(ReMapkey)(
            input_key="ref_cam_view_idx_sample_position",
            output_key="ref_cam_view_idx_sample_position",
            dropout_rate=0.0,
            dtype=None,
        ),
    ),
    conditioning_strategy=str(ConditioningStrategy.FRAME_REPLACE),
    min_num_conditional_frames_per_view=1,
    max_num_conditional_frames_per_view=1,
    condition_locations=[ConditionLocation.FIRST_RANDOM_N],
    net=_BRIDGE_STATE_PRED_NET,
    precision="bfloat16",
    rectified_flow_t_scaling_factor=1.0,
    resize_online=True,
    resolution="720",
    ema=L(EMAConfig)(enabled=False),
    sigma_conditional=0.0001,
    sigma_data=1.0,
    state_ch=16,
    state_t=6,
    tokenizer=L(TokenizerInterface)(
        chunk_duration=81,
        temporal_window=16,
        load_mean_std=False,
        name="tokenizer",
        vae_pth=get_cosmos_predict2_video2world_tokenizer(model_size="2B"),
    ),
    prompt_refiner_config=CosmosReason1Config(
        checkpoint_dir=COSMOS_REASON1_MODEL_DIR,
        offload_model_to_cpu=True,
        enabled=False,
    ),
    guardrail_config=CosmosGuardrailConfig(
        checkpoint_dir=CHECKPOINTS_DIR,
        offload_model_to_cpu=True,
        enabled=False,
    ),
)


def get_cosmos_predict2_multiview_state_pred_pipeline(
    *,
    model_size: Literal["2B"],
    views: int,
    frames: int,
    resolution: Literal["720"] = "720",
    fps: Literal[10] = 10,
) -> MultiviewStatePredPipelineConfig:
    """Return an isolated config for the supported Bridge checkpoint."""

    requested = (model_size, resolution, fps, views, frames)
    supported = ("2B", "720", 10, 3, 21)
    if requested != supported:
        raise ValueError(f"Unsupported Bridge world-model config {requested}; expected {supported}")
    return deepcopy(_BRIDGE_STATE_PRED_PIPELINE)
