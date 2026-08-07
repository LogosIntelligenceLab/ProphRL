"""
modeling_prismatic.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions.
Inherits from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained,
but exactly replicate the logic in `prismatic.models.vlms.prismatic.py`.
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
import math
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    STOP_INDEX,
    NormalizationType,
    NUM_TOKENS
)
from .configuration_prismatic import OpenVLAConfig, PrismaticConfig



# Set up logger
logger = logging.getLogger(__name__)


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper



# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor



def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma



# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    """
    Vision backbone for Prismatic models that handles image feature extraction.

    Supports both single backbone (e.g., SigLIP) and fused backbone (e.g., SigLIP + DINOv2) configurations.
    For fused backbones, features from both models are concatenated along the feature dimension.
    """

    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        """
        Initialize the vision backbone.

        Args:
            use_fused_vision_backbone: Whether to use two backbones and fuse their features
            image_sizes: List of image sizes for each backbone
            timm_model_ids: List of TIMM model IDs to use for each backbone
            timm_override_act_layers: List of activation layer overrides for each backbone
        """
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.num_images_in_input = 1  # Default value, can be overridden later

        # Validate number of (fused) vision backbones
        if len(timm_model_ids) > 2:
            raise ValueError("Prismatic models only support up to 2 (fused) vision backbones!")

        # Create primary featurizer
        self.featurizer = self._create_featurizer(
            model_id=timm_model_ids[0], img_size=image_sizes[0], act_layer=timm_override_act_layers[0]
        )
        self.embed_dim = self.featurizer.embed_dim

        # Create secondary featurizer if using fused backbone
        if self.use_fused_vision_backbone:
            self.fused_featurizer = self._create_featurizer(
                model_id=timm_model_ids[1], img_size=image_sizes[1], act_layer=timm_override_act_layers[1]
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch LayerScale modules for HF compatibility
        self._patch_layer_scales()


    def _create_featurizer(self, model_id: str, img_size: int, act_layer: Optional[str]) -> nn.Module:
        """
        Create a TIMM-based featurizer model with appropriate configurations.

        Args:
            model_id: The TIMM model ID to load
            img_size: Input image size for the model
            act_layer: Override for the activation layer type

        Returns:
            A configured featurizer model
        """
        featurizer = timm.create_model(
            model_id,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
            act_layer=act_layer,
        )

        # Monkey-patch the forward function to extract the second-to-last layer features
        num_blocks = len(featurizer.blocks)
        featurizer.forward = unpack_tuple(partial(featurizer.get_intermediate_layers, n={num_blocks - 2}))

        return featurizer


    def _patch_layer_scales(self) -> None:
        """
        Patch all LayerScale modules to be compatible with HF's parameter naming.

        HF Transformers overwrites parameters with names containing 'gamma',
        so we need to rename and modify the forward method.
        """
        # Patch primary featurizer
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        # Patch secondary featurizer if it exists
        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)


    def get_num_patches(self) -> int:
        """
        Returns the number of vision patches output by the vision backbone.

        Returns:
            Number of patches per image
        """
        return self.featurizer.patch_embed.num_patches


    def get_num_images_in_input(self) -> int:
        """
        Returns the number of input images for the vision backbone.

        Returns:
            Number of images expected in the input
        """
        return self.num_images_in_input


    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """
        Sets the number of input images for the vision backbone.

        Args:
            num_images_in_input: Number of images to expect in the input
        """
        self.num_images_in_input = num_images_in_input


    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the vision backbone.

        If `self.use_fused_vision_backbone == True`, uses both SigLIP and DINOv2 transformers to extract visual features
        (otherwise uses SigLIP only). Allows multi-image inputs (but only for fused vision backbone).

        Args:
            pixel_values (torch.Tensor): Pixels for input image(s), (B, C, H, W).
        """
        if self.num_images_in_input == 1:
            if not self.use_fused_vision_backbone:
                return self.featurizer(pixel_values)

            # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

            return torch.cat([patches, patches_fused], dim=2)

        else:
            assert self.use_fused_vision_backbone, "Multi-image inputs require using fused backbone!"

            # Split `pixel_values` into individual images (each with 6 channels: 3 for SigLIP + 3 for DINOv2)
            images = torch.split(pixel_values, [6] * self.num_images_in_input, dim=1)

            # Process each image and collect patches
            all_patches = []
            for img in images:
                # Split each image further into two stacks of channels (each with 3 channels)
                img_regular, img_fused = torch.split(img, [3, 3], dim=1)

                # Get patches from both SigLIP and DINOv2 vision transformers
                patches = self.featurizer(img_regular)
                patches_fused = self.fused_featurizer(img_fused)

                # Concatenate SigLIP and DINOv2 patches along the hidden dimension
                combined_patches = torch.cat([patches, patches_fused], dim=2)
                all_patches.append(combined_patches)

            # Concatenate all patches along the patch dimension
            return torch.cat(all_patches, dim=1)



# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features



# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None



class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa



class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )
        
        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )

        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.llm_dim = config.text_config.hidden_size
        
        #Action query token
        self.action_queries = nn.Embedding(NUM_TOKENS, self.llm_dim)
        self.action_queries.weight.data.zero_()

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()
    def set_version(self, version: str):
        self.version = version
        return self.version


    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    def _replace_input_embeddings(self, input_embeddings, all_actions_mask, noisy_action_features):
        """
        Replace embeddings in input_embeddings at positions where all_actions_mask is True
        with embeddings from noisy_action_features, using vectorized operations.

        Args:
            input_embeddings: Tensor of shape (B, S, D)
            all_actions_mask: Boolean tensor of shape (B, S)
            noisy_action_features: Tensor of shape (B, K, D) where K is the number of True values in mask per sample

        Returns:
            Modified input_embeddings tensor
        """
        # Clone input to avoid modifying the original tensor
        new_input_embeddings = input_embeddings.clone()

        # Create a tensor with the same shape of input_embeddings to hold the noisy action features
        repositioned_noisy_action_features = torch.zeros_like(input_embeddings)

        # Create batch indices for splicing
        batch_indices = torch.arange(input_embeddings.shape[0], device=input_embeddings.device)
        batch_indices = batch_indices.unsqueeze(1).expand(-1, noisy_action_features.shape[1])

        # Get indices where mask is True for each sample
        masked_indices = torch.stack([torch.where(mask)[0] for mask in all_actions_mask])

        # Move the noisy action features into their correct positions
        noisy_action_features = noisy_action_features.to(repositioned_noisy_action_features.device)
        
        repositioned_noisy_action_features[batch_indices, masked_indices] = noisy_action_features.to(input_embeddings.dtype)

        # Combine original input embeddings and noisy action embeddings using the mask
        new_input_embeddings = torch.where(
            all_actions_mask.unsqueeze(-1), repositioned_noisy_action_features, new_input_embeddings
        )

        return new_input_embeddings

    def _process_action_masks(self, labels):
        """Helper to get action masks from labels"""
        current_action_mask = get_current_action_mask(labels)
        next_actions_mask = get_next_actions_mask(labels)
        all_actions_mask = current_action_mask | next_actions_mask  # (B, seq_len)
        return all_actions_mask

    def _process_vision_features(self, pixel_values, language_embeddings=None, use_film=False):
        """Process vision features with optional FiLM conditioning"""
        if use_film:
            # FiLM: Infuse language inputs into visual features
            patch_features = self.vision_backbone(pixel_values, language_embeddings)  # (bsz, 256 * num_images, D)
        else:
            patch_features = self.vision_backbone(pixel_values)  # (bsz, 256 * num_images, D)

        # Project patch embeddings into language embedding space
        proj_param = next(self.projector.parameters(), None)
        if proj_param is not None:
            # 若 projector 不在同一设备/精度，迁移过去（仅首次生效）
            if proj_param.device != patch_features.device or proj_param.dtype != patch_features.dtype:
                self.projector.to(device=patch_features.device, dtype=patch_features.dtype)
        return self.projector(patch_features)

    def _process_proprio_features(self, projected_patch_embeddings, proprio, proprio_projector):
        """Process proprioceptive features and append to vision features"""
        if proprio_projector is not None and proprio is not None:
            proprio = proprio.reshape(projected_patch_embeddings.shape[0], -1)  # (bsz, proprio_dim)
            proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
            proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
            # For simplicity, just append proprio token to the end of projected vision patch tokens
            return torch.cat((projected_patch_embeddings, proprio_features), dim=1)
        return projected_patch_embeddings

    # def _build_multimodal_attention(self, input_embeddings, projected_patch_embeddings, attention_mask):
    #     """Build multimodal embeddings and attention mask"""
    #     # Update attention mask
        
    #     if attention_mask is not None:

    #     # Build multimodal embeddings & attention mask; insert embeddings after <BOS> token (1:)

    #     if attention_mask is not None:


    def _build_multimodal_attention(self, input_embeddings, projected_patch_embeddings, attention_mask):
        """
        input_embeddings:            (B, S, D)  左填充：mask 形如 0...0111...
        projected_patch_embeddings:  (B, P, D)  要插在 <BOS> 之后
        attention_mask:              (B, S)
        return:
            multimodal_embeddings:   (B, S+P, D)
            multimodal_attention_mask:(B, S+P)
        """
        B, S, D = input_embeddings.shape
        assert projected_patch_embeddings.dim() == 3 and projected_patch_embeddings.size(0) == B
        P = projected_patch_embeddings.size(1)
        device = input_embeddings.device

        if attention_mask is None:
            # 无 mask 的退化路径：默认 BOS 在位置 0，保持原来的“1:”逻辑
            multimodal_embeddings = torch.cat(
                [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
            )
            multimodal_attention_mask = None
            return multimodal_embeddings, multimodal_attention_mask

        # 1) 计算每个样本的 BOS 位置 = 左侧连续 0 的个数（左填充长度）
        attn01 = (attention_mask != 0).to(torch.long)                  # 1=有效, 0=PAD
        left_pad_len = (torch.cumsum(attn01, dim=1) == 0).sum(dim=1)   # [B]
        # 极端情况防御：全 0 时把 BOS 当作最后一位（不会越界）
        bos_idx = left_pad_len
        # 2) 预分配新张量
        new_S = S + P
        multimodal_embeddings     = input_embeddings.new_zeros((B, new_S, D))
        multimodal_attention_mask = attention_mask.new_zeros((B, new_S))

        # 3) 逐样本拼装： [ ..PAD.. | BOS ] + [patches] + [ 余下原token... ]
        for i in range(B):
            b = int(bos_idx[i].item())     # BOS 索引
            a_end = b + 1                  # 含 BOS 的右开端点

            # 复制 <PAD..BOS>
            multimodal_embeddings[i, :a_end, :]      = input_embeddings[i, :a_end, :]
            multimodal_attention_mask[i, :a_end]     = attention_mask[i, :a_end]

            # 插入 patches
            multimodal_embeddings[i, a_end:a_end+P, :]  = projected_patch_embeddings[i]
            multimodal_attention_mask[i, a_end:a_end+P] = 1

            # 复制 BOS 之后剩余 token
            tail_len = S - a_end
            if tail_len > 0:
                multimodal_embeddings[i, a_end+P:a_end+P+tail_len, :]  = input_embeddings[i, a_end:, :]
                multimodal_attention_mask[i, a_end+P:a_end+P+tail_len] = attention_mask[i, a_end:]

        return multimodal_embeddings, multimodal_attention_mask


    def _build_multimodal_labels(self, labels, projected_patch_embeddings):
        """Build multimodal labels with IGNORE_INDEX for patch embeddings"""
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            return torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
        return None

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        proprio=None,
        proprio_projector=None,
        noisy_actions=None,
        noisy_action_projector=None,
        diffusion_timestep_embeddings=None,
        use_film: bool = False,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during multimodal forward!"

            # Get input embeddings (from language model embeddings)
            input_embeddings = self.get_input_embeddings()(input_ids)  # (B, seq_len, D)

            
            # Extract action masks
            all_actions_mask = self._process_action_masks(labels)

            # Extract the language portion of the input embeddings (i.e. remove the action tokens portion)
            
            language_embeddings = input_embeddings[~all_actions_mask].reshape(
                input_embeddings.shape[0], -1, input_embeddings.shape[2]
            )  # (B, lang_seq_len, llm_dim)

            # Get visual features
            projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

            # Process action embeddings
            if noisy_actions is not None:
                

                action_queries = self.action_queries.weight  # (1, h)
                action_queries = action_queries.view(1, action_queries.shape[0], action_queries.shape[1]).repeat(input_embeddings.shape[0], 1, 1)  # (b, chunk_size, h)
                all_actions_mask = self._process_action_masks(labels)
                input_embeddings = self._replace_input_embeddings(
                    input_embeddings, all_actions_mask, action_queries)
                

            else:
                action_queries = self.action_queries.weight  # (1, h)
                action_queries = action_queries.view(1, action_queries.shape[0], action_queries.shape[1]).repeat(input_embeddings.shape[0], 1, 1)  # (b, chunk_size, h)
                all_actions_mask = self._process_action_masks(labels)
                input_embeddings = self._replace_input_embeddings(
                    input_embeddings, all_actions_mask, action_queries)

            # Build multimodal embeddings & attention mask
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )
            
            # Build labels for multimodal sequence if needed
            multimodal_labels = self._build_multimodal_labels(labels, projected_patch_embeddings)

            # Dispatch to language model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                ) 

        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
            )


    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)



class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats
        

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
        """Prepares input for action prediction by adding necessary tokens"""
        # Add (ACTION_DIM * NUM_ACTIONS_CHUNK) placeholder tokens to input_ids to simulate action tokens
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], NUM_TOKENS)).to(input_ids.device).to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
        stop_token_id = torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype) * STOP_INDEX
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        mask_extension = (
            torch.ones((attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1]))
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    # def _prepare_input_for_action_prediction(self, input_ids, attention_mask):
    #     """
    #     attention_mask:   [B, S] (1=有效, 0=PAD；右填充，单调非增)
    #     作用：在每个样本的最后一个非PAD后插入 NUM_TOKENS 个 action 占位符 + 1 个 stop token，
    #         然后把 PAD 续在序列末尾，保持右填充。
    #     """

    #     # 你自己的占位符与stop id（建议用明确的常量）

    #     # 每个样本的有效长度（右填充前缀的1个数）


    #     # 先整体开好空间：input_ids 用 pad_id 填充，mask 用 0 填充

    #     # 批量插入
    #     # 说明：用简单for循环最直观可靠；B一般不大，这样足够快且不易出错
    #     for i in range(B):

    #         # 1) 复制原始有效token
    #         if L > 0:

    #         # 2) 插入 action 占位符
    #         if NUM_TOKENS > 0:

    #         # 3) 插入 stop token

    #         # 4) 更新 mask：有效范围 = 原有效长度 + action 段 + stop

    #     # 可选一致性检查：右填充应为单调非增


    def _prepare_labels_for_action_prediction(self, labels, input_ids):
        """Creates labels tensor for action prediction if not provided"""
        # Extend labels tensor with fake action labels
        ARBITRARY_ACTION_TOKEN_IDX = ACTION_TOKEN_BEGIN_IDX + 1
        labels_extension = (
            torch.ones((labels.shape[0], input_ids.shape[-1] - labels.shape[-1])).to(labels.device).to(labels.dtype)
            * ARBITRARY_ACTION_TOKEN_IDX
        )
        labels = torch.cat([labels, labels_extension], dim=-1)

        # Replace last label token with stop token
        labels[:, -1] = STOP_INDEX

        return labels

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """Unnormalize actions using dataset statistics"""
        action_norm_stats = self.get_action_stats(unnorm_key)

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )

        return actions


    def _regression_or_discrete_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
        proprio=None,
        proprio_projector=None,
    ):
        """Run L1 regression-based continuous action prediction or discrete action tokens prediction."""

        action_queries = self.action_queries.weight  # (1, h)
        action_queries = action_queries.view(1, action_queries.shape[0], action_queries.shape[1]).repeat(input_embeddings.shape[0], 1, 1)  # (b, chunk_size, h)
        # Replace action token embeddings with noisy action embeddings
        input_embeddings = self._replace_input_embeddings(input_embeddings.clone(), all_actions_mask, action_queries)

        # Build multimodal embeddings and attention mask
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # Forward pass through language model
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden states for action tokens
        multi_layer_hidden_states = []
        
        for item in language_model_output.hidden_states[0:]:
            # Get hidden states for text portion of prompt+response (after the vision patches)
            text_hidden_states = item
            # Get hidden states for action portion of response
            actions_hidden_states = text_hidden_states[:, NUM_PATCHES+ NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + NUM_TOKENS, :,].reshape(1, 1, NUM_TOKENS, -1).to(torch.bfloat16)
            
            batch_size = item.shape[0]
            task_latten_states = item[:, :NUM_PATCHES].reshape(batch_size, 1, NUM_PATCHES , -1)
            all_hidden_states = torch.cat((task_latten_states, actions_hidden_states),2)
            multi_layer_hidden_states.append(all_hidden_states)
            
        multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim = 1)
        

        # Handle different prediction methods
        if action_head is not None:
            # L1 regression prediction
            normalized_actions = action_head.predict_action(multi_layer_hidden_states,
                                                proprio=proprio,
                                                proprio_projector=proprio_projector)
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.float().cpu().detach().numpy()
        else:
            # Discrete token-based prediction
            predicted_action_token_ids = (
                language_model_output.logits[
                    :,
                    NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                ]
                .argmax(dim=2)
                .cpu()
                .numpy()
            )
            discretized_actions = self.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized_actions = self.bin_centers[discretized_actions]
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        ret_dict = {}
        return normalized_actions, ret_dict


    def _flow_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
        noisy_action_projector=None,
        proprio=None,
        proprio_projector=None,
        use_sde=False,
        recompute_log_prob=False,
        a_shape=(20,7)
    ):
        B, L, D = input_embeddings.shape
        action_queries = self.action_queries.weight  # (1, h)

        action_queries = action_queries.view(1, action_queries.shape[0], action_queries.shape[1]).repeat(input_embeddings.shape[0], 1, 1)  # (b, chunk_size, h)
        # Replace action token embeddings with noisy action embeddings
        input_embeddings = self._replace_input_embeddings(input_embeddings.clone(), all_actions_mask, action_queries)

        # Build multimodal embeddings and attention mask
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # Forward pass through language model
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )


        
        last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
        attn01 = (multimodal_attention_mask != 0).to(torch.long) 
        pad_len = (torch.cumsum(attn01, dim=1) == 0).sum(dim=1)
        # # pad_len = 
        def gather_span(x, start_idx, L):
            """从每个样本的start_idx开始，取长度L的连续片段 -> (B, L, D)"""
            rng = torch.arange(L, device=x.device)   # [L]
            idx = start_idx[:, None] + rng[None, :]  # [B, L]
            # 保险：不越界（若你的长度已保证充足，可改成assert）
            return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        
        # Get hidden states for text portion of prompt+response (after the vision patches)
        
        task_start = pad_len 
        task_latent_states = gather_span(last_hidden_states, task_start, NUM_PATCHES) \
                                .reshape(B, 1, NUM_PATCHES, D)
        actions_start = pad_len + NUM_PATCHES + NUM_PROMPT_TOKENS   # [B]
        actions_hidden_states = gather_span(last_hidden_states, actions_start, NUM_TOKENS) \
                                .reshape(B, 1, NUM_TOKENS, D) \
                                .to(torch.bfloat16)
        
        all_hidden_states = torch.cat((task_latent_states, actions_hidden_states), 2)
        
        if use_sde:
            x_t, return_dict = self.sample_action_sde(all_hidden_states, action_head, noisy_action_projector, a_shape=a_shape)
        else:
            x_t, return_dict = self.sample_action(all_hidden_states, action_head, noisy_action_projector, a_shape=a_shape)
        
        normalized_actions = x_t
        normalized_actions = normalized_actions.reshape(-1, a_shape[0], ACTION_DIM)
        normalized_actions = normalized_actions.float().cpu().detach().numpy()
        
        return normalized_actions, return_dict
    
    def sample_action(self, all_hidden_states, action_head, noisy_action_projector, a_shape=(20,7)):
        B = all_hidden_states.shape[0]
        device = all_hidden_states.device
        dt = -1.0 / 10
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        actions_shape = (B, *a_shape)
        noise = self.action_head.sample_noise(actions_shape, device)
            
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        x_t_all, t_all, x_next_all, log_probs = [], [], [], []
        while time >= -dt / 2:
            expanded_time = time.expand(B)
            x_t_ = x_t.reshape(B, -1).unsqueeze(-1).to(torch.bfloat16)
            rearranged_actions_hidden_states = self.noisy_action_projector(x_t_)
            rearranged_actions_hidden_states = rearranged_actions_hidden_states.reshape(B, NUM_ACTIONS_CHUNK, -1)
            v_t = self.action_head.flow_predictor(obs=rearranged_actions_hidden_states,hidden_states=all_hidden_states,time_step=expanded_time, proprio_states=None)
            # Euler step
            x_next = x_t + dt * v_t
            
            x_t_all.append(x_t)
            t_all.append(expanded_time.detach().cpu())
            x_next_all.append(x_next)
            
            x_t = x_next
            time += dt
            
        return_dict = {
            "x_t": x_t_all,
            "t": t_all,
            "x_next": x_next_all,
        }
        return x_t, return_dict
    
    def sample_action_sde(self, all_hidden_states, action_head, noisy_action_projector, a_shape=(20,7)):
        B = all_hidden_states.shape[0]
        device = all_hidden_states.device
        dt = -1.0 / 10
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        sqrt_abs_dt = torch.sqrt(-dt)
        sde_sigma_max = 0.07
        sde_sigma_power = 1.5
        
        actions_shape = (B, *a_shape)
        
        noise = self.action_head.sample_noise(actions_shape, device)
            
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        x_t_all, t_all, x_next_all, log_probs = [], [], [], []
        while time >= -dt / 2:
            expanded_time = time.expand(B)
            x_t_ = x_t.reshape(B, -1).unsqueeze(-1).to(torch.bfloat16)
            rearranged_actions_hidden_states = self.noisy_action_projector(x_t_)
            rearranged_actions_hidden_states = rearranged_actions_hidden_states.reshape(B, a_shape[0], -1)
        
            v_t = self.action_head.flow_predictor(obs=rearranged_actions_hidden_states,hidden_states=all_hidden_states,time_step=expanded_time, proprio_states=None)
            
            sigmas = torch.tensor([1.0000, 0.9601, 0.9133, 0.8577, 0.7904, 0.7073, 0.6022, 0.4649, 0.2780, 0.0089, 0.0000], device=v_t.device, dtype=v_t.dtype)
            index = (10 * (1 - time)).to(torch.long)
            sigma = sigmas[index]
            sigma_max = sigmas[1]
            noise_level = 0.7
            std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level
            
            drift = v_t + (std_dev_t ** 2 / (2 * time + 1e-6)) * (x_t + (1 - time) * v_t)
            mean = x_t + drift * dt
            std  = (sqrt_abs_dt * std_dev_t).clamp_min(1e-6)
            eps = torch.randn_like(x_t)
            x_next = mean + std * eps
            
            x_t_all.append(x_t)
            t_all.append(expanded_time.detach().cpu())
            x_next_all.append(x_next)
            
            x_t = x_next
            time += dt
            
        return_dict = {
            "x_t": x_t_all,
            "t": t_all,
            "x_next": x_next_all,
        }
        return x_t, return_dict
    

    def _embed_tokens_fsdp_safe(self, input_ids_ext):
        lm = self.language_model
        dev = next(lm.parameters()).device
        ids = input_ids_ext.to(device=dev, dtype=torch.long)

        # 重要：解扁平/聚合成 2-D 权重，再安全地调用 embedding
        # with torch.no_grad():
            # recurse=False 只解当前 FSDP 根；writeback=False 不修改原参数
        with FSDP.summon_full_params(lm, writeback=False, recurse=False):
            emb_layer = lm.get_input_embeddings()
            w = emb_layer.weight
            assert w.dim() == 2, f"expect 2-D, got {w.dim()}-D"
            out = emb_layer(ids)            # [BS, L_ext, H]
        return out

    def recompute_logp_from_batch(
        self,
        input_ids,          # [B,S,L]
        pixel_values,       # [B,S,I,H,W]
        attention_mask,     # [B,S,L] (1=非PAD)
        x_t,                # [B,S,K,CH,D] 或 [B,K,CH,D]
        t,                  # [B,S,K]       或 [B,K]
        x_next,             # [B,S,K,CH,D] 或 [B,K,CH,D]
        finish_step,        # [B]  以 CH 为基数的全局步计数
        action_head,
        noisy_action_projector,
        use_film: bool = False,
    ):
        device = input_ids.device
        B, S, L = input_ids.shape
        # ---- 统一 x_t/t/x_next 形状：引入 S 维 ----
        if x_t.dim() == 4:     # [B,K,CH,D]
            x_t    = x_t.unsqueeze(1)
            t      = t.unsqueeze(1)
            x_next = x_next.unsqueeze(1)
        _, _, K, CH, D = x_t.shape

        # ---- 展平 S 维到 BS ----
        BS = B * S
        input_ids_bs      = input_ids.reshape(BS, L)
        attention_bs      = attention_mask.reshape(BS, L)
        I, H, W           = pixel_values.shape[-3:]
        pixel_values_bs   = pixel_values.reshape(BS, I, H, W)

        # ---- labels & prompt长度（逐样本）----
        labels_bs = input_ids_bs.clone()
        labels_bs[:] = IGNORE_INDEX
        num_prompt_tokens_bs = attention_bs.to(torch.long).sum(dim=-1) - 1   # [BS]

        # ---- 扩展输入：插入 action 占位符与 stop ----
        input_ids_ext, attention_ext = self._prepare_input_for_action_prediction(input_ids_bs, attention_bs)
        labels_ext = self._prepare_labels_for_action_prediction(labels_bs, input_ids_ext)
        all_actions_mask = self._process_action_masks(labels_ext)            # [BS, L_ext] (bool)

        # ---- token 嵌入，并把 action 位置替换为 action_queries ----
        emb_layer = self.language_model.get_input_embeddings()
        emb_layer.to(input_ids.device)
        input_embeddings = emb_layer(input_ids_ext)
        
        # if action_queries.dim() == 2 and action_queries.size(0) == 1:
        action_queries = self.action_queries.weight  # (1, h)
        # with FSDP.summon_full_params(self.action_queries, writeback=False, rank0_only=False):
        #     # 如果你设计的是 (1,h)：
        action_queries = action_queries.view(1, action_queries.shape[0], action_queries.shape[1]).repeat(input_embeddings.shape[0], 1, 1)
        input_embeddings = self._replace_input_embeddings(input_embeddings, all_actions_mask, action_queries)

        # ---- 视觉投影 & 多模态拼装（支持 left/right pad）----
        lang_emb_bs = input_embeddings[~all_actions_mask].reshape(BS, -1, input_embeddings.size(-1))
        proj_patches_bs = self._process_vision_features(pixel_values_bs, lang_emb_bs, use_film)
        mm_emb, mm_att = self._build_multimodal_attention(input_embeddings, proj_patches_bs, attention_ext)  # [BS,Lmm,H], [BS,Lmm]

        # ---- 过语言模型 ----
        lm_out = self.language_model(
            input_ids=None,
            attention_mask=mm_att,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=mm_emb,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hs = lm_out.hidden_states[-1]   # [BS, Lmm, Hh]
        BS_, Lmm, Hh = last_hs.shape
        assert BS_ == BS

        # ---- 提取 task / action 片段（按样本 BOS 插入位置）----
        att01 = (mm_att != 0).to(torch.long)                                 # [BS, Lmm]
        bos_idx = (torch.cumsum(att01, dim=1) == 0).sum(dim=1).clamp(max=Lmm-1)  # [BS]
        NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
        NUM_TOKENS  = 64

        def gather_span(x, start_idx, Ltake):
            rng = torch.arange(Ltake, device=x.device)[None, :]              # [1,Ltake]
            idx = start_idx[:, None] + rng                                   # [BS,Ltake]
            return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1))) # [BS,Ltake,Hh]

        task_latent = gather_span(last_hs, bos_idx, NUM_PATCHES).reshape(BS, 1, NUM_PATCHES, Hh)
        action_start = bos_idx + NUM_PATCHES + num_prompt_tokens_bs      # [BS]
        action_hs = gather_span(last_hs, action_start, NUM_TOKENS).reshape(BS, 1, NUM_TOKENS, Hh).to(torch.bfloat16)
        all_hidden_states_bs = torch.cat([task_latent, action_hs], dim=2)    # [BS,1,P+T,Hh]

        # ---- 复制到 [BS*K,...] 并计算 v_t ----
        BSK = BS * K
        # x_t: [B,S,K,CH,D] -> [BSK,CH,D]
        x_t_bsk   = x_t.reshape(BSK, CH, D).to(device=device, dtype=torch.bfloat16)
        t_bsk     = t.reshape(BSK).to(device=device, dtype=torch.float32)
        all_hs_rep = all_hidden_states_bs.repeat_interleave(K, dim=0)        # [BSK,1,P+T,Hh]

        x_t_vec = x_t_bsk.reshape(BSK, -1).unsqueeze(-1)                     # [BSK,CH*D,1]
        obs = self.noisy_action_projector(x_t_vec).reshape(BSK, CH, -1)                              # [BSK,CH,?]

        v_t = self.action_head.flow_predictor(
            obs=obs,
            hidden_states=all_hs_rep,
            time_step=t_bsk,
            proprio_states=None,
        ).to(torch.float32).reshape(B, S, K, CH, D)

        # ---- SDE / 似然 ----
        t_f    = t.to(device=device, dtype=torch.float32)
        t_safe = t_f.clamp(1e-4, 1.0 - 1e-4)
        t3     = t_safe[..., None, None]                                      # [B,S,K,1,1]

        sigmas = torch.tensor([1.0000, 0.9601, 0.9133, 0.8577, 0.7904, 0.7073, 0.6022, 0.4649, 0.2780, 0.0089, 0.0000],
                            device=device, dtype=torch.float32)

        sigmas = torch.tensor([1.0000, 0.9601, 0.9133, 0.8577, 0.7904, 0.7073, 0.6022, 0.4649, 0.2780, 0.0089, 0.0000], device=v_t.device, dtype=v_t.dtype)
        index = (K * (1 - t_f)).to(torch.long)
        sigma = sigmas[index]
        sigma_max = sigmas[1]
        noise_level = 0.7
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level
        std_dev_t = std_dev_t[..., None, None]
        
        dt = -1.0 / float(K)
        sqrt_abs_dt = math.sqrt(-dt)

        x_t_f   = x_t.to(torch.float32)
        x_nextf = x_next.to(torch.float32)
        drift = v_t + (std_dev_t**2 / (2.0 * t3 + 1e-6)) * (x_t_f + (1.0 - t3) * v_t)     # [B,S,K,CH,D]
        mean  = x_t_f + dt * drift
        std   = torch.clamp(std_dev_t * sqrt_abs_dt, min=1e-6)                             # [B,S,K,1,1]

        # ---- 有效动作 mask（按 finish_step 截断）----
        CH_idx = torch.arange(CH, device=device)[None, None, :]    # [1,1,CH]
        S_idx  = torch.arange(S,  device=device)[None, :, None]    # [1,S,1]
        s_fin  = (finish_step.to(device) // CH).view(B, 1, 1)
        c_fin  = (finish_step.to(device) %  CH).view(B, 1, 1)
        mask_before = (S_idx <  s_fin).float()
        mask_equal  = (S_idx == s_fin).float() * (CH_idx < c_fin).float()
        mask_actions = (mask_before + mask_equal)                                   # [B,S,CH]
        mask_elem    = mask_actions[:, :, None, :, None]                            # [B,S,1,CH,1]

        # ---- 高斯 log-prob ----
        lp_elem = (
            -((x_nextf.detach() - mean) ** 2) / (2 * ((std)**2))
            - torch.log(std)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        n_action_steps = 20
        lp_elem = lp_elem[..., :n_action_steps, :]
        lp_elem = lp_elem * mask_elem

        logp_action = lp_elem.sum(dim=-3)                # [B,S,K,D]（保留维度）
        logp_step   = lp_elem.sum(dim=(-1, -2))          # [B,S,K]
        logp_outer  = logp_step.sum(dim=2)               # [B,S]
        logp_joint  = logp_outer.sum(dim=1)              # [B]

        # （可选）熵
        c0 = 0.5 * (1.0 + math.log(2.0 * math.pi))
        h_per_dim = c0 + torch.log(std).squeeze(-1).squeeze(-1)    # [B,S,K]
        num_valid_actions = mask_actions.sum(dim=-1)                # [B,S]
        num_valid_dims    = (num_valid_actions * D).unsqueeze(-1).to(h_per_dim.dtype)
        ent_step  = h_per_dim * num_valid_dims                      # [B,S,K]
        ent_outer = ent_step.sum(dim=2)                             # [B,S]
        ent_joint = ent_outer.sum(dim=1)                            # [B]
        original_action_dim = 7
        mean = mean[..., :original_action_dim]
        std_dev_t = std_dev_t[..., :original_action_dim]
        std = std[..., :original_action_dim]
        out_metric = self.summarize_logprob_metrics(
            logp_step, logp_outer, logp_joint,
            mean, std, x_nextf, mask_actions, mask_elem,
            t, sigmas, finish_step, original_action_dim=7
        )
        return {
            "logp_action": logp_action,  # [B,S,K,D]
            "logp_step":   logp_step,    # [B,S,K]
            "logp_outer":  logp_outer,   # [B,S]
            "logp_elem":  lp_elem, 
            "logp_joint":  logp_joint,   # [B]
            "entropy_step":  ent_step,   # [B,S,K]
            "entropy_outer": ent_outer,  # [B,S]
            "entropy_joint": ent_joint,  # [B]
            "mean": mean,                # [B,S,K,CH,D]
            "std":  std,                 # [B,S,K,1,1]
            "v_t":  v_t,                 # [B,S,K,CH,D]
            "mask_actions": mask_actions, # [B,S,CH]
            "out_metric": out_metric,
        }
    
    def summarize_logprob_metrics(
        self,
        logp_step, logp_outer, logp_joint,
        mean, std, x_next_f, mask_actions, mask_elem,
        t, sigmas, finish_step, original_action_dim=7
    ):
        out = {}
        # 1) NLL / BPD
        B = logp_joint.shape[0]
        num_valid_actions = mask_actions.sum(dim=-1)         # [B,S]
        valid_dims = (num_valid_actions * original_action_dim).sum(dim=-1)  # [B]
        bpd_joint = -logp_joint / (valid_dims.clamp_min(1) * math.log(2.0))
        out["bpd_joint_mean"] = bpd_joint
        out["logp_joint_mean"] = logp_joint

        # 2) Entropy
        c0 = 0.5 * (1.0 + math.log(2.0 * math.pi))
        std_use = std[..., :original_action_dim]
        H_elem = (c0 + torch.log(std_use)) * mask_elem[..., :original_action_dim]
        H_joint = H_elem.sum(dim=(-1,-2,-3,-4))  # [B]
        out["H_joint_mean"] = H_joint

        # 3) Calibration
        err = (x_next_f[..., :7] - mean[..., :7]) / (std[..., :7] + 1e-12)
        n_action_steps = 20
        err = err[..., :n_action_steps, :]
        with torch.autocast('cuda', enabled=False):
            err = err.to(torch.float64)
            m = mask_elem[..., :7].expand_as(err).to(torch.float64)   # 显式扩成与 err 同形状

            denom = m.sum(dim=(1,2,3,4)).clamp_min(1)
            z2 = (err.pow(2) * m).sum(dim=(1,2,3,4)) / denom
            z4 = (err.pow(4) * m).sum(dim=(1,2,3,4)) / denom

        out["z_m2"], out["z_m4"] = z2, z4

        # 4) K-decomp
        out["logp_k_first"] = torch.nanmean(logp_step[..., 0], dim=1)
        out["logp_k_last"]  = torch.nanmean(logp_step[..., -1], dim=1)

        # 5) Masks / stability
        out["frac_valid_actions"] = mask_actions.float().mean(dim=(1,2))

        return out

    
    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        use_sde: bool = False,
        recompute_log_prob: bool = False,
        a_shape = (20,7),
        **kwargs: str,
    ) -> np.ndarray:
        """Predict actions from input sequence, with options for different prediction methods.

        Args:
            input_ids: Input token ids
            unnorm_key: Key for unnormalization statistics
            proprio: Proprioceptive features
            proprio_projector: Projector for proprioceptive features
            action_head: Optional head for L1 regression or diffusion-based prediction
            noisy_action_projector: Projector for noisy actions in diffusion-based prediction
            use_film: Whether to use FiLM conditioning
            **kwargs: Additional arguments including pixel_values and attention_mask

        Returns:
            Tuple of (unnormalized_actions, action_hidden_states)
        """
        
        pixel_values = kwargs["pixel_values"] # [1, 12, 224, 224]
        attention_mask = kwargs["attention_mask"] # 

        # Create fake labels tensor (needed for action mask)
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        # Get number of tokens in prompt (excluding the start token)
        NUM_PROMPT_TOKENS = attention_mask.sum(dim=1) - 1
        

        # Prepare inputs by adding necessary tokens
        input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)
        # Update labels tensor for action mask computation later
        labels = self._prepare_labels_for_action_prediction(labels, input_ids)

        # Get input embeddings and action masks
        emb_layer = self.language_model.get_input_embeddings()
        emb_layer.to(input_ids.device)
        input_embeddings = emb_layer(input_ids)
        all_actions_mask = self._process_action_masks(labels)

        # Extract language embeddings
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        # Process vision features
        projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

        # Add proprioceptive features if provided
        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)

        # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
        NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()

        # Run regression or discrete token-based prediction
        if noisy_action_projector is not None:
            normalized_actions, return_dict = self._flow_prediction(
                input_embeddings,
                all_actions_mask,
                projected_patch_embeddings,
                attention_mask,
                labels,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                proprio=proprio, # [8]
                proprio_projector=proprio_projector,
                use_sde=use_sde,
                recompute_log_prob=recompute_log_prob,
                a_shape=a_shape,
                )
        else:
            normalized_actions, actions_hidden_states = self._regression_or_discrete_prediction(
                input_embeddings,
                all_actions_mask,
                projected_patch_embeddings,
                attention_mask,
                labels,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                action_head=action_head,
                proprio=proprio, # [8]
                proprio_projector=proprio_projector,
                )
           
        # Unnormalize predicted actions
        actions = self._unnormalize_actions(normalized_actions, unnorm_key)
        return actions, return_dict



    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        """Validate and resolve the unnormalization key for action statistics"""
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["min"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
