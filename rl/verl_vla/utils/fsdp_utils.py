# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from torch.distributed.fsdp.wrap import (
    _or_policy,
    _module_wrap_policy,
    lambda_auto_wrap_policy,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from contextlib import contextmanager, nullcontext
from packaging import version
import torch

from transformers.trainer_pt_utils import get_module_class_from_name
import torch.nn as nn
from verl_vla.utils.vla_utils.openvla_oft.modeling_prismatic import  PrismaticProjector
if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import Shard

    fully_shard_module = torch.distributed.fsdp._fully_shard._fully_shard
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard

    fully_shard_module = torch.distributed._composable.fsdp.fully_shard
else:
    fully_shard, MixedPrecisionPolicy, FSDPModule, CPUOffloadPolicy, fully_shard_module = None, None, None, None, None
    

def fsdp_version(model):
    if isinstance(model, FSDP):
        return 1
    elif isinstance(model, FSDPModule):
        return 2
    else:
        return 0

def get_fsdp_full_state_dict(model: torch.nn.Module, offload_to_cpu: bool = True, rank0_only: bool = True):
    """
    Get the full state dict from an FSDP model.

    Args:
        model (torch.nn.Module): The FSDP model to get state dict from
        offload_to_cpu (bool, optional): Whether to offload the state dict to CPU. Defaults to True.
        rank0_only (bool, optional): Whether to only get state dict on rank 0. Defaults to True.

    Returns:
        dict: The full state dict of the model

    Raises:
        NotImplementedError: If the FSDP version is unknown
    """
    if fsdp_version(model) == 1:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        state_dict_config = FullStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=rank0_only)
        with get_fsdp_state_ctx(
            model, state_type=StateDictType.FULL_STATE_DICT, state_cfg=state_dict_config, optim_cfg=None
        ):
            state_dict = model.state_dict()
        return state_dict
    elif fsdp_version(model) == 2:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        state_dict_config = StateDictOptions(
            full_state_dict=True, cpu_offload=offload_to_cpu, broadcast_from_rank0=not rank0_only
        )
        state_dict = get_model_state_dict(model, options=state_dict_config)
        return state_dict
    else:
        raise NotImplementedError(f"Unknown FSDP version {fsdp_version}")

def get_fsdp_state_ctx(model, state_type, state_cfg, optim_cfg):
    if fsdp_version(model) == 1:
        return FSDP.state_dict_type(model, state_type, state_cfg, optim_cfg)
    else:
        return nullcontext()
      
def init_fn(x: torch.nn.Module):
    if not torch.distributed.get_rank() == 0:
        x = x.to_empty(device=torch.cuda.current_device(), recurse=False)
        torch.cuda.empty_cache()
    return x


def get_init_weight_context_manager(use_meta_tensor=True):
    from accelerate import init_empty_weights
    cpu_init_weights = lambda: torch.device('cpu')
    if use_meta_tensor:
        init_context = init_empty_weights if torch.distributed.get_rank() != 0 else cpu_init_weights
    else:
        init_context = cpu_init_weights
    return init_context

# Copyright 2020-present the HuggingFace Inc. team.
# Adapted from https://github.com/huggingface/transformers/src/transformers/trainer.py
# def get_fsdp_wrap_policy(module, config=None):
#     if config is None:

#     if config.get('disable', False):

#     fsdp_transformer_layer_cls_to_wrap = config.get("transformer_layer_cls_to_wrap",
#                                                     default_transformer_cls_names_to_wrap)
#     if min_num_params > 0:
#         for layer_class in fsdp_transformer_layer_cls_to_wrap:
#             if transformer_cls is None:

#             transformer_auto_wrap_policy,
#             # Transformer layer class to wrap

def get_fsdp_wrap_policy(module, config=None, is_lora=False):
    """Get FSDP wrap policy for the module.
    
    Args:
        module: The module to get wrap policy for
        config: Configuration for wrap policy
        is_lora: Whether to enable lambda policy for LoRA modules
    """
    if config is None:
        config = {}

    if config.get('disable', False):
        return None

    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("transformer_layer_cls_to_wrap",
                                                    default_transformer_cls_names_to_wrap)
    min_num_params = config.get('min_num_params', 0)
    auto_wrap_policy = None

    policies = []

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    # Add lambda policy for LoRA modules if is_lora is True
    if is_lora:

        def lambda_policy_fn(module):
            if (len(list(module.named_children())) == 0 and getattr(module, "weight", None) is not None and
                    module.weight.requires_grad):
                return True
            return False

        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
        policies.append(lambda_policy)

    if min_num_params > 0:
        size_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        policies.append(size_policy)
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        transformer_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )
        policies.append(transformer_policy)

    if len(policies) > 0:
        auto_wrap_policy = functools.partial(_or_policy, policies=policies)

    return auto_wrap_policy

def get_fsdp_wrap_policy_vla(module, config=None, is_lora=False):
    
    from timm.models.vision_transformer import Block, VisionTransformer
    from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy, lambda_auto_wrap_policy
    vit_wrap_policy = functools.partial(_module_wrap_policy, module_classes={VisionTransformer})
    transformer_block_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    vision_fsdp_wrapping_policy = functools.partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    default_transformer_cls_names_to_wrap = getattr(module.language_model, "_no_split_modules", None)
    
    fsdp_transformer_layer_cls_to_wrap = default_transformer_cls_names_to_wrap

    llm_wrap_policy = None
    
    if fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            print("layer_class is :", layer_class)
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        llm_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            # Transformer layer class to wrap
            transformer_layer_cls=transformer_cls_to_wrap,
        )
    print("llm_wrap_policy:",llm_wrap_policy)
    assert llm_wrap_policy is not None

      




    # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector`
    #     _module_wrap_policy,
    prismatic_fsdp_wrapping_policy = functools.partial(
        _module_wrap_policy,
        module_classes={PrismaticProjector},
    )

    
    # Add lambda policy for LoRA modules if is_lora is True
    if is_lora:
        def lambda_policy_fn(module):
            return bool(
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )
        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)


    # Return union (_or_) over constituent policies
    #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
    #            automatically be folded into the root VLM FSDP instance.
    if is_lora:
        vla_policies=[
                vision_fsdp_wrapping_policy,
                llm_wrap_policy,
                prismatic_fsdp_wrapping_policy,
                lambda_policy
            ]
    else:
        vla_policies=[
            vision_fsdp_wrapping_policy,
            llm_wrap_policy,
            prismatic_fsdp_wrapping_policy,
        ]
    
    return functools.partial(
        _or_policy,
        policies=vla_policies
    )


def tag_value_head(root: nn.Module):
    name_list = []
    for name, m in root.named_modules():
        if name.endswith(("value_head")):
            setattr(m, "_fsdp_wrap_me", True)
            name_list.append(name)
    print(name_list)
                
                
def get_fsdp_wrap_policy_pi05(root_module, wrap_qkv_linears: bool = False, is_critic: bool = False, is_lora: bool = False):
    """
    返回 (auto_wrap_policy, ignored_modules)
    - 只 wrap：语言塔 decoder block、视觉 ViT block、（可选）attention 子模块
    - 忽略：视觉 patch-embed/stem/早期 Conv2d
    """
    # ---------- 语言塔 ----------
    if is_critic:
        tag_value_head(root_module)
    policies = []
    
    for name, m in root_module.named_modules():
        if name.endswith(("self_attn.q_proj", "self_attn.k_proj",
                          "self_attn.v_proj", "self_attn.o_proj")):
            if "vision_model" not in name:
                setattr(m, "_fsdp_wrap_me", True)
    
    for name, m in root_module.named_modules():
        if name.endswith(("state_proj", "action_in_proj", "action_out_proj", 
                          "time_mlp_in", "time_mlp_out")):
            setattr(m, "_fsdp_wrap_me", True)
            
    target_suffix = "paligemma.language_model.embed_tokens"
    for name, m in root_module.named_modules():
        # if name.endswith(target_suffix) and isinstance(m, nn.Embedding):
        if isinstance(m, nn.Embedding):
            setattr(m, "_fsdp_wrap_me", True)

    policies = []

    def is_tagged(m: nn.Module) -> bool:
        return getattr(m, "_fsdp_wrap_me", False)
    tag_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=is_tagged)
    policies.append(tag_policy)
        
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector
    from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer, GemmaRMSNorm, GemmaMLP
    
    module_classes_to_wrap = {
        GemmaDecoderLayer,
        GemmaRMSNorm,
        GemmaMLP,
        PaliGemmaMultiModalProjector,
        SiglipVisionModel,
    }
    
    module_policy = functools.partial(_module_wrap_policy, module_classes=module_classes_to_wrap)
    policies.append(module_policy)

    auto_wrap_policy = functools.partial(_or_policy, policies=policies)
    return auto_wrap_policy, None

      
def tag_lm_head(root: nn.Module):
    for name, m in root.named_modules():
        if name.endswith(("lm_head")):
            setattr(m, "_fsdp_wrap_me", True)
                    
def get_fsdp_wrap_policy_vla_adapter(module, config = None, is_lora: bool = False):
    """Get FSDP wrap policy for the module.
    
    Args:
        module: The module to get wrap policy for
        config: Configuration for wrap policy
        is_lora: Whether to enable lambda policy for LoRA modules
    """
    if config is None:
        config = {}

    if config.get('disable', False):
        return None

    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("transformer_layer_cls_to_wrap",
                                                    default_transformer_cls_names_to_wrap)
    min_num_params = config.get('min_num_params', 0)
    auto_wrap_policy = None

    policies = []

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    # Add lambda policy for LoRA modules if is_lora is True
    # if is_lora:

    #     def lambda_policy_fn(module):
    #         if (len(list(module.named_children())) == 0 and getattr(module, "weight", None) is not None and
    #                 module.weight.requires_grad):


    if min_num_params > 0:
        size_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        policies.append(size_policy)
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        transformer_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )
        policies.append(transformer_policy)


    from verl_vla.utils.vla_utils.vla_adapter.prismatic.extern.hf.modeling_prismatic import PrismaticVisionBackbone
    from transformers import Qwen2ForCausalLM
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
    from verl_vla.utils.vla_utils.vla_adapter.prismatic.models.action_heads import FlowMatchingActionHead
    from verl_vla.utils.vla_utils.vla_adapter.prismatic.models.projectors import NoisyActionProjector
    from prismatic.models.diffusion_transformer import  DiT_SingleTokenAction_OneCtx
    module_classes_to_wrap = {
        PrismaticVisionBackbone,
        # Qwen2ForCausalLM,
        # nn.Embedding,
        Qwen2DecoderLayer,
        Qwen2RMSNorm,
        PrismaticProjector,
        NoisyActionProjector,
        # FlowMatchingActionHead,
        DiT_SingleTokenAction_OneCtx,
    }
    
    module_policy = functools.partial(_module_wrap_policy, module_classes=module_classes_to_wrap)
    policies.append(module_policy)
    
    # if hasattr(lm, "lm_head"):
    # def is_tagged(m: nn.Module) -> bool:
    
    if len(policies) > 0:
        auto_wrap_policy = functools.partial(_or_policy, policies=policies)

    
    
    ignored_modules = [module.action_queries]
    lm = getattr(module, "language_model", module)
    ignored_modules.append(lm.lm_head)
    # if hasattr(lm, "get_input_embeddings"):
    # assert emb_layer is not None, "找不到 embedding 模块"
    
    # assert emb_layer is not None, "找不到 embedding 模块"
    
    return auto_wrap_policy, ignored_modules


def get_fsdp_wrap_policy_openvla_oft_flow(module, config = None, is_lora: bool = False):
    """Get FSDP wrap policy for the module.
    
    Args:
        module: The module to get wrap policy for
        config: Configuration for wrap policy
        is_lora: Whether to enable lambda policy for LoRA modules
    """
    if config is None:
        config = {}

    if config.get('disable', False):
        return None

    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("transformer_layer_cls_to_wrap",
                                                    default_transformer_cls_names_to_wrap)
    min_num_params = config.get('min_num_params', 0)
    auto_wrap_policy = None

    policies = []

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    if min_num_params > 0:
        size_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        policies.append(size_policy)
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        transformer_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )
        policies.append(transformer_policy)


    from verl_vla.utils.vla_utils.openvla_oft_flow.prismatic.extern.hf.modeling_prismatic import PrismaticVisionBackbone
    from transformers import Qwen2ForCausalLM
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
    from verl_vla.utils.vla_utils.openvla_oft_flow.prismatic.models.action_heads import FlowMatchingActionHead
    from verl_vla.utils.vla_utils.openvla_oft_flow.prismatic.models.projectors import NoisyActionProjector
    from prismatic.models.diffusion_transformer import  DiT_SingleTokenAction_OneCtx
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention, LlamaRMSNorm, LlamaMLP
    module_classes_to_wrap = {
        PrismaticVisionBackbone,
        # Qwen2ForCausalLM,
        # nn.Embedding,
        Qwen2DecoderLayer,
        Qwen2RMSNorm,
        PrismaticProjector,
        NoisyActionProjector,
        # FlowMatchingActionHead,
        DiT_SingleTokenAction_OneCtx,
        LlamaDecoderLayer,
        LlamaAttention,
        LlamaRMSNorm
    }
    
    module_policy = functools.partial(_module_wrap_policy, module_classes=module_classes_to_wrap)
    policies.append(module_policy)
    
    
    if len(policies) > 0:
        auto_wrap_policy = functools.partial(_or_policy, policies=policies)

    
    
    ignored_modules = []
    lm = getattr(module, "language_model", module)
    ignored_modules.append(lm.lm_head)
    
    emb_module = None
    if hasattr(lm, "get_input_embeddings"):
        try:
            emb_module = lm.get_input_embeddings()
        except Exception:
            emb_module = None
    if isinstance(emb_module, torch.nn.Module):
        ignored_modules.append(emb_module)
        
    return auto_wrap_policy, ignored_modules


def get_fsdp_wrap_policy_vla_adapter_params_full(module, config = None, is_lora: bool = False):
    """Get FSDP wrap policy for the module.
    
    Args:
        module: The module to get wrap policy for
        config: Configuration for wrap policy
        is_lora: Whether to enable lambda policy for LoRA modules
    """
    if config is None:
        config = {}

    if config.get('disable', False):
        return None

    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("transformer_layer_cls_to_wrap",
                                                    default_transformer_cls_names_to_wrap)
    min_num_params = config.get('min_num_params', 0)
    auto_wrap_policy = None

    policies = []

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    # Add lambda policy for LoRA modules if is_lora is True
    if is_lora:

        def lambda_policy_fn(module):
            if (len(list(module.named_children())) == 0 and getattr(module, "weight", None) is not None and
                    module.weight.requires_grad):
                return True
            return False

        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
        policies.append(lambda_policy)

    if min_num_params > 0:
        size_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        policies.append(size_policy)
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        transformer_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )
        policies.append(transformer_policy)


    from verl_vla.utils.vla_utils.vla_adapter.prismatic.extern.hf.modeling_prismatic import PrismaticVisionBackbone, PrismaticProjector
    from transformers import Qwen2ForCausalLM
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
    module_classes_to_wrap = {
        PrismaticVisionBackbone,
        # Qwen2ForCausalLM,
        nn.Embedding,
        Qwen2DecoderLayer,
        Qwen2RMSNorm,
        PrismaticProjector,
    }
    
    module_policy = functools.partial(_module_wrap_policy, module_classes=module_classes_to_wrap)
    policies.append(module_policy)
    
    # if hasattr(lm, "lm_head"):
    # def is_tagged(m: nn.Module) -> bool:
    
    # def is_tagged(m: nn.Module) -> bool:
    
    if len(policies) > 0:
        auto_wrap_policy = functools.partial(_or_policy, policies=policies)

    
    
    ignored_modules = [module.action_queries]
    # if hasattr(lm, "get_input_embeddings"):
    # assert emb_layer is not None, "找不到 embedding 模块"
    
    # assert emb_layer is not None, "找不到 embedding 模块"

    lm = getattr(module, "language_model", module)
    ignored_modules.append(lm.lm_head)

    
    return auto_wrap_policy, ignored_modules

def tag_action_embedder(root: nn.Module):
    name_list = []
    for name, m in root.named_modules():
        if ("action_embedder_B_D_agi" in name) \
            or ("action_embedder_B_3D_agi" in name) \
            or ("x_embedder" in name) \
            or ("t_embedder" in name) \
            or ("t_embedding_norm" in name) \
            or ("view_embeddings" in name):
            # or ("blocks" in name):
            setattr(m, "_fsdp_wrap_me", True)
            name_list.append(name)
        else:
            if isinstance(m, CheckpointWrapper):
                setattr(m, "_fsdp_wrap_me", True)
            name_list.append(name)
            
    
def get_fsdp_wrap_policy_wm(root_module):
    
    tag_action_embedder(root_module)
    def tag_mem_head(root: nn.Module):
        for name, m in root.named_modules():
            if (name.endswith(("mem_to_k"))) or \
                (name.endswith(("mem_to_v"))) or \
                    (name.endswith(("mem_to_crossattn_emb"))):
                setattr(m, "_fsdp_wrap_me", True)
    tag_mem_head(root_module)
    policies = []
    def is_tagged(m: nn.Module) -> bool:
        return getattr(m, "_fsdp_wrap_me", False)
    tag_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=is_tagged)
    policies.append(tag_policy)

    from cosmos_predict2.models.video2world_action_dit import FramePackMemory as FramePackMemory_sv
    from cosmos_predict2.models.video2world_multiview_action_state_pred_dit_11 import (
        FramePackMemory,
        Mlp,
        PoseAwareHistoryTokenizer,
    )
    module_classes_to_wrap = {
        FramePackMemory,
        FramePackMemory_sv,
        Mlp,
        PoseAwareHistoryTokenizer
    }
    module_policy = functools.partial(_module_wrap_policy, module_classes=module_classes_to_wrap)
    policies.append(module_policy)

    auto_wrap_policy = functools.partial(_or_policy, policies=policies)
    return auto_wrap_policy

def get_fsdp_wrap_policy_rm(root_module):
    
    
    # def is_tagged(m: nn.Module) -> bool:
    
    return None

def offload_fsdp_grad(module):
    for _, param in module.named_parameters():
        if param.grad is not None:
            param.grad = param.grad.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()


def load_fsdp_grad(module, device_id):
    for _, param in module.named_parameters():
        if param.grad is not None:
            param.grad = param.grad.to(device_id, non_blocking=True)
    torch.cuda.empty_cache()


def offload_fsdp_param_and_grad(module, offload_grad=False):
    for _, param in module.named_parameters():
        if hasattr(param, "_local_shard"):
            param._local_shard = param._local_shard.to("cpu", non_blocking=True)
        param.data = param.data.to('cpu', non_blocking=True)
        if offload_grad and param.grad is not None:
            param.grad = param.grad.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()


def load_fsdp_param_and_grad(module, device_id, load_grad=False):
    for _, param in module.named_parameters():
        if hasattr(param, "_local_shard"):
            param._local_shard = param._local_shard.to(device_id, non_blocking=True)
        param.data = param.data.to(device_id, non_blocking=True)
        if load_grad and param.grad is not None:
            param.grad = param.grad.to(device_id, non_blocking=True)
    torch.cuda.empty_cache()


def offload_fsdp_optimizer(optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()


def load_fsdp_optimizer(optimizer, device_id):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)
    torch.cuda.empty_cache()
