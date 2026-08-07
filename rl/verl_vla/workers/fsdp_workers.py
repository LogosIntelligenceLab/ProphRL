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
"""
The main entry point to run the PPO algorithm
"""

import os
import logging
import warnings
import ray
import torch
import torch.distributed
import numpy as np
from omegaconf import DictConfig, open_dict
from transformers import AutoModelForCausalLM
from safetensors.torch import load_file as safe_load_file
import sys

from verl_vla.single_controller.base import Worker
from verl_vla.single_controller.base.decorator import register, Dispatch
import verl_vla.utils.torch_functional as verl_F
from verl_vla import DataProto
from verl_vla.utils.model import compute_position_id_with_mask
from verl_vla.utils.fs import copy_local_path_from_hdfs
from verl_vla.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    load_fsdp_grad,
    offload_fsdp_grad,
    init_fn,
    get_init_weight_context_manager,
    get_fsdp_wrap_policy_vla,
    get_fsdp_wrap_policy_pi05,
    get_fsdp_wrap_policy_wm,
    get_fsdp_wrap_policy_rm,
    get_fsdp_wrap_policy_vla_adapter,
    get_fsdp_wrap_policy_vla_adapter_params_full,
    get_fsdp_wrap_policy_openvla_oft_flow
)
from verl_vla.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_param_and_grad, load_fsdp_optimizer, load_fsdp_param_and_grad
from verl_vla.utils.import_utils import import_external_libs
from verl_vla.utils.debug import log_gpu_memory_usage
import verl_vla.utils.hdfs_io as hdfs_io
from verl_vla.utils import hf_tokenizer
from ..trainer.ppo import core_algos
from verl_vla.utils.py_functional import append_to_dict
from codetiming import Timer


from verl_vla.utils.openvla_utils import update_auto_map , check_model_logic_mismatch
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
import json

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from torch.distributed.fsdp import FullStateDictConfig, ShardedStateDictConfig, LocalStateDictConfig
from torch.distributed.fsdp.api import BackwardPrefetch, ShardingStrategy
from torch.optim.lr_scheduler import LambdaLR
from verl_vla.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager, FSDPVLACheckpointManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))

def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import ListConfig, DictConfig
    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj

def _fsdp_get_state_dict(fsdp_model: FSDP, mode="sharded"):
    torch.distributed.barrier()
    with torch.no_grad():
        torch.cuda.synchronize()

        if mode == "full":
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            sdt = StateDictType.FULL_STATE_DICT
        elif mode == "local":
            cfg = LocalStateDictConfig()
            sdt = StateDictType.LOCAL_STATE_DICT
        else:  # "sharded"
            cfg = ShardedStateDictConfig(offload_to_cpu=True)
            sdt = StateDictType.SHARDED_STATE_DICT

        with FSDP.state_dict_type(fsdp_model, sdt, cfg):
            sd = fsdp_model.state_dict()
    torch.distributed.barrier()
    return sd


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy
    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


class RobActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # build device mesh
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

        self._is_lora = self.config.model.get('lora_rank', 0) > 0
        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            self._is_offload_grad = self.config.actor.fsdp_config.get('grad_offload', False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get('param_offload', False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.shape[0]
            self.config.actor.ppo_micro_batch_size //= self.device_mesh.shape[0]
        if self._is_rollout:
            self.config.rollout.log_prob_micro_batch_size //= self.device_mesh.shape[0]
        if self._is_ref:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.shape[0]
        
        self.use_world_model = self.config.world_model.dit_path != ""

        self.env_rollout = self.config.world_model.get('env_rollout', False)
        self.use_reward_model = self.config.rollout.reward_type == "vlac"
        

    def _build_model_optimizer(self,
                               model_path,
                               meta_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False,
                               unnorm_key=""):
        from verl_vla.utils.model import print_model_size, update_model_config
        from verl_vla.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, \
            CPUOffload
        from torch import optim
        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        local_path = copy_local_path_from_hdfs(model_path)
        #add oft
        if self.config.model.vla == "openvla-oft":
            from verl_vla.utils.vla_utils.openvla_oft.configuration_prismatic import OpenVLAConfig
            from verl_vla.utils.vla_utils.openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
            from verl_vla.utils.vla_utils.openvla_oft.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
            
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
            if self.rank == 0:
                update_auto_map(local_path)
                check_model_logic_mismatch(local_path)
            torch.distributed.barrier()
            
        elif self.config.model.vla == "openvla":
            from verl_vla.utils.vla_utils.openvla.configuration_prismatic import OpenVLAConfig
            from verl_vla.utils.vla_utils.openvla.modeling_prismatic import OpenVLAForActionPrediction
            from verl_vla.utils.vla_utils.openvla.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
            if self.rank == 0:
                update_auto_map(local_path)
                check_model_logic_mismatch(local_path)
            torch.distributed.barrier()
        
        elif self.config.model.vla == "pi05":
            from verl_vla.utils.vla_utils.pi05 import openpi as _openpi  # noqa: F401
            from openpi.policies import policy_config
            from openpi.shared import download
            from openpi.training import config as _config
            torch.distributed.barrier()
        elif self.config.model.vla == "vla-adapter":
            from verl_vla.utils.vla_utils.vla_adapter.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
            from verl_vla.utils.vla_utils.vla_adapter.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
            from verl_vla.utils.vla_utils.vla_adapter.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
            
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
            # if self.rank == 0:
            torch.distributed.barrier()
        elif self.config.model.vla == "openvla-oft-flow":
            from verl_vla.utils.vla_utils.openvla_oft_flow.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
            from verl_vla.utils.vla_utils.openvla_oft_flow.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
            from verl_vla.utils.vla_utils.openvla_oft_flow.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
            
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
            # if self.rank == 0:
            torch.distributed.barrier()
        #add end

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code, model = self.config.model.vla)

        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        if self.config.model.vla == "pi05":
            actor_model_config = _config.get_config("pi05_bridge")
        else:
            actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
            
            
        if self.config.model.use_remove_padding:
            from verl_vla.models.registry import check_model_support_rmpad
            check_model_support_rmpad(actor_model_config.model_type)
        if self.tokenizer is None:
            override_config_kwargs = {
                'bos_token_id': -1,
                'eos_token_id': -1,
                'pad_token_id': -1,
            }
        else:
            override_config_kwargs = {
                'bos_token_id': self.tokenizer.bos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
            }
        override_config_kwargs.update(override_model_config)
        if self.config.model.vla == "pi05":
            pass
        else:
            update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f'Model config after override: {actor_model_config}')

        if self.config.model.vla == "pi05":
            use_meta_tensor = False
        else:
            use_meta_tensor = not actor_model_config.tie_word_embeddings
        init_context = get_init_weight_context_manager(use_meta_tensor=use_meta_tensor)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.config.model.vla == "openvla-oft":
                
                actor_module = OpenVLAForActionPrediction.from_pretrained(
                                                        pretrained_model_name_or_path=local_path,
                                                        torch_dtype=torch_dtype,
                                                        attn_implementation="flash_attention_2",
                                                        config=actor_model_config,              
                                                        trust_remote_code=True,
                                                    )
                #oft add
                actor_module.vision_backbone.set_num_images_in_input(self.config.actor.num_images_in_input)
                
                dataset_statistics_path = os.path.join(local_path, "dataset_statistics.json")
                if os.path.isfile(dataset_statistics_path):
                    with open(dataset_statistics_path, "r") as f:
                        norm_stats = json.load(f)
                    actor_module.norm_stats = norm_stats
                else:
                    print(
                        "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
                        "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
                        "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
                    )
            elif self.config.model.vla == 'vla-adapter':
                from verl_vla.utils.vla_utils.vla_adapter.prismatic.models.action_heads import FlowMatchingActionHead
                from verl_vla.utils.vla_utils.vla_adapter.prismatic.models.projectors import NoisyActionProjector
                from verl_vla.utils.vla_utils.vla_adapter.openvla_utils import update_auto_map, check_model_logic_mismatch, _load_dataset_stats, find_checkpoint_file, load_component_state_dict

                from torch.nn.parallel import DistributedDataParallel as DDP
                
                actor_module = OpenVLAForActionPrediction.from_pretrained(pretrained_model_name_or_path=local_path,
                                                    torch_dtype=torch_dtype,
                                                    attn_implementation='flash_attention_2',
                                                    low_cpu_mem_usage=False,
                                                    config=actor_model_config,  
                                                    trust_remote_code=True)
                actor_module.vision_backbone.set_num_images_in_input(1)
                actor_module.set_version('v1')
                _load_dataset_stats(actor_module, local_path)
                self.raw_state_dice = actor_module.state_dict()
                
                ACTION_DIM = 7
                NUM_FLOW_MATCHING_STEPS = 10
                if unnorm_key == 'bridge_orig':
                    NUM_ACTIONS_CHUNK = 5
                else:
                    NUM_ACTIONS_CHUNK = 20
                fsdp_mesh = self.device_mesh
                device_id = torch.cuda.current_device()

                llm_dim = actor_module.llm_dim
                noisy_action_projector = NoisyActionProjector(
                    llm_dim=llm_dim).to(device=device_id, dtype=torch.bfloat16)
                noisy_action_projector_path = find_checkpoint_file(local_path, "noisy_action_projector")
                noisy_action_projector_state_dict = load_component_state_dict(noisy_action_projector_path)
                noisy_action_projector.load_state_dict(noisy_action_projector_state_dict)
                
                actor_module.noisy_action_projector = noisy_action_projector
                self.noisy_action_projector = actor_module.noisy_action_projector
                # if self._is_ref:
                action_head = FlowMatchingActionHead(
                        input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM, num_flow_steps=NUM_FLOW_MATCHING_STEPS, num_actions=NUM_ACTIONS_CHUNK,
                    ).to(device=device_id, dtype=torch.bfloat16)
                action_head_path = find_checkpoint_file(local_path, "action_head")
                action_head_state_dict = load_component_state_dict(action_head_path)
                action_head.load_state_dict(action_head_state_dict)
                
                actor_module.action_head = action_head
                self.action_head = actor_module.action_head
            elif self.config.model.vla == 'openvla-oft-flow':
                from verl_vla.utils.vla_utils.openvla_oft_flow.prismatic.models.action_heads import FlowMatchingActionHead
                from verl_vla.utils.vla_utils.openvla_oft_flow.prismatic.models.projectors import NoisyActionProjector
                from verl_vla.utils.vla_utils.openvla_oft_flow.openvla_utils import update_auto_map, check_model_logic_mismatch, _load_dataset_stats, find_checkpoint_file, load_component_state_dict

                from torch.nn.parallel import DistributedDataParallel as DDP
                
                actor_module = OpenVLAForActionPrediction.from_pretrained(pretrained_model_name_or_path=local_path,
                                                    torch_dtype=torch_dtype,
                                                    attn_implementation='flash_attention_2',
                                                    low_cpu_mem_usage=False,
                                                    config=actor_model_config,  
                                                    trust_remote_code=True)
                actor_module.vision_backbone.set_num_images_in_input(1)
                _load_dataset_stats(actor_module, local_path)
                self.raw_state_dice = actor_module.state_dict()
                
                ACTION_DIM = 7
                NUM_FLOW_MATCHING_STEPS = 10
                if unnorm_key == 'bridge_orig':
                    NUM_ACTIONS_CHUNK = 5
                else:
                    NUM_ACTIONS_CHUNK = 20
                fsdp_mesh = self.device_mesh
                device_id = torch.cuda.current_device()
                
                actor_module.lm_head = actor_module.language_model.lm_head.to(device_id)

                llm_dim = actor_module.llm_dim
                noisy_action_projector = NoisyActionProjector(
                    llm_dim=llm_dim).to(device=device_id, dtype=torch.bfloat16)
                noisy_action_projector_path = find_checkpoint_file(local_path, "noisy_action_projector")
                noisy_action_projector_state_dict = load_component_state_dict(noisy_action_projector_path)
                noisy_action_projector.load_state_dict(noisy_action_projector_state_dict)
                
                actor_module.noisy_action_projector = noisy_action_projector
                self.noisy_action_projector = actor_module.noisy_action_projector
                # if self._is_ref:
                action_head = FlowMatchingActionHead(
                        input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM, num_flow_steps=NUM_FLOW_MATCHING_STEPS, num_actions=NUM_ACTIONS_CHUNK,
                    ).to(device=device_id, dtype=torch.bfloat16)
                action_head_path = find_checkpoint_file(local_path, "action_head")
                action_head_state_dict = load_component_state_dict(action_head_path)
                action_head.load_state_dict(action_head_state_dict)
                
                actor_module.action_head = action_head
                self.action_head = actor_module.action_head
            elif self.config.model.vla == "openvla":
                actor_module = AutoModelForVision2Seq.from_pretrained(
                                                    pretrained_model_name_or_path=local_path,
                                                    torch_dtype=torch_dtype,
                                                    attn_implementation="flash_attention_2",
                                                    config=actor_model_config,              
                                                    trust_remote_code=True,
                                                )
            elif self.config.model.vla == "pi05":
                actor_module = policy_config.create_trained_policy(actor_model_config, local_path)
                actor_module.n_action_steps = self.config.model.action_chunks_len
                actor_module.noisy_action_projector = None
                actor_module.action_head = None
                self.action_head = None
                self.noisy_action_projector = None
                actor_module.paligemma_with_expert.training = True
                paligemma = actor_module.paligemma_with_expert.paligemma 

                vit = paligemma.model.vision_tower

                for p in vit.parameters():
                    p.requires_grad = False
            # if torch_dtype != torch.float32:

            
            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable()
            # lora add
            if self._is_lora:
                print("Applying LoRA to actor module")
                
                lora_config = {
                    'r': self.config.model.lora_rank,
                    'lora_alpha': self.config.model.lora_alpha,
                    "lora_dropout": 0 ,
                    'target_modules': convert_to_regular_types(self.config.model.target_modules),
                    'init_lora_weights': "gaussian"
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))  
                # for name, param in actor_module.named_parameters():
                #     if "action_queries" in name:
                actor_module.print_trainable_parameters()
            # lora end
            actor_module.to(torch_dtype)
            if self.config.model.vla in ['vla-adapter', 'openvla-oft-flow']:
                self.noisy_action_projector.to(torch_dtype)
                self.action_head.to(torch_dtype) 
                
        torch.distributed.barrier()
        def _find_meta_or_zero_storage(mod):
            bad = []
            for n, p in mod.named_parameters():
                if getattr(p, "is_meta", False):
                    bad.append(("META", n, tuple(p.shape)))
                    continue
                try:  # 2.1+ 才有 untyped_storage
                    if p.numel() > 0 and p.untyped_storage().nbytes() == 0:
                        bad.append(("ZERO", n, tuple(p.shape)))
                except Exception:
                    pass
            return bad
        bad = _find_meta_or_zero_storage(actor_module)
        assert not bad, f"Found meta/zero-storage params before FSDP: {bad[:5]} (total={len(bad)})"
        if self.rank == 0:
            print_model_size(actor_module)
        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)
        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)
        if torch_dtype == torch.float32:
            mixed_precision = None
        if self._is_ref:
            mixed_precision = None
        
        if self.config.model.vla == "pi05":
            auto_wrap_policy, ignored = get_fsdp_wrap_policy_pi05(actor_module, wrap_qkv_linears=True)
        elif self.config.model.vla == "openvla-oft":
            auto_wrap_policy = get_fsdp_wrap_policy_vla(module=actor_module, config=fsdp_config.get('wrap_policy', None), is_lora=self.config.model.get('lora_rank', 0) > 0)
        elif self.config.model.vla == "vla-adapter":
            # if optim_config.params == 'full':
            auto_wrap_policy, ignored = get_fsdp_wrap_policy_vla_adapter(actor_module, is_lora=self.config.model.get('lora_rank', 0) > 0)
        elif self.config.model.vla == "openvla-oft-flow":
            auto_wrap_policy, ignored = get_fsdp_wrap_policy_openvla_oft_flow(actor_module, is_lora=self.config.model.get('lora_rank', 0) > 0)
        else:
            raise ValueError()

        print(f'wrap_policy: {auto_wrap_policy}')

        # # TODO(sgm): support hybrid
        # if auto_wrap_policy is None:
        if self.config.model.vla == 'pi05':
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

            # TODO: add transformer policy
            actor_module_fsdp = FSDP(
                actor_module,
                use_orig_params=True,
                auto_wrap_policy=auto_wrap_policy,
                device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                limit_all_gathers=True,
                device_mesh=self.device_mesh)
        elif self.config.model.vla == 'vla-adapter':
            
            # for name, param in actor_module.named_parameters():
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
            actor_module_fsdp = FSDP(
                actor_module,
                use_orig_params=True,
                auto_wrap_policy=auto_wrap_policy,
                device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                limit_all_gathers=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
                ignored_modules=ignored,)
    
            actor_module_fsdp._fsdp_wrapped_module.action_queries.to(
                torch.cuda.current_device(), dtype=torch_dtype
                )
            bad = []
            for n, p in actor_module_fsdp.named_parameters():
                if p.requires_grad and not hasattr(p, "_fsdp_flattened"):
                    p.requires_grad = False
                    bad.append(n)
            print("trainable but NOT managed by FSDP:", bad)
        elif self.config.model.vla == 'openvla-oft-flow':
            # for name, param in actor_module.named_parameters():
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
            actor_module_fsdp = FSDP(
                actor_module,
                use_orig_params=True,
                auto_wrap_policy=auto_wrap_policy,
                device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                limit_all_gathers=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
                ignored_modules=ignored,)

            bad = []
            for n, p in actor_module_fsdp.named_parameters():
                if p.requires_grad and not hasattr(p, "_fsdp_flattened"):
                    p.requires_grad = False
                    bad.append(n)
            print("trainable but NOT managed by FSDP:", bad)

        log_gpu_memory_usage('After Actor FSDP init', logger=logger)

        # TODO: add more optimizer args into config
        if self._is_actor:
            if self.config.model.vla == 'pi05':
                from verl_vla.utils.torch_functional import get_constant_schedule_with_warmup
                trainable_params = [p for p in actor_module_fsdp.parameters() if p.requires_grad]
                actor_optimizer = optim.AdamW(trainable_params,
                                            lr=optim_config.lr,
                                            betas=optim_config.get('betas', (0.9, 0.999)),
                                            weight_decay=optim_config.get('weight_decay', 1e-2))

                total_steps = optim_config.get('total_training_steps', 0)
                num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

                print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

                actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                    num_warmup_steps=num_warmup_steps)
            elif self.config.model.vla == 'vla-adapter':
                if optim_config.params == 'action_head':
                # if True:
                    from verl_vla.utils.torch_functional import get_constant_schedule_with_warmup
                    def _oget(cfg, key, default):
                        if hasattr(cfg, "get"):
                            try:
                                return cfg.get(key, default)
                            except Exception:
                                pass
                        return getattr(cfg, key, default) if hasattr(cfg, key) else default

                    base_lr = _oget(optim_config, "lr", 1e-4)
                    lora_lr = _oget(optim_config, "lora_lr", base_lr)  # 可单独给 LoRA 设 lr
                    wd      = _oget(optim_config, "weight_decay", 1e-2)
                    betas   = _oget(optim_config, "betas", (0.9, 0.999))

                    if self._is_lora or (optim_config.params == 'full'):
                        lora_params = []
                        for n, p in actor_module_fsdp.named_parameters():
                            # if ('lora' in n) and (p.requires_grad):
                            if p.requires_grad:
                                lora_params.append((n, p))
                    else:
                        lora_params = []
                    
                        
                    # —— Parameter grouping —— #
                    def split_decay(named_params):
                        decay, no_decay = [], []
                        for n, p in named_params:
                            lname = n.lower()
                            if lname.endswith("bias") or ("norm" in lname) or ("ln" in lname) or ("layernorm" in lname):
                                no_decay.append(p)
                            else:
                                decay.append(p)
                        return decay, no_decay

                    # 只在可训练参数上分组
                    head_named = [(n, p) for n, p in (getattr(self.action_head, "named_parameters", lambda: [])()) if p.requires_grad] if self.action_head else []
                    proj_named = [(n, p) for n, p in (getattr(self.noisy_action_projector, "named_parameters", lambda: [])()) if p.requires_grad] if self.noisy_action_projector else []

                    head_decay, head_nodecay = split_decay(head_named)
                    proj_decay, proj_nodecay = split_decay(proj_named)
                    lora_decay, lora_nodecay = split_decay(lora_params)   # 理论上 LoRA 多数也归到 no_decay，这里仍按规则分

                    # ---------- 统计信息 ----------
                    if self.rank == 0:
                        def _count(ps): return sum(p.numel() for p in ps)
                        print(f"# head params (trainable):     {_count(head_decay)+_count(head_nodecay):,}")
                        print(f"# projector params (trainable): {_count(proj_decay)+_count(proj_nodecay):,}")
                        print(f"# lora params (trainable):      {_count(lora_decay)+_count(lora_nodecay):,}")
                        total = _count(head_decay) + _count(head_nodecay) + _count(proj_decay) + _count(proj_nodecay) + _count(lora_decay) + _count(lora_nodecay)
                        print(f"# total trainable params:       {total:,}")

                    # ---------- param groups ----------
                    param_groups = []

                    # LoRA：常见做法是不做 weight decay（两组都设 0.0）
                    if lora_decay:
                        param_groups.append({"params": lora_decay,   "lr": lora_lr, "weight_decay": 0.0})
                    if lora_nodecay:
                        param_groups.append({"params": lora_nodecay, "lr": lora_lr, "weight_decay": 0.0})

                    # head/projector：bias/Norm 不 decay，其余用 wd
                    if head_decay:
                        param_groups.append({"params": head_decay,   "lr": base_lr, "weight_decay": wd})
                    if head_nodecay:
                        param_groups.append({"params": head_nodecay, "lr": base_lr, "weight_decay": 0.0})
                    if proj_decay:
                        param_groups.append({"params": proj_decay,   "lr": base_lr, "weight_decay": wd})
                    if proj_nodecay:
                        param_groups.append({"params": proj_nodecay, "lr": base_lr, "weight_decay": 0.0})

                    # 若没有任何可训练参数，给个提示
                    if len(param_groups) == 0 and self.rank == 0:
                        print("[warn] No trainable parameters found for optimizer groups.")

                    actor_optimizer = optim.AdamW(param_groups, betas=betas)

                    # ---------- scheduler ----------
                    total_steps = _oget(optim_config, "total_training_steps", 0)
                    num_warmup_steps = _oget(optim_config, "lr_warmup_steps", -1)
                    if num_warmup_steps < 0:
                        num_warmup_steps_ratio = _oget(optim_config, "lr_warmup_steps_ratio", 0.0)
                        num_warmup_steps = int(num_warmup_steps_ratio * max(total_steps, 1))

                    if self.rank == 0:
                        print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

                    def warmup_then_const(step: int) -> float:
                        if num_warmup_steps <= 0:
                            return 1.0
                        return min(1.0, float(step) / float(num_warmup_steps))

                    # 给所有 param group 用相同的 lr_lambda（也可以传 list 与组数对应）
                    actor_lr_scheduler = LambdaLR(actor_optimizer, lr_lambda=warmup_then_const)
                elif optim_config.params == 'full':
                    from verl_vla.utils.torch_functional import get_constant_schedule_with_warmup
                    actor_params = [p for p in actor_module_fsdp.parameters() if p.requires_grad]
                    actor_optimizer = optim.AdamW(actor_params,
                                                lr=optim_config.lr,
                                                betas=optim_config.get('betas', (0.9, 0.999)),
                                                weight_decay=optim_config.get('weight_decay', 1e-2))

                    total_steps = optim_config.get('total_training_steps', 0)
                    num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
                    num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

                    print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

                    actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                        num_warmup_steps=num_warmup_steps)
            elif self.config.model.vla == 'openvla-oft-flow':
                if optim_config.params == 'action_head':
                # if True:
                    from verl_vla.utils.torch_functional import get_constant_schedule_with_warmup
                    def _oget(cfg, key, default):
                        if hasattr(cfg, "get"):
                            try:
                                return cfg.get(key, default)
                            except Exception:
                                pass
                        return getattr(cfg, key, default) if hasattr(cfg, key) else default

                    base_lr = _oget(optim_config, "lr", 1e-4)
                    lora_lr = _oget(optim_config, "lora_lr", base_lr)  # 可单独给 LoRA 设 lr
                    wd      = _oget(optim_config, "weight_decay", 1e-2)
                    betas   = _oget(optim_config, "betas", (0.9, 0.999))

                    if self._is_lora or (optim_config.params == 'full'):
                        lora_params = []
                        for n, p in actor_module_fsdp.named_parameters():
                            # if ('lora' in n) and (p.requires_grad):
                            if p.requires_grad:
                                lora_params.append((n, p))
                    else:
                        lora_params = []
                    
                        
                    # —— Parameter grouping —— #
                    def split_decay(named_params):
                        decay, no_decay = [], []
                        for n, p in named_params:
                            lname = n.lower()
                            if lname.endswith("bias") or ("norm" in lname) or ("ln" in lname) or ("layernorm" in lname):
                                no_decay.append(p)
                            else:
                                decay.append(p)
                        return decay, no_decay

                    # 只在可训练参数上分组
                    head_named = [(n, p) for n, p in (getattr(self.action_head, "named_parameters", lambda: [])()) if p.requires_grad] if self.action_head else []
                    proj_named = [(n, p) for n, p in (getattr(self.noisy_action_projector, "named_parameters", lambda: [])()) if p.requires_grad] if self.noisy_action_projector else []

                    head_decay, head_nodecay = split_decay(head_named)
                    proj_decay, proj_nodecay = split_decay(proj_named)
                    lora_decay, lora_nodecay = split_decay(lora_params)   # 理论上 LoRA 多数也归到 no_decay，这里仍按规则分

                    # ---------- 统计信息 ----------
                    if self.rank == 0:
                        def _count(ps): return sum(p.numel() for p in ps)
                        print(f"# head params (trainable):     {_count(head_decay)+_count(head_nodecay):,}")
                        print(f"# projector params (trainable): {_count(proj_decay)+_count(proj_nodecay):,}")
                        print(f"# lora params (trainable):      {_count(lora_decay)+_count(lora_nodecay):,}")
                        total = _count(head_decay) + _count(head_nodecay) + _count(proj_decay) + _count(proj_nodecay) + _count(lora_decay) + _count(lora_nodecay)
                        print(f"# total trainable params:       {total:,}")

                    # ---------- param groups ----------
                    param_groups = []

                    # LoRA：常见做法是不做 weight decay（两组都设 0.0）
                    if lora_decay:
                        param_groups.append({"params": lora_decay,   "lr": lora_lr, "weight_decay": 0.0})
                    if lora_nodecay:
                        param_groups.append({"params": lora_nodecay, "lr": lora_lr, "weight_decay": 0.0})

                    # head/projector：bias/Norm 不 decay，其余用 wd
                    if head_decay:
                        param_groups.append({"params": head_decay,   "lr": base_lr, "weight_decay": wd})
                    if head_nodecay:
                        param_groups.append({"params": head_nodecay, "lr": base_lr, "weight_decay": 0.0})
                    if proj_decay:
                        param_groups.append({"params": proj_decay,   "lr": base_lr, "weight_decay": wd})
                    if proj_nodecay:
                        param_groups.append({"params": proj_nodecay, "lr": base_lr, "weight_decay": 0.0})

                    # 若没有任何可训练参数，给个提示
                    if len(param_groups) == 0 and self.rank == 0:
                        print("[warn] No trainable parameters found for optimizer groups.")

                    actor_optimizer = optim.AdamW(param_groups, betas=betas)

                    # ---------- scheduler ----------
                    total_steps = _oget(optim_config, "total_training_steps", 0)
                    num_warmup_steps = _oget(optim_config, "lr_warmup_steps", -1)
                    if num_warmup_steps < 0:
                        num_warmup_steps_ratio = _oget(optim_config, "lr_warmup_steps_ratio", 0.0)
                        num_warmup_steps = int(num_warmup_steps_ratio * max(total_steps, 1))

                    if self.rank == 0:
                        print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

                    def warmup_then_const(step: int) -> float:
                        if num_warmup_steps <= 0:
                            return 1.0
                        return min(1.0, float(step) / float(num_warmup_steps))

                    # 给所有 param group 用相同的 lr_lambda（也可以传 list 与组数对应）
                    actor_lr_scheduler = LambdaLR(actor_optimizer, lr_lambda=warmup_then_const)
                elif optim_config.params == 'full':
                    from verl_vla.utils.torch_functional import get_constant_schedule_with_warmup
                    actor_params = [p for p in actor_module_fsdp.parameters() if p.requires_grad]
                    actor_optimizer = optim.AdamW(actor_params,
                                                lr=optim_config.lr,
                                                betas=optim_config.get('betas', (0.9, 0.999)),
                                                weight_decay=optim_config.get('weight_decay', 1e-2))

                    total_steps = optim_config.get('total_training_steps', 0)
                    num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
                    num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

                    print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

                    actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                        num_warmup_steps=num_warmup_steps)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)
        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config
    
    
    def _build_model_optimizer_wm(self,
                               model_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False):
        from verl_vla.utils.model import print_model_size, update_model_config
        from verl_vla.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, \
            CPUOffload
        from torch import optim
        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)

        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)
        
        # Initialize the optional world model on CPU before FSDP wrapping.
        from verl_vla.utils.wm_utils import CosMosWorldModel
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch

        wm_config = self.config.world_model
        print("--- [FSDP Worker] Initializing CosMosWorldModel ---")
        wm_module = CosMosWorldModel(wm_config)
        wm_module.to(torch_dtype)

        if self._is_lora:
            raise
                
                
        torch.distributed.barrier()
        def _find_meta_or_zero_storage(mod):
            bad = []
            for n, p in mod.named_parameters():
                if getattr(p, "is_meta", False):
                    bad.append(("META", n, tuple(p.shape)))
                    continue
                try:  # 2.1+ 才有 untyped_storage
                    if p.numel() > 0 and p.untyped_storage().nbytes() == 0:
                        bad.append(("ZERO", n, tuple(p.shape)))
                except Exception:
                    pass
            return bad
        bad = _find_meta_or_zero_storage(wm_module)
        assert not bad, f"Found meta/zero-storage params before FSDP: {bad[:5]} (total={len(bad)})"

        if self.rank == 0:
            print_model_size(wm_module)
        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)
        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        if self._is_ref:
            mixed_precision = None

        auto_wrap_policy = get_fsdp_wrap_policy_wm(wm_module)

        print(f'wrap_policy: {auto_wrap_policy}')

        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        # TODO: add transformer policy
        wm_module_fsdp = FSDP(
            wm_module,
            use_orig_params=True,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            limit_all_gathers=True,
            device_mesh=self.device_mesh)

        log_gpu_memory_usage('After Actor FSDP init', logger=logger)

        wm_optimizer = None
        wm_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)
        return wm_module_fsdp
    
    
    def _build_model_optimizer_rm(self,
                               model_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False):
        from verl_vla.utils.model import print_model_size, update_model_config
        from verl_vla.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, \
            CPUOffload
        from torch import optim
        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)

        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)
        
        from verl_vla.utils.rm_utils.evo_vlac import GAC_model_client
        rm_module = GAC_model_client(tag='critic')
        rm_module.init_model(model_path=model_path, model_type='internvl2', use_server=True, device_map=f'cuda:0')
        rm_module.temperature=0.5
        rm_module.top_k=1
        rm_module.set_config()
        rm_module.set_system_prompt()
        try:
            rm_gateway = ray.get_actor("rm_gateway")
        except ValueError:
            # 可选：如果你在 rank0 创建，则这里获取前先 barrier 等待
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
            rm_gateway = ray.get_actor("rm_gateway")
        rm_module.bind_gateway(rm_gateway)
        
        if self._is_lora:
            raise
                
                
        torch.distributed.barrier()

        return rm_module
    

    def _build_rollout(self):
        if self.config.rollout.name == 'hf':
            from verl_vla.workers.rollout import RobHFRollout
            from verl_vla.workers.hybrid_engine import BaseShardingManager
            rollout = RobHFRollout(
                module=self.actor_module_fsdp, 
                action_head=self.action_head,
                noisy_action_projector=self.noisy_action_projector,
                config=self.config.rollout)
            sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == 'vllm':
            raise ValueError
            # rollout = vLLMRollout(actor_module=self.actor_module_fsdp,
            #                       model_hf_config=self.actor_model_config)
            # if torch.distributed.get_world_size() == 1:
            # sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
            #                                            full_params='hf' in self.config.rollout.load_format)

        return rollout, sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl_vla.workers.actor import RobDataParallelPPOActor
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        unnorm_key = self.config.rollout.unnorm_key
        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                meta_path=self.config.model.meta_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                trust_remote_code=True,
                unnorm_key=unnorm_key,
                ) #self.config.model.get('trust_remote_code', True)
            # if self._is_rollout:
            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module
            

            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage('After offload actor grad during init', logger=logger)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            self.actor = RobDataParallelPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_fsdp,
                                              action_head=self.action_head,
                                              noisy_action_projector=self.noisy_action_projector,
                                              actor_optimizer=self.actor_optimizer,
                                              processor_path=self.config.rollout.pretrained_checkpoint)
            if self.config.model.vla == 'pi05':
                self.checkpoint_manager = FSDPVLACheckpointManager(
                    model=self.actor_module_fsdp,
                    optimizer=self.actor_optimizer,
                    lr_scheduler=self.actor_lr_scheduler,
                    processing_class=None,
                    checkpoint_config=None,
                )
            elif self.config.model.vla in ['vla-adapter', 'openvla-oft-flow']:
                # self.checkpoint_manager = FSDPCheckpointManager_w_lora_extra_model(
                self.checkpoint_manager = FSDPVLACheckpointManager(
                    model=self.actor_module_fsdp,
                    action_head=self.action_head,
                    noisy_action_projector=self.noisy_action_projector,
                    optimizer=self.actor.actor_optimizer,
                    lr_scheduler=self.actor_lr_scheduler,
                    processing_class=self.tokenizer,
                    checkpoint_contents=None,)
                    # checkpoint_contents=['model', 'optimizer', 'extra', 'adapter'] )
            if self.config.model.load_ckpt != "":
                self.checkpoint_manager.load_checkpoint(local_path=self.config.model.load_ckpt)
                if self.config.model.merge_save_ckpt_only:
                    from pathlib import Path
                    rank = torch.distributed.get_rank()
                    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(self.actor_module_fsdp, StateDictType.FULL_STATE_DICT, full_cfg):
                        full_sd = self.actor_module_fsdp.state_dict()
                    if rank == 0:
                        out = Path(self.config.model.merged_out)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(full_sd, out)
                        print(f"[rank0] merged model state_dict saved to: {out} (keys={len(full_sd)})")
                    
                    torch.distributed.barrier()
                    print("finish merge!")
                    raise
                    try:
                        if torch.distributed.is_initialized():
                            torch.distributed.destroy_process_group()
                    except Exception:
                        pass

                    
                
            # self.checkpoint_manager.save_checkpoint(
            #     local_path='debug', hdfs_path=None, global_step=0, max_ckpt_to_keep=None

        if self._is_rollout:
            self.rollout, self.sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.model.path,
                                                               meta_path=self.config.model.meta_path,
                                                               fsdp_config=self.config.ref.fsdp_config,
                                                               optim_config=None,
                                                               override_model_config=override_model_config,
                                                               trust_remote_code=True,
                                                               unnorm_key=unnorm_key)[0] #self.config.model.get('trust_remote_code', False)
                                                                   
            if self._is_offload_param:
                offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)

            OmegaConf.set_struct(self.config.ref, True)
            self.ref_policy = RobDataParallelPPOActor(
                config=self.config.ref, 
                actor_module=self.ref_module_fsdp,
                action_head=self.action_head,
                noisy_action_projector=self.noisy_action_projector,
                processor_path=self.config.rollout.pretrained_checkpoint
            )
        
        if self._is_rollout:
            if self.use_world_model:
                self.world_model_fsdp = self._build_model_optimizer_wm(model_path=self.config.world_model.dit_path,
                                                                fsdp_config=self.config.world_model.fsdp_config,
                                                                optim_config=None,
                                                                override_model_config=override_model_config,
                                                                trust_remote_code=True) #self.config.model.get('trust_remote_code', False)
                self.world_model = self.world_model_fsdp
                self.rollout.world_model = self.world_model
                self.rollout.use_world_model = True
            else:
                self.rollout.use_world_model = False
            self.rollout.env_rollout = self.env_rollout

            # if self.use_reward_model:
            #     self.reward_model = self._build_model_optimizer_rm(model_path=self.config.reward_model.model_path,
            #                                                     trust_remote_code=True) #self.config.model.get('trust_remote_code', False)
        
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())


        log_gpu_memory_usage('Before update policy', logger=logger)
        if self.actor.config.vla == 'pi05':
            metrics = self.actor.update_policy_pi05(data=data)
        elif self.actor.config.vla == 'vla-adapter':
            metrics = self.actor.update_policy_vla_adapter(data=data)
        elif self.actor.config.vla == 'openvla-oft-flow':
            metrics = self.actor.update_policy_openvla_oft_flow(data=data)
        else:
            metrics = self.actor.update_policy(data=data)

        self.actor_lr_scheduler.step()
        lr = self.actor_lr_scheduler.get_last_lr()[0]
        metrics['actor/lr(1e-4)'] = lr * 1e4

        log_gpu_memory_usage('After update policy', logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={'metrics': metrics})
        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_entropy(self, data: DataProto):
        
        data = data.to('cuda')

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before compute entropy', logger=logger)

        metrics = self.actor.compute_entropy(bacth_data=data)

        log_gpu_memory_usage('After compute entropy', logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={'metrics': metrics})
        output = output.to('cpu')
        
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts):
        prompts = prompts.to('cuda')
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        prompts.batch = prompts.batch.cuda()
        if self.tokenizer is None:
            meta_info = {'eos_token_id': -1, 'pad_token_id': -1}
        else:
            meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        prompts.meta_info.update(meta_info)
        
        # with Timer(name=f'gen seq will start, and the num samples are: {tmp_sample}', text="{name}: {seconds:.1f} seconds") as timer:    
        with self.sharding_manager:
            log_gpu_memory_usage('After entering sharding manager', logger=logger)    
            prompts = self.sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)
            log_gpu_memory_usage('After rollout generation', logger=logger)

            output = self.sharding_manager.postprocess_data(output)
            torch.cuda.synchronize()
            
        output = output.to('cpu')
        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        data = data.to('cuda')
        # we should always recompute old_log_probs when it is HybridEngine

        data.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info['pad_token_id'] = self.tokenizer.pad_token_id if self.tokenizer is not None else -1
        old_log_probs, _, _, out_metric = self.actor.compute_log_prob(data=data)
        output = DataProto.from_dict(
            tensors={
                'old_log_probs': old_log_probs,
                'old_logp_elem': out_metric['logp_elem'],
                })
        return output
            
        
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.ref_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
            
        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info['pad_token_id'] = self.tokenizer.pad_token_id
        output = self.ref_policy.compute_log_prob(data=data)
        output = DataProto.from_dict(tensors={'ref_log_prob': output})

        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob_pi05(self, data: DataProto):
        assert self._is_ref

        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.ref_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info['pad_token_id'] = self.tokenizer.pad_token_id if self.tokenizer is not None else -1
        output = self.ref_policy.compute_log_prob(data=data)
        if len(output) == 4:
            output = output[0]
        output = DataProto.from_dict(tensors={'ref_log_prob': output})
        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output
    
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob_vla_adapter(self, data: DataProto):
        assert self._is_ref

        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.ref_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info['pad_token_id'] = self.tokenizer.pad_token_id if self.tokenizer is not None else -1
        output = self.ref_policy.compute_log_prob(data=data)
        if len(output) == 4:
            output = output[0]
        output = DataProto.from_dict(tensors={'ref_log_prob': output})
        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob_openvla_oft_flow(self, data: DataProto):
        assert self._is_ref

        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.ref_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info['pad_token_id'] = self.tokenizer.pad_token_id if self.tokenizer is not None else -1
        output = self.ref_policy.compute_log_prob(data=data)
        if len(output) == 4:
            output = output[0]
        output = DataProto.from_dict(tensors={'ref_log_prob': output})
        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, blocking=True):
        assert self._is_actor
        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # build device mesh
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            self._is_offload_grad = self.config.actor.fsdp_config.get('grad_offload', False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get('param_offload', False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.shape[0]
            self.config.actor.ppo_micro_batch_size //= self.device_mesh.shape[0]
        if self._is_rollout:
            self.config.rollout.log_prob_micro_batch_size //= self.device_mesh.shape[0]
        if self._is_ref:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.shape[0]

    def _build_model_optimizer(self,
                               model_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False):
        from verl_vla.utils.model import print_model_size, update_model_config
        from verl_vla.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, \
            CPUOffload
        from torch import optim

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        local_path = copy_local_path_from_hdfs(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        if self.config.model.use_remove_padding:
            from verl_vla.models.registry import check_model_support_rmpad
            check_model_support_rmpad(actor_model_config.model_type)
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f'Model config after override: {actor_model_config}')

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            actor_module = AutoLigerKernelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                torch_dtype=torch_dtype,
                                                                config=actor_model_config,
                                                                attn_implementation='flash_attention_2',
                                                                trust_remote_code=trust_remote_code)
            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable()
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        if self._is_ref:
            mixed_precision = None

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))

        if self._is_rollout and self.config.rollout.name == 'hf':
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f'wrap_policy: {auto_wrap_policy}')

        # TODO(sgm): support hybrid
        if auto_wrap_policy is None:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD

        # TODO: add transformer policy
        actor_module_fsdp = FSDP(
            actor_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh)

        log_gpu_memory_usage('After Actor FSDP init', logger=logger)

        # TODO: add more optimizer args into config
        if self._is_actor:
            from verl_vla.utils.torch_functional import get_constant_schedule_with_warmup
            actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),
                                          lr=optim_config.lr,
                                          betas=optim_config.get('betas', (0.9, 0.999)),
                                          weight_decay=optim_config.get('weight_decay', 1e-2))

            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   num_warmup_steps=num_warmup_steps)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self):
        if self.config.rollout.name == 'hf':
            from verl_vla.workers.rollout import HFRollout
            from verl_vla.workers.hybrid_engine import BaseShardingManager
            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == 'vllm':
            from verl_vla.workers.rollout.vllm_rollout import vLLMRollout
            from verl_vla.workers.hybrid_engine import FSDPVLLMShardingManager
            log_gpu_memory_usage('Before building vllm rollout', logger=None)
            rollout = vLLMRollout(actor_module=self.actor_module_fsdp,
                                  config=self.config.rollout,
                                  tokenizer=self.tokenizer,
                                  model_hf_config=self.actor_model_config)
            log_gpu_memory_usage('After building vllm rollout', logger=None)
            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = 'dummy_hf'
            sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
                                                       inference_engine=rollout.inference_engine,
                                                       model_config=self.actor_model_config,
                                                       full_params='hf' in self.config.rollout.load_format)
            log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl_vla.workers.actor import DataParallelPPOActor
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                trust_remote_code=self.config.model.get('trust_remote_code', False))

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage('After offload actor grad during init', logger=logger)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            self.actor = DataParallelPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_fsdp,
                                              actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.model.path,
                                                               fsdp_config=self.config.ref.fsdp_config,
                                                               optim_config=None,
                                                               override_model_config=override_model_config,
                                                               trust_remote_code=self.config.model.get(
                                                                   'trust_remote_code', False))[0]
            if self._is_offload_param:
                offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)

            OmegaConf.set_struct(self.config.ref, True)
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        data = data.to('cuda')

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update policy', logger=logger)

        metrics = self.actor.update_policy(data=data)

        self.actor_lr_scheduler.step()
        lr = self.actor_lr_scheduler.get_last_lr()[0]
        metrics['actor/lr(1e-4)'] = lr * 1e4

        log_gpu_memory_usage('After update policy', logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={'metrics': metrics})
        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_entropy(self, data: DataProto):
        
        data = data.to('cuda')

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before compute entropy', logger=logger)

        metrics = self.actor.compute_entropy(bacth_data=data)

        log_gpu_memory_usage('After compute entropy', logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={'metrics': metrics})
        output = output.to('cpu')
        
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to('cuda')
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        prompts.batch = prompts.batch.cuda()
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        prompts.meta_info.update(meta_info)
        with self.sharding_manager:
            log_gpu_memory_usage('After entering sharding manager', logger=logger)
            prompts = self.sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage('After rollout generation', logger=logger)

            output = self.sharding_manager.postprocess_data(output)
            torch.cuda.synchronize()

        if self._is_actor and recompute_log_prob:
            # we should always recompute old_log_probs when it is HybridEngine
            output.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size
            output.meta_info['temperature'] = self.config.rollout.temperature
            output.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
            output.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
            old_log_probs = self.actor.compute_log_prob(data=output)
            output.batch['old_log_probs'] = old_log_probs

        output = output.to('cpu')

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.ref_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        output = self.ref_policy.compute_log_prob(data=data)
        output = DataProto.from_dict(tensors={'ref_log_prob': output})

        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.actor.actor_module, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.actor.actor_module.state_dict()
        if self.rank == 0:
            print(f'Saving actor checkpoint to {local_path}')
            os.makedirs(local_path, exist_ok=True)
            self.actor_module.save_pretrained(local_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(local_path)
            if hdfs_path is not None:
                print(f'Uploading actor checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)






# def dummy_reset_parameters(self):
#     pass


class CriticWorker(Worker):

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_grad = self.config.model.fsdp_config.grad_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size()
        self.config.ppo_micro_batch_size //= torch.distributed.get_world_size()

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from verl_vla.utils.model import LambdaLayer, print_model_size, squeeze
        from verl_vla.utils.torch_dtypes import PrecisionType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, \
            CPUOffload
        from torch import optim

        local_path = copy_local_path_from_hdfs(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.
        # TODO: support loading critic weights from RM. Support using AutoModelForTokenClassification
        from transformers import AutoTokenizer

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get('trust_remote_code', False))

        from omegaconf import OmegaConf
        override_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f'Critic overriding config {override_config_kwargs}')

        torch_dtype = self.config.model.fsdp_config.get('model_dtype', 'fp32')
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForCausalLM
        from torch import nn

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        init_context = get_init_weight_context_manager()
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                 torch_dtype=torch_dtype,
                                                                 config=critic_model_config,
                                                                 attn_implementation='flash_attention_2',
                                                                 trust_remote_code=trust_remote_code)
            critic_module.lm_head = nn.Sequential(nn.Linear(critic_model_config.hidden_size, 1, dtype=torch_dtype),
                                                  LambdaLayer(fn=squeeze))

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get('enable_gradient_checkpointing', False):
                critic_module.gradient_checkpointing_enable()
        if self.rank == 0:
            print_model_size(critic_module)

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy)

        log_gpu_memory_usage('Before critic FSDP', logger=None)

        critic_module = FSDP(critic_module,
                             param_init_fn=init_fn,
                             use_orig_params=False,
                             auto_wrap_policy=auto_wrap_policy,
                             device_id=torch.cuda.current_device(),
                             sharding_strategy=ShardingStrategy.FULL_SHARD,
                             mixed_precision=mixed_precision,
                             sync_module_states=True)

        log_gpu_memory_usage('After critic FSDP', logger=None)

        critic_optimizer = optim.AdamW(critic_module.parameters(),
                                       lr=config.optim.lr,
                                       betas=config.optim.get('betas', (0.9, 0.999)),
                                       weight_decay=config.optim.get('weight_decay', 1e-2))

        total_steps = config.optim.get('total_training_steps', 0)
        num_warmup_steps_ratio = config.optim.get('lr_warmup_steps_ratio', 0.)
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

        from verl_vla.utils.torch_functional import get_constant_schedule_with_warmup
        critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer,
                                                                num_warmup_steps=num_warmup_steps)

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from verl_vla.workers.critic import DataParallelPPOCritic
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        self.critic = DataParallelPPOCritic(config=self.config,
                                            critic_module=self.critic_module,
                                            critic_optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        micro_batch_size = self.config.ppo_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['max_token_len'] = self.config.forward_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
        values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={'values': values})
        output = output.to('cpu')
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to('cuda')
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=torch.cuda.current_device())
        metrics = self.critic.update_critic(data=data)

        self.critic_lr_scheduler.step()
        lr = self.critic_lr_scheduler.get_last_lr()[0]
        metrics['critic/lr(1e-4)'] = lr * 1e4

        output = DataProto(batch=None, meta_info={'metrics': metrics})
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()
        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.critic_module, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.critic_module.state_dict()
        if self.rank == 0:
            print(f'Saving critic checkpoint to {local_path}')
            os.makedirs(local_path, exist_ok=True)
            self.critic_module._fsdp_wrapped_module.save_pretrained(local_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(local_path)
            if hdfs_path is not None:
                print(f'Uploading critic checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)

class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForSequenceClassification.
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        self.config.micro_batch_size //= torch.distributed.get_world_size()

    def _build_model(self, config):
        # the following line is necessary
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.model.path)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_local_path_from_hdfs(config.model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path,
                                                trust_remote_code=config.model.get('trust_remote_code', False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get('trust_remote_code', False))

        trust_remote_code = config.model.get('trust_remote_code', False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward_module = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                               torch_dtype=torch.bfloat16,
                                                                               attn_implementation='flash_attention_2',
                                                                               trust_remote_code=trust_remote_code)
            reward_module.to(torch.bfloat16)
        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # zero3
            sync_module_states=True,
            cpu_offload=CPUOffload(offload_params=self.config.model.fsdp_config.param_offload))

        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        self.reward_module = self._build_model(config=self.config)
        torch.cuda.empty_cache()

    def _forward_micro_batch(self, micro_batch):
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.reward_module(input_ids=micro_batch['input_ids'],
                                        attention_mask=micro_batch['attention_mask'],
                                        position_ids=micro_batch['position_ids'])
            rm_score = output.logits  # (batch_size,)
            rm_score = rm_score.squeeze(-1)
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']
        response_length = data.batch['responses'].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch['attention_mask'].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            chat: list = data.non_tensor_batch['raw_prompt'][i].tolist()

            # extract response
            response_ids = data.batch['responses'][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch['attention_mask'][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, '')

            chat.append({'role': 'assistant', 'content': response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat,
                                                                             add_generation_prompt=False,
                                                                             tokenize=False)
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f'Switch template. chat: {prompt_with_chat_template}')

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get('max_length', src_max_length)
            if max_length is None:
                max_length = src_max_length
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=target_tokenizer,
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get('truncation', 'right'))  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {'input_ids': rm_input_ids, 'attention_mask': rm_attention_mask, 'position_ids': rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        data = data.to('cuda')
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)

        rm_data.batch = rm_data.batch.cuda()
        micro_batches = rm_data.batch.split(self.config.micro_batch_size)
        output = []
        for micro_batch in micro_batches:
            rm_score = self._forward_micro_batch(micro_batch)
            output.append(rm_score)
        scores = torch.cat(output, dim=0)  # (batch_size)
        token_level_scores = self._expand_to_token_level(data, scores)
        # Note that this is only the scores, may not be the final rewards used to train RL
        output = DataProto.from_dict(tensors={'rm_scores': token_level_scores})
        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output

class PRIMERewardModelWorker(Worker):
    """
    PRIME reward model.
    Can update itself whenever compute_rm_score is called.
    """
    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        world_size = torch.distributed.get_world_size()
        self.config.mini_batch_size //= world_size
        self.config.micro_batch_size //= world_size
        # build device mesh
        
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

        self._is_offload_param = self.config.prime_model.fsdp_config.get('param_offload', False)
        self._is_offload_grad = self.config.prime_model.fsdp_config.get('grad_offload', False)
        self._is_offload_optimizer = self.config.prime_model.fsdp_config.get('optimizer_offload', False)

    def _build_model_optimizer(self, config, enable_gradient_checkpointing=False):
        # the following line is necessary
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload

        # download the checkpoint from hdfs
        prime_model_path = config.prime_model.get("path")
        if not prime_model_path:
            raise ValueError(
                "reward_model.prime_model.path is required when the PRIME reward model is enabled; "
                "set PRIME_MODEL_PATH or provide a Hydra override."
            )
        prime_ref_path = None
        if config.prime_model.get("ref_type", "freeze") == "freeze":
            prime_ref_path = config.prime_model.get("ref_path")
            if not prime_ref_path:
                raise ValueError(
                    "reward_model.prime_model.ref_path is required when ref_type=freeze; "
                    "set PRIME_REF_PATH or provide a Hydra override."
                )
        local_path = copy_local_path_from_hdfs(prime_model_path)

        if self.config.prime_model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_local_path_from_hdfs(config.prime_model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path,
                                                trust_remote_code=config.prime_model.get('trust_remote_code', False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.prime_model.get('trust_remote_code', False))

        trust_remote_code = config.prime_model.get('trust_remote_code', False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        if config.prime_model.use_remove_padding:
            from verl_vla.models.registry import check_model_support_rmpad
            check_model_support_rmpad(model_config.model_type)
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            reward_module = AutoLigerKernelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                               torch_dtype=torch.float32,
                                                                               attn_implementation='flash_attention_2',
                                                                               trust_remote_code=trust_remote_code)
            reward_module.to(torch.float32)
            if enable_gradient_checkpointing:
                reward_module.gradient_checkpointing_enable()
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                         buffer_dtype=torch.float32)
        if config.prime_model.get('enable_gradient_checkpointing', False):
            reward_module.gradient_checkpointing_enable()

        if config.prime_model.get("ref_type", 'freeze') == 'freeze':
            reference_module = AutoLigerKernelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=copy_local_path_from_hdfs(prime_ref_path),
                torch_dtype=torch.bfloat16,
                attn_implementation='flash_attention_2',
                trust_remote_code=trust_remote_code)
            reference_module.to(torch.bfloat16)
            for param in reference_module.parameters():
                param.requires_grad = False
        else:
            reference_module = None

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.prime_model.fsdp_config)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # zero3
            mixed_precision=mixed_precision,
            device_mesh=self.device_mesh,
            sync_module_states=True)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reference_module, config=self.config.prime_model.fsdp_config)
        if reference_module is not None:
            reference_module = FSDP(
                reference_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=torch.cuda.current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,  # zero3
                device_mesh=self.device_mesh,
                sync_module_states=True)

        self.update_dpo_type = self.config.prime_model.get('update', 'none')
        if self.update_dpo_type in ['before', 'after']:

            from torch import optim
            self.reward_optimizer = optim.AdamW(reward_module.parameters(),
                                                lr=config.prime_model.optim.lr,
                                                betas=config.prime_model.optim.get('betas', (0.9, 0.999)),
                                                weight_decay=config.prime_model.optim.get('weight_decay', 1e-2))

            total_steps = config.prime_model.optim.get('total_training_steps', 0)
            num_warmup_steps_ratio = config.prime_model.optim.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            from verl_vla.utils.torch_functional import get_constant_schedule_with_warmup
            self.reward_lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.reward_optimizer,
                                                                         num_warmup_steps=num_warmup_steps)

            # fsdp offload configurations
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.reward_optimizer)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=reward_module, offload_grad=self._is_offload_grad)
            if reference_module is not None:
                offload_fsdp_param_and_grad(module=reference_module, offload_grad=self._is_offload_grad)

        return reward_module, reference_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl_vla.workers.actor import DataParallelPRIME
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.prime_model.get('external_lib', None))
        self.reward_module, self.reference_module = self._build_model_optimizer(config=self.config, enable_gradient_checkpointing=self.config.prime_model.get('enable_gradient_checkpointing', False))
        self.prm = DataParallelPRIME(config=self.config,
                                    reward_module=self.reward_module,
                                    reference_module=self.reference_module,
                                    reward_optimizer=self.reward_optimizer,
                                    prime_loss_fn=self.config.prime_model.get('loss_type', 'ce'))
        torch.cuda.empty_cache()

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch['attention_mask'].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            chat: list = data.non_tensor_batch['raw_prompt'][i].tolist()

            # extract response
            response_ids = data.batch['responses'][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch['attention_mask'][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, '')

            chat.append({'role': 'assistant', 'content': response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat,
                                                                             add_generation_prompt=False,
                                                                             tokenize=False)
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f'Switch template. chat: {prompt_with_chat_template}')

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get('max_length', src_max_length)
            if max_length is None:
                max_length = src_max_length
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=target_tokenizer,
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get('truncation', 'right'))  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {'input_ids': rm_input_ids, 'attention_mask': rm_attention_mask, 'position_ids': rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        n_samples=data.meta_info['n_samples']
        beta=self.config.prime_model.get('beta_train', 0.05)
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_data=data

        if self.update_dpo_type!='none':
            if self._is_offload_optimizer:
                load_fsdp_optimizer(optimizer=self.reward_optimizer, device_id=torch.cuda.current_device())
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.reward_module,device_id=torch.cuda.current_device(),load_grad=self._is_offload_grad)
            if self.reference_module is not None:
                load_fsdp_param_and_grad(module=self.reference_module,device_id=torch.cuda.current_device(),load_grad=self._is_offload_grad)
        
        token_level_scores, metrics = self.prm.update_policy(rm_data)

        output=DataProto.from_dict(tensors = {'rm_scores': token_level_scores}, meta_info = {'metrics': metrics})

        if self.update_dpo_type != 'none':
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.reward_optimizer)
            self.reward_lr_scheduler.step()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.reward_module, offload_grad=self._is_offload_grad)
            if self.reference_module is not None:
                offload_fsdp_param_and_grad(module=self.reference_module, offload_grad=self._is_offload_grad)

        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.reward_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.reward_module, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.reward_module.state_dict()
        if self.rank == 0:
            print(f'Saving reward checkpoint to {local_path}')
            os.makedirs(local_path, exist_ok=True)
            self.reward_module._fsdp_wrapped_module.save_pretrained(local_path, state_dict=state_dict)
            if hdfs_path is not None:
                print(f'Uploading reward checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.reward_module, offload_grad=self._is_offload_grad)
