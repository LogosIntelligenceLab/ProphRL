import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
import jax
import numpy as np

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
import openpi.models.model as _model

def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.bfloat16,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.bfloat16, device=device)
        

    @torch.no_grad()
    def predict_action_chunk(
        self, 
        batch: dict[str, Tensor], 
        noise: Tensor | None = None, 
        use_sde: bool = False, 
        return_logprob: bool = False,
        recompute_log_prob: bool = False
    ) -> Tensor:
        # Make a copy since transformations may modify the inputs in place.
        sample_rng_or_pytorch_device = batch['observation/image_is_pad'].device
        img_np = batch['observation/image'].cpu().numpy()
        batch['observation/image_is_pad'] = batch['observation/image_is_pad'].cpu().float()
        
        bs = batch['observation/image'].shape[0]
        if batch.get("prompt", None) is None:
            prompt = 'pad'
        else:
            prompt = batch['prompt']
            
        image = {'observation/image': img_np,
                 'observation/state': np.zeros([bs, 8]),
                 'prompt': prompt}
        inputs = self._input_transform(image)
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(sample_rng_or_pytorch_device, dtype=torch.bfloat16), inputs)
        
        if recompute_log_prob:
            logp_input = batch
        else:
            logp_input = None
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            actions_pred, lang_tokens, lang_masks, return_dict = self._get_action_chunk(
                inputs,
                noise,
                use_sde=use_sde,
                return_logprob=return_logprob,
                recompute_log_prob=recompute_log_prob,
                logp_input=logp_input,
            )



        if actions_pred is not None:
            actions = actions_pred[:, :self.n_action_steps]
        else:
            actions = None
        return actions, lang_tokens, lang_masks, return_dict
    
    def predict_action_chunk_update(
        self, 
        batch: dict[str, Tensor], 
        noise: Tensor | None = None, 
        use_sde: bool = False, 
        return_logprob: bool = False,
        recompute_log_prob: bool = False
    ) -> Tensor:
        # Make a copy since transformations may modify the inputs in place.
        if not recompute_log_prob:
            self.eval()
        sample_rng_or_pytorch_device = batch['observation/image_is_pad'].device
        img_np = batch['observation/image'].cpu().numpy()
        batch['observation/image_is_pad'] = batch['observation/image_is_pad'].cpu().float()
        
        bs = batch['observation/image'].shape[0]
        if batch.get("prompt", None) is None:
            prompt = 'pad'
        else:
            prompt = batch['prompt']
            
        image = {'observation/image': img_np,
                 'observation/state': np.zeros([bs, 8]),
                 'prompt': prompt}
        inputs = self._input_transform(image)
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(sample_rng_or_pytorch_device, dtype=torch.bfloat16), inputs)
        
        if recompute_log_prob:
            logp_input = batch
        else:
            logp_input = None
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            actions_pred, lang_tokens, lang_masks, return_dict = self._get_action_chunk(
                inputs,
                noise,
                use_sde=use_sde,
                return_logprob=return_logprob,
                recompute_log_prob=recompute_log_prob,
                logp_input=logp_input,
            )



        if actions_pred is not None:
            actions = actions_pred[:, :self.n_action_steps]
        else:
            actions = None
        return actions, lang_tokens, lang_masks, return_dict

    def _get_action_chunk(
        self, 
        batch: dict[str, Tensor], 
        noise: Tensor | None = None, 
        use_sde: bool = False, 
        return_logprob: bool = False,
        recompute_log_prob: bool = False,
        logp_input = None,
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        # for k in batch:
        #     if k in self._queues and k != ACTION:
                
        # if batch.get("lang_tokens", None) is None:
            
        
        observation = _model.Observation.from_dict(batch)
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)


            
            
        bsize = lang_tokens.shape[0]
        if recompute_log_prob:
            x_t = logp_input['x_t']
            t = logp_input['t']
            x_next = logp_input['x_next']
            finish_step = logp_input['finish_step']
            images = images[0]
            img_masks = img_masks[0]
            B, S, _, H, W = images.shape
            lang_tokens, lang_masks = logp_input["lang_tokens"], logp_input["lang_masks"]
            img_masks = img_masks.unsqueeze(-1).unsqueeze(-1).repeat(1, S, 1)
            (logp_action,
             logp_step, 
             logp_outer, 
             logp_joint, 
             logp_elem,
             ent_step, 
             ent_outer, 
             ent_joint,
             mean,
             std_dev_t,
             out_metric,
             ) = self.recompute_logprob(images, img_masks, lang_tokens, lang_masks, state, x_t, t, x_next, finish_step)
            return_dict = {
                "logp_action": logp_action,
                "logp_step": logp_step, 
                "logp_outer": logp_outer, 
                "logp_joint": logp_joint,
                "logp_elem": logp_elem,
                "ent_step": ent_step, 
                "ent_outer": ent_outer, 
                "ent_joint": ent_joint,
                "mean": mean,
                "std": std_dev_t,
                "out_metric": out_metric,
                }
            return None, lang_tokens, lang_masks, return_dict

        if use_sde:
            images = [img.to(torch.bfloat16) for img in images]
            lang_tokens = lang_tokens.to(torch.long)
            return_dict = self.sample_actions_sde(images, img_masks, lang_tokens, lang_masks, state, noise=noise)
            if return_logprob:
                return_dict, log_probs = return_dict
            actions = return_dict['x_next'][-1].detach().cpu().to(torch.float32).numpy()
        else:
            return_dict = self.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)
            actions = return_dict['x_next'][-1].detach().cpu().numpy()

        original_action_dim = 7
        actions = actions[:, :, :original_action_dim]

        outputs = {
            "state": np.zeros([bsize, 8]),
            "actions": actions,
        }
        actions = self._output_transform(outputs)['action']
        actions = actions[:, :self.n_action_steps, :]

        return (actions, lang_tokens, lang_masks, return_dict, log_probs) if return_logprob else actions, lang_tokens, lang_masks, return_dict


    # def embed_prefix(
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Embed images with SigLIP and language tokens with embedding layer to prepare
    #     for PaliGemma transformer processing.
    #     """

    #     # Process images
    #     for img, img_mask in zip(images, img_masks, strict=True):
                
    #         def image_embed_func(img):




    #         # Create attention masks so that image tokens attend to each other

    #     # Process language tokens
    #     def lang_embed_func(lang_tokens):



    #     # full attention between image and language inputs


    #     # Get batch size from the first dimension of the concatenated tensors



    def _standardize_img_mask(self, mask, B, V, T, device) -> torch.Tensor:
        """
        统一到 [B, V*T] 的 bool（True=pad）。允许：
        None, 标量/长度1, [T], [V,T], [B,1], [B,T], [B,V,1], [B,V,T], [B,V*T]
        标量或长度1会被广播到所有图像 token。
        """
        if mask is None:
            return torch.zeros(B, V * T, dtype=torch.bool, device=device)

        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        # 标量或长度1 -> 全局广播
        if mask.ndim == 0 or (mask.ndim == 1 and mask.numel() == 1):
            val = bool(mask.reshape(()).item())  # 兼容标量/长度1
            return torch.full((B, V * T), val, dtype=torch.bool, device=device)

        # [T] -> 广播到 [B, V*T]
        if mask.ndim == 1 and mask.shape[0] == T:
            return mask[None, None, :].expand(B, V, T).reshape(B, V * T).to(device)

        # [V,T] -> [B, V*T]
        if mask.ndim == 2 and mask.shape == (V, T):
            return mask.reshape(1, V * T).expand(B, V * T).to(device)

        # [B,1] -> 广播到 [B, V*T]
        if mask.ndim == 2 and mask.shape == (B, 1):
            return mask.expand(B, V * T).to(device)

        # [B,T] -> [B, V*T]
        if mask.ndim == 2 and mask.shape == (B, T):
            return mask[:, None, :].expand(B, V, T).reshape(B, V * T).to(device)

        # [B, V*T] -> 原样
        if mask.ndim == 2 and mask.shape == (B, V * T):
            return mask.to(device)

        # [B,V,1] -> [B,V*T]
        if mask.ndim == 3 and mask.shape == (B, V, 1):
            return mask.expand(B, V, T).reshape(B, V * T).to(device)

        # [B,V,T] -> [B,V*T]
        if mask.ndim == 3 and mask.shape == (B, V, T):
            return mask.reshape(B, V * T).to(device)

        raise ValueError(f"Unsupported img_mask shape {tuple(mask.shape)}; expected broadcastable to [B, V, T]")

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, compute_logp: bool=False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量友好的 embed：
        - images: 可迭代，每个元素形状 [B,C,H,W] 或 [B,V,C,H,W]
        - img_masks: 与 images 等长的可迭代，元素为 None/Bool Mask（见 _standardize_img_mask）
        - lang_tokens: [B,L]，lang_masks: [B,L] (bool; True=pad)
        返回:
        embs: [B, S, D]
        pad_masks: [B, S] (bool; True=pad)
        att_masks: [B, S] (这里构造全 0，表示允许完全注意力)
        """
        B = lang_tokens.shape[0]
        embs_parts = []
        pad_parts = []
        
        for img, img_mask in zip(images, img_masks, strict=True):
            if img.dim() == 4:
                if img.shape[0] != B:
                    raise ValueError(f"Inconsistent batch: image B={img.shape[0]} vs lang B={B}")
                V = 1
                def image_embed_func(x):
                    return self.paligemma_with_expert.embed_image(x)  # -> [B, T, D]
                img_emb = self._apply_checkpoint(image_embed_func, img)
                T = img_emb.shape[1]
                img_mask = img_mask.unsqueeze(-1)
                img_pad = self._standardize_img_mask(img_mask, B=B, V=V, T=T, device=img_emb.device)
                embs_parts.append(img_emb)                 # [B, T, D]
                pad_parts.append(img_pad)                  # [B, T]

            elif img.dim() == 5:
                # [B, V, C, H, W]，展平 BV 后再还原到 [B, V*T, D]
                if img.shape[0] != B:
                    raise ValueError(f"Inconsistent batch: image B={img.shape[0]} vs lang B={B}")
                B_, V = img.shape[:2]
                img_flat = img.reshape(B_ * V, *img.shape[2:])  # [B*V, C, H, W]

                def image_embed_func(x):
                    return self.paligemma_with_expert.embed_image(x)  # -> [B*V, T, D]
                img_emb_flat = self._apply_checkpoint(image_embed_func, img_flat)
                T = img_emb_flat.shape[1]
                D = img_emb_flat.shape[2]
                img_emb = img_emb_flat.view(B_, V * T, D)

                # 标准化 mask -> [B, V*T]
                img_pad = self._standardize_img_mask(img_mask, B=B_, V=V, T=T, device=img_emb.device)

                embs_parts.append(img_emb)     # [B, V*T, D]
                pad_parts.append(img_pad)      # [B, V*T]
            else:
                raise ValueError(f"Unsupported image ndim={img.dim()} (expect 4 or 5)")

        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)  # [B, L, D]
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)  # [B, L, D]
        if lang_emb.shape[0] != B:
            raise ValueError(f"Inconsistent batch: lang_emb B={lang_emb.shape[0]} vs lang_tokens B={B}")
        embs_parts.append(lang_emb)              # [B, L, D]
        pad_parts.append(lang_masks.to(torch.bool))  # [B, L]

        embs = torch.cat(embs_parts, dim=1)     # [B, S, D]
        pad_masks = torch.cat(pad_parts, dim=1) # [B, S] (bool)

        att_masks = torch.zeros_like(pad_masks, dtype=torch.bool, device=pad_masks.device)  # [B, S]

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        x_t_all, t_all, x_next_all, log_probs = [], [], [], []
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
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
        return return_dict

    def sample_actions_sde(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        
        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        sqrt_abs_dt = torch.sqrt(-dt)

        sde_sigma_max = 0.07
        sde_sigma_power = 1.5
        x_t = noise
        t = torch.tensor(1.0, dtype=torch.float32, device=device)
        x_t_all, t_all, x_next_all, log_probs = [], [], [], []
        while t >= -dt / 2:
            t_b = t.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                t_b,
            )
            
            sigmas = torch.tensor([1.0000, 0.9601, 0.9133, 0.8577, 0.7904, 0.7073, 0.6022, 0.4649, 0.2780, 0.0089, 0.0000], device=v_t.device, dtype=v_t.dtype)
            index = (num_steps * (1 - t)).to(torch.long)
            sigma = sigmas[index]
            sigma_max = sigmas[1]
            noise_level = 0.7
            std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level
            
            drift = v_t + (std_dev_t ** 2 / (2 * t + 1e-6)) * (x_t + (1 - t) * v_t)
            mean = x_t + drift * dt
            std  = (sqrt_abs_dt * std_dev_t).clamp_min(1e-6)
            eps = torch.randn_like(x_t)
            x_next = mean + std * eps


            x_t_all.append(x_t)
            t_all.append(t_b.detach().cpu())
            x_next_all.append(x_next)
            x_t  = x_next
            t = t + dt
            
        return_dict = {
            "x_t": x_t_all,
            "t": t_all,
            "x_next": x_next_all,
        }
        return return_dict



    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


    def recompute_logprob(
        self,
        images,             # [B,S,3,H,W]
        img_masks,          # [B,S,1]
        lang_tokens,        # [B,S,L]
        lang_masks,         # [B,S,L]
        state,              # [B,S,1,Ds]
        x_t,                # [B,S,K,50,Da]
        t,                  # [B,S,K]
        x_next,             # [B,S,K,50,Da]
        finish_step,        # [B]
    ):
        """
        返回:
        logp_step  : [B,S,K]   —— 每个外层步、每个流步的 log-prob（已mask）
        logp_outer : [B,S]     —— 每个外层步联合（对K求和，已mask）
        logp_joint : [B]       —— 整条有效轨迹联合（对S再求和，已mask）
        """
        # ---------- 基本维度 ----------
        device = lang_tokens.device
        B, S, K, CH, D = x_t.shape  # CH=50, D=动作维
        assert t.shape == (B, S, K)
        assert x_next.shape == (B, S, K, CH, D)
        dt = -1.0 / float(K)                       # 按你的设定：K个等步，t: 1→0
        sqrt_abs_dt = math.sqrt(-dt)
        dtype = torch.float32                      # 计算用 fp32 更稳

        # ---------- 构建 prefix KV（按 [B*S] 批量） ----------
        BS = B * S
        # 展平外层时间 S 维
        def _flat(x, keep_tail=0):
            # 把 [B,S,...] → [B*S,...]
            if keep_tail == 0:
                return x.reshape(BS, *x.shape[2:])
            elif keep_tail == 1:
                return x.reshape(BS, x.shape[-1])
            else:
                raise ValueError

        images_flat    = _flat(images)
        img_masks_flat = _flat(img_masks)
        lang_tokens_f  = _flat(lang_tokens)
        lang_masks_f   = _flat(lang_masks)
        state = state.unsqueeze(1).repeat(1, S, 1)
        state_flat     = _flat(state) if state is not None else None

        x_t_flat    = x_t.reshape(BS * K, CH, D).to(device=device, dtype=dtype)
        x_next_f    = x_next.to(device=device, dtype=dtype) # 保持原始形状，后面用
        t_flat      = t.reshape(BS * K).to(device=device, dtype=dtype)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            [images_flat], [img_masks_flat.squeeze(-1)], lang_tokens_f, lang_masks_f, compute_logp=True
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state_flat, x_t_flat, t_flat)
        prefix_embs_rep = prefix_embs.repeat_interleave(K, dim=0)
        prefix_pad_masks_rep = prefix_pad_masks.repeat_interleave(K, dim=0)
        prefix_att_masks_rep = prefix_att_masks.repeat_interleave(K, dim=0)
        pad_masks = torch.cat([prefix_pad_masks_rep, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks_rep, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        # (_, suffix_out), _ = self.paligemma_with_expert.forward(
        
        def forward_func(prefix_embs_rep, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs_rep, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs_rep, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t_flat = self.action_out_proj(suffix_out)
        v_t_all    = v_t_flat.view(B, S, K, CH, D)


        # # 用来收集每个 k 的输出

        # # 你可以根据显存调这个数，比如 1、2、4

        # for k_start in range(0, K, K_CHUNK):

        #     # 这一块的索引范围

        #     # 取出这一小块的 suffix

        #     # 把 prefix 复制到这一块的 batch 上
        #     # prefix: [BS, Lp, H] → [BS*cur_chunk, Lp, H]

        #     # 拼 mask

        #     # 真正 forward —— 这里只跑这一小块
        #     # ( _ , suffix_out), _ = self.paligemma_with_expert.forward(
        #     def forward_func(prefix_embs_rep, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
        #         (_, suffix_out), _ = self.paligemma_with_expert.forward(

        #         forward_func, prefix_embs_rep, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond

        #     # 后处理保持你原来的逻辑

        # # 把 K 方向拼回来
        
        x_t_f = x_t.to(device=device, dtype=dtype)
        t_f = t.to(device=device, dtype=dtype)

        t_safe = t_f.clamp(1e-4, 1 - 1e-4).view(B, S, K)
        t3     = t_safe[..., None, None]
        
        sigmas = torch.tensor([1.0000, 0.9601, 0.9133, 0.8577, 0.7904, 0.7073, 0.6022, 0.4649, 0.2780, 0.0089, 0.0000], device=v_t_flat.device, dtype=v_t_flat.dtype)
        index = (K * (1 - t)).to(torch.long)
        sigma = sigmas[index]
        sigma_max = sigmas[1]
        noise_level = 0.7
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level
        std_dev_t = std_dev_t[..., None, None]
        
        drift = v_t_all + (std_dev_t**2 / (2 * t3 + 1e-6)) * (x_t_f + (1 - t3) * v_t_all)

        mean = x_t_f + dt * drift
        std  = (sqrt_abs_dt * std_dev_t).clamp_min(1e-6)

        CH = self.n_action_steps
        CH_idx = torch.arange(CH, device=device)[None, None, :]
        S_idx  = torch.arange(S,  device=device)[None, :, None]
        s_fin  = (finish_step.to(device) // CH).view(B, 1, 1)
        c_fin  = (finish_step.to(device) %  CH).view(B, 1, 1)

        mask_before = (S_idx <  s_fin).float()
        mask_equal  = (S_idx == s_fin).float() * (CH_idx < c_fin).float()
        mask_actions = mask_before.expand(B, S, CH) + mask_equal
        mask_elem = mask_actions[:, :, None, :, None]
        lp_elem = (
            -((x_next_f.detach() - mean) ** 2) / (2 * ((std)**2))
            - torch.log(std)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        lp_elem = lp_elem[..., :self.n_action_steps, :]            # [B,S,K,CH,D]
        lp_elem = lp_elem * mask_elem
        original_action_dim = 7
        lp_elem = lp_elem[..., :original_action_dim]
        logp_action = lp_elem.sum(dim=(-3))
        
        logp_step = lp_elem.sum(dim=(-1, -2))
        logp_outer = logp_step.sum(dim=2)
        logp_joint = logp_outer.sum(dim=1)
        
        c0 = 0.5 * (1.0 + math.log(2.0 * math.pi))
        h_per_dim = c0 + torch.log(std).squeeze(-1).squeeze(-1)
        num_valid_actions = mask_actions.sum(dim=-1)
        num_valid_dims    = (num_valid_actions * D).unsqueeze(-1).to(h_per_dim.dtype)
        
        ent_step  = h_per_dim * num_valid_dims
        ent_outer = ent_step.sum(dim=2)
        ent_joint = ent_outer.sum(dim=1)
        

        mean = mean[..., :original_action_dim]
        std_dev_t = std_dev_t[..., :original_action_dim]
        std = std[..., :original_action_dim]
        out_metric = self.summarize_logprob_metrics(
            logp_step, logp_outer, logp_joint,
            mean, std, x_next_f, mask_actions, mask_elem,
            t, sigmas, finish_step, original_action_dim=7
        )
        return (
            logp_action, 
            logp_step, 
            logp_outer, 
            logp_joint, 
            lp_elem,
            ent_step, 
            ent_outer, 
            ent_joint, 
            mean, 
            # std_dev_t,
            std,
            out_metric,
        )
    
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
        err = err[..., :self.n_action_steps, :]
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
