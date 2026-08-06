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


import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import math

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.models.video2world_dit import MinimalV1LVGDiT
from imaginaire.utils.graph import create_cuda_graph
from contextlib import contextmanager

@contextmanager
def temporarily_remove_hooks(root: torch.nn.Module):
    saved = []  # [(m, f_pre_dict, f_dict, b_dict)]
    for m in root.modules():
        fpre = dict(m._forward_pre_hooks)
        fwd  = dict(m._forward_hooks)
        bwd  = dict(m._backward_hooks)
        if fpre or fwd or bwd:
            saved.append((m, fpre, fwd, bwd))
            m._forward_pre_hooks.clear()
            m._forward_hooks.clear()
            m._backward_hooks.clear()
    try:
        yield
    finally:
        for m, fpre, fwd, bwd in saved:
            for _, fn in fpre.items():
                m.register_forward_pre_hook(fn)
            for _, fn in fwd.items():
                m.register_forward_hook(fn)
            for _, fn in bwd.items():
                m.register_full_backward_hook(fn)

class FramePackMemory(nn.Module):
    def __init__(self, in_ch: int, D: int,
                 kernels=((1,2,2),(1,4,4),(1,8,8)),
                 lam: float = 2.0,
                 max_tokens: int = 1024,
                 tail: str = "one"):
        super().__init__()
        self.lam, self.max_tokens, self.tail = lam, max_tokens, tail
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool3d(k, k),
                nn.Conv3d(in_ch, D, kernel_size=1)
            ) for k in kernels
        ])

    def _levels(self, T: int):
        L = len(self.proj)
        return [min(int(round(math.log(max(1, i+1), self.lam))), L-1) for i in range(T)]

    def forward(self, hist_B_C_T_H_W: torch.Tensor) -> torch.Tensor | None:
        if hist_B_C_T_H_W is None: return None
        B,C,T,H,W = hist_B_C_T_H_W.shape
        if T == 0: return None

        levels = self._levels(T)
        toks = []
        for lvl, block in enumerate(self.proj):
            sel = [t for t,l in enumerate(levels) if l == lvl]
            if not sel: continue
            x = hist_B_C_T_H_W[:, :, sel]                   # [B,C,Tk,H,W]
            x = block(x)                                    # [B,D,Tk,H',W']
            res = x.flatten(2).transpose(1,2)
            toks.append(res)        # → [B, Tk*H'*W', D]

        if not toks:
            return None

        mem = torch.cat(toks, dim=1)                        # [B, L_hist, D]
        mem = mem[:, :self.max_tokens]
        return mem


def sinusoidal_pe(T: int, D: int, device):
    dim = torch.arange(D, device=device).float()
    pos = torch.arange(T, device=device).float()[:, None]
    div = torch.exp(-torch.arange(0, D, 2, device=device).float() * (torch.log(torch.tensor(10000.0)) / D))
    pe = torch.zeros(T, D, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(torch.bfloat16)

class ZeroConv3d1x1(nn.Conv3d):
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, kernel_size=1, bias=True)
        nn.init.zeros_(self.weight); nn.init.zeros_(self.bias)

class ActionLatentConditioner(nn.Module):
    """
    blacks: (B, Cz=16, tl, hl, wl)  →  cond_D: (B, tl, D),  cond_3D: (B, tl, 3D)
    """
    def __init__(
        self,
        in_channels: int,          # 16
        model_channels: int,       # D
        nheads: int = 8,
        nlayers: int = 2,
        ff_mult: int = 4,
        use_temporal_transformer: bool = False,
        zero_init_gate: float = 0.0,
        norm: str = "ln",
    ):
        super().__init__()
        D = model_channels
        self.D = D
        self.use_temporal_transformer = use_temporal_transformer

        self.proj_in = ZeroConv3d1x1(in_channels, D)
        self.dw_spatial = nn.Conv3d(D, D, kernel_size=(1,3,3), padding=(0,1,1), groups=D, bias=False)
        self.pw_spatial = nn.Conv3d(D, D, kernel_size=1, bias=False)

        if use_temporal_transformer:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=D, nhead=nheads, dim_feedforward=ff_mult*D,
                batch_first=True, dropout=0.0, activation="gelu", norm_first=True
            )
            self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        self.to_D   = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D))
        self.to_3D  = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 3*D))

    @torch.no_grad()
    def _ensure_time_match(self, x_B_C_T_H_W, T_target: int):
        B, C, T, H, W = x_B_C_T_H_W.shape
        if T == T_target:
            return x_B_C_T_H_W
        x = x_B_C_T_H_W.permute(0,1,3,4,2).contiguous()      # (B,C,H,W,T)
        x = F.interpolate(x, size=T_target, mode="linear", align_corners=False)  # 1D over T
        x = x.permute(0,1,4,2,3).contiguous()                # (B,C,T,H,W)
        return x

    def forward(
        self,
        blacks: torch.Tensor,                                   # (B,16,tl,hl,wl)
        T_target: int | None = None,                            # 与 t_embedding 的 T 对齐
        condition_video_input_mask_B_C_T_H_W: torch.Tensor | None = None,  # (B,1,tl,hl,wl)
    ):
        assert blacks.dim() == 5, f"expect (B,Cz,T,H,W), got {tuple(blacks.shape)}"
        B, Cz, T, H, W = blacks.shape
        if T_target is not None:
            blacks = self._ensure_time_match(blacks, T_target)
            T = T_target

        x = blacks                                                    # (B,16,T,H,W)
        x = self.proj_in(x)
        x = self.dw_spatial(x); x = self.pw_spatial(x)                # (B,D,T,H,W)

        if condition_video_input_mask_B_C_T_H_W is not None:
            gen_mask = (1 - condition_video_input_mask_B_C_T_H_W).to(dtype=x.dtype, device=x.device)  # (B,1,T,H,W)
            x = x * gen_mask

        x = x.mean(dim=(3,4))                                         # (B,D,T)
        x = rearrange(x, 'b d t -> b t d').contiguous()               # (B,T,D)

        pe = sinusoidal_pe(T, self.D, x.device)                       # (T,D)
        x = x + pe

        if self.use_temporal_transformer:
            x = self.temporal_encoder(x)                              # (B,T,D)
        else:
            pass

        cond_D  = self.to_D(x)                         # (B,T,D)
        cond_3D = self.to_3D(x)                        # (B,T,3D)
        return cond_D, cond_3D


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ActionConditionedMinimalV1LVGDiT(MinimalV1LVGDiT):
    def __init__(self, *args, **kwargs):
        assert "action_dim" in kwargs, "action_dim must be provided"
        action_dim = kwargs["action_dim"]
        del kwargs["action_dim"]
        use_black = bool(kwargs.pop("use_black", False))
        use_history = bool(kwargs.pop("use_history", False))
        super().__init__(*args, **kwargs)

        # self.action_embedder_B_D = Mlp(
        #     in_features=action_dim,
        #     hidden_features=self.model_channels * 4,
        #     out_features=self.model_channels,
        #     act_layer=lambda: nn.GELU(approximate="tanh"),
        #     drop=0,
        # )
        # self.action_embedder_B_3D = Mlp(
        #     in_features=action_dim,
        #     hidden_features=self.model_channels * 4,
        #     out_features=self.model_channels * 3,
        #     act_layer=lambda: nn.GELU(approximate="tanh"),
        #     drop=0,
        # )

        self.action_embedder_B_D_agi = Mlp(
            in_features=action_dim,
            hidden_features=self.model_channels * 4,
            out_features=self.model_channels,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.action_embedder_B_3D_agi = Mlp(
            in_features=action_dim,
            hidden_features=self.model_channels * 4,
            out_features=self.model_channels * 3,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.use_black = use_black
        if self.use_black:
            self.action_latent_cond = ActionLatentConditioner(
                in_channels=16,
                model_channels=self.model_channels,
                nheads=8, nlayers=2, ff_mult=4,
                use_temporal_transformer=False,
                zero_init_gate=0.0,
                norm="ln",
            )
        self.use_history = use_history
        if self.use_history:
            self.framepack_mem = FramePackMemory(
                in_ch=16,
                D=self.model_channels,
                kernels=((1,2,2),(1,4,4),(1,8,8)),
                lam=2.0, max_tokens=1024, tail="one",
            )
            self.mem_to_k = nn.Linear(self.model_channels, self.model_channels, bias=False)
            self.mem_to_v = nn.Linear(self.model_channels, self.model_channels, bias=False)

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor, # (B, 16, tl, hl, wl)
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor | None = None, # (B, 1, tl, hl, wl)
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        data_type: DataType | None = DataType.VIDEO,
        use_cuda_graphs: bool = False,
        action: torch.Tensor | None = None,
        blacks: torch.Tensor | None = None, #(B, 16, tl, hl, wl)
        history: torch.Tensor | None = None, # (B, 16, thl, hl, wl)
        **kwargs,
    ) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        del kwargs

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )
        # NOTE: project action to action embedding
        assert action is not None, "action must be provided"
        action = rearrange(action, "b t d -> b 1 (t d)")
        action_emb_B_D = self.action_embedder_B_D_agi(action)
        action_emb_B_3D = self.action_embedder_B_3D_agi(action)
        if self.use_history and history is not None:
            mem_tokens = self.framepack_mem(history.to(torch.bfloat16)) #(B, th, D)
            if mem_tokens is not None:
                k_mem = self.mem_to_k(mem_tokens) #(B, th, D)
                v_mem = self.mem_to_v(mem_tokens) #(B, th, D)
        else:
            k_mem, v_mem = None, None

        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )
        assert not (self.training and use_cuda_graphs), "CUDA Graphs are supported only for inference"
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)

        T_target = t_embedding_B_T_D.shape[1]
        if self.use_black and blacks is not None:
            cond_D, cond_3D = self.action_latent_cond(
                blacks=blacks.to(torch.bfloat16),
                T_target=T_target,
                condition_video_input_mask_B_C_T_H_W=condition_video_input_mask_B_C_T_H_W,
            )
        # NOTE: add action embedding to the timestep embedding and adaln_lora
        t_embedding_B_T_D = t_embedding_B_T_D + action_emb_B_D # (B, tl, 2048), (B, 1, 2048)
        if adaln_lora_B_T_3D is not None:
            adaln_lora_B_T_3D = adaln_lora_B_T_3D + action_emb_B_3D # (B, tl, 6144), (B, 1, 6144)
        else:
            adaln_lora_B_T_3D = action_emb_B_3D
        if self.use_black and blacks is not None:
            t_embedding_B_T_D = t_embedding_B_T_D + cond_D
            adaln_lora_B_T_3D = adaln_lora_B_T_3D + cond_3D
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # for logging purpose
        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
            )

        if use_cuda_graphs:
            with temporarily_remove_hooks(self):
                shapes_key = create_cuda_graph(  # noqa: F821
                    self.cuda_graphs,
                    self.blocks,
                    x_B_T_H_W_D,
                    t_embedding_B_T_D,
                    crossattn_emb,
                    rope_emb_L_1_1_D,
                    adaln_lora_B_T_3D,
                    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                    k_mem=k_mem,
                    v_mem=v_mem,
                )
            blocks = self.cuda_graphs[shapes_key]
        else:
            blocks = self.blocks

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            "k_mem": k_mem,
            "v_mem": v_mem,
        }
        for block in blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D, #（B, ld + 1, lt, lh, lw）
                t_embedding_B_T_D, # (B, lt, 2048)
                crossattn_emb, # (B, 512, 1024)
                **block_kwargs,
            )
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D) #(B, lt, lh/2, lw/2, 64)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O) # (B, ld, lt, lh, lw)
        return x_B_C_Tt_Hp_Wp
