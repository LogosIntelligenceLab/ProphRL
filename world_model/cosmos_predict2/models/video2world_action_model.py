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

import torch
from megatron.core import parallel_state
from torch.distributed.device_mesh import init_device_mesh

from cosmos_predict2.models.video2world_model import Predict2Video2WorldModel, Predict2Video2WorldModelConfig
from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedPipeline, Video2WorldActionConditionedBatchPipeline
from imaginaire.model import ImaginaireModel
from imaginaire.utils import log

import os
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import cv2
from imaginaire.utils.io import save_image_or_video
from imaginaire.utils.easy_io import easy_io

@torch.no_grad()
def _to_float01_gpu(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.max() > 1.0:
        x = x / 255.0
    return x.clamp(0, 1)

def _gaussian_window_1d(kernel_size=11, sigma=1.5, device="cuda", dtype=torch.float32):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1)/2
    g = torch.exp(-(coords**2) / (2*sigma*sigma))
    g = g / g.sum()
    return g

def _create_gaussian_kernel_2d(kernel_size=11, sigma=1.5, channels=3, device="cuda", dtype=torch.float32):
    g1 = _gaussian_window_1d(kernel_size, sigma, device, dtype)
    g2 = g1[:, None] @ g1[None, :]
    g2 = g2.expand(channels, 1, kernel_size, kernel_size).contiguous()
    return g2  # (C,1,kh,kw)

def _ssim_gpu(x, y, kernel, C1=0.01**2, C2=0.03**2, eps=1e-12):
    C = x.shape[1]
    padding = kernel.shape[-1] // 2
    # depthwise conv
    mu_x = F.conv2d(x, kernel, padding=padding, groups=C)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=padding, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=padding, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=C) - mu_xy

    ssim_map = ((2*mu_xy + C1) * (2*sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + eps)
    return ssim_map.mean(dim=(1,2,3))

def psnr_from_mse(mse, eps=1e-12):
    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=eps))

@torch.no_grad()
def eval_videos_cuda(gen: torch.Tensor, gt: torch.Tensor, lpips_net: str | None = None):
    assert gen.device.type == "cuda" and gt.device.type == "cuda"
    assert gen.shape == gt.shape and gen.dim() == 4 and gen.size(-1) == 3

    gen = _to_float01_gpu(gen)
    gt  = _to_float01_gpu(gt)

    T, H, W, _ = gen.shape
    gen_chw = gen.permute(0,3,1,2).contiguous()
    gt_chw  = gt.permute(0,3,1,2).contiguous()

    mse_per = F.mse_loss(gen_chw, gt_chw, reduction="none").mean(dim=(1,2,3))   # (T,)
    psnr_per = psnr_from_mse(mse_per)

    kernel = _create_gaussian_kernel_2d(kernel_size=11, sigma=1.5, channels=3,
                                        device=gen.device, dtype=gen_chw.dtype)
    ssim_per = _ssim_gpu(gen_chw, gt_chw, kernel)  # (T,)

    lpips_avg = None
    if lpips_net is not None:
        try:
            import lpips
            loss_fn = lpips.LPIPS(net=lpips_net).to(gen.device)
            to_lpips = lambda x: x.mul(2).sub(1)
            bs = 16
            vals = []
            for i in range(0, T, bs):
                v = loss_fn(to_lpips(gen_chw[i:i+bs]), to_lpips(gt_chw[i:i+bs]))
                vals.append(v.view(-1))
            lpips_avg = torch.cat(vals, 0).mean().item()
        except Exception:
            lpips_avg = None

    if T > 1:
        gen_dt = gen_chw[1:] - gen_chw[:-1]
        gt_dt  = gt_chw[1:]  - gt_chw[:-1]
        tssim_per = _ssim_gpu(gen_dt, gt_dt, kernel)  # (T-1,)
        tssim_avg = tssim_per.mean().item()
    else:
        tssim_per = torch.empty(0, device=gen.device)
        tssim_avg = None

    out = {
        "MSE_mean":  mse_per.mean().item(),
        "PSNR_mean": psnr_per.mean().item(),
        "SSIM_mean": ssim_per.mean().item(),
        "tSSIM_mean": tssim_avg,
        # "LPIPS_mean": lpips_avg,
        # "per_frame": {
        #     "MSE":  mse_per.detach().cpu().numpy(),
        #     "PSNR": psnr_per.detach().cpu().numpy(),
        #     "SSIM": ssim_per.detach().cpu().numpy(),
        #     "tSSIM": tssim_per.detach().cpu().numpy(),
        # }
    }
    return out


def compute_flows_opencv(frames_rgb):
    """(T,H,W,3)->list of (H,W,2) float32 using Farnebäck"""
    flows = []
    prev = cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2GRAY)
    for t in range(1, len(frames_rgb)):
        nxt = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None,
                                            pyr_scale=0.5, levels=3,
                                            winsize=15, iterations=3,
                                            poly_n=5, poly_sigma=1.2, flags=0)
        flows.append(flow.astype(np.float32))
        prev = nxt
    return flows

def flow_bgr_vis(flow):
    fx, fy = flow[...,0], flow[...,1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=False)
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[...,0] = (ang / (2*np.pi) * 180).astype(np.uint8)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def compare_videos_with_flow(video_real, video_gen):
    f1 = compute_flows_opencv(video_real)
    f2 = compute_flows_opencv(video_gen)
    n = min(len(f1), len(f2))
    epe_list, cos_list = [], []
    flow_a_list, flow_b_list, epe_vis_list = [], [], []
    for i in range(n):
        a, b = f1[i], f2[i]
        diff = a - b
        epe = np.sqrt((diff**2).sum(-1))
        epe_list.append(float(epe.mean()))

        na = np.linalg.norm(a, axis=-1); nb = np.linalg.norm(b, axis=-1)
        cos = ((a*b).sum(-1)) / (na*nb + 1e-6)
        valid = (na > 1e-3) & (nb > 1e-3)
        cos_list.append(float(np.mean(cos[valid])) if np.any(valid) else np.nan)

        flow_a_list.append(flow_bgr_vis(a))
        flow_b_list.append(flow_bgr_vis(b))
        epe_norm = cv2.normalize(epe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        epe_vis_list.append(cv2.applyColorMap(epe_norm, cv2.COLORMAP_JET))

    arr = np.asarray(cos_list, dtype=np.float64)
    valid = np.isfinite(arr)
    if valid.any():
        with np.errstate(all='ignore'):
            dir_cos_mean   = float(np.nanmean(arr[valid]))
            dir_cos_median = float(np.nanmedian(arr[valid]))
    else:
        dir_cos_mean = dir_cos_median = 0.0

    return {
        "pairs": n,
        "EPE_mean": float(np.mean(epe_list)),
        "EPE_median": float(np.median(epe_list)),
        "dir_cos_mean": dir_cos_mean,
        "dir_cos_median": dir_cos_median,
        "flow_a": np.array(flow_a_list),
        "flow_b": np.array(flow_b_list),
        "epe": np.array(epe_vis_list),
    }

class Predict2Video2WorldActionConditionedModel(Predict2Video2WorldModel):
    def __init__(self, config: Predict2Video2WorldModelConfig):
        super(ImaginaireModel, self).__init__()

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

        # 5. other additional configs
        self.use_self_forcing = getattr(config, "use_self_forcing", False)

        # 7. training states
        if parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

        # NOTE: replace the pipeline with action-conditioned setup
        self.pipe = Video2WorldActionConditionedPipeline.from_config(
            config.pipe_config,
            dit_path=config.model_manager_config.dit_path,
            use_text_encoder=False,
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
        else:
            self.pipe.denoising_model().requires_grad_(True)
        # del self.pipe.text_encoder
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
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

    @torch.no_grad()
    def validation_step(self, data_batch: dict, data_batch_idx: int) -> tuple[dict, torch.Tensor]:
        # action: (B, F, action_dim)
        # video: (B, 3, F + 1, S, S)
        # num_frames: [F + 1]
        first_frame = data_batch['first_frame'][0].permute(1,2,0).detach().cpu().numpy()
        input_action = data_batch['action'][0].float().detach().cpu().numpy()
        if 'blacks' in data_batch.keys():
            input_blacks = data_batch['blacks'].float().detach().cpu().numpy()
        else:
            input_blacks = None
        if 'history' in data_batch.keys():
            input_history = data_batch['history'].float().detach().cpu().numpy()
        else:
            input_history = None
        # video = self.pipe(first_frame, input_action, num_conditional_frames=1, guidance=0)
        video = self.pipe(first_frame, input_action, blacks=input_blacks, history=input_history, num_conditional_frames=1, guidance=0, use_cuda_graphs=False)
        output_batch, kendall_loss = self.training_step(data_batch, data_batch_idx)
        output_batch['generated_video'] = video
        return output_batch, kendall_loss

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        path_local = output_batch['path_local']
        video_dir = os.path.join(path_local, 'videos', str(iteration).zfill(5))
        gt_video_dir = os.path.join(path_local, 'videos_gt', str(iteration).zfill(5))
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(gt_video_dir, exist_ok=True)
        video, gt_video = output_batch['generated_video'], data_batch['video'] # (1, 3, F + 1, S, S)

        production = False
        if 'use_action_bias' in data_batch.keys():
            production = True
            use_action_bias = bool(data_batch['use_action_bias'][0])
        if '__task__' in data_batch.keys():
            task = data_batch['__task__'][0]
            task_suffix = '_' + task
        else:
            task = 'total'
            task_suffix = ''

        if not production:
            samples_name = data_batch['__key__'][0].split('/')[-1].replace('.mp4', '') + task_suffix + '.mp4'
        elif use_action_bias:
            samples_name = data_batch['__key__'][0].split('/')[-1].replace('.mp4', '') + task_suffix + '_biased.mp4'
        elif not use_action_bias:
            samples_name = data_batch['__key__'][0].split('/')[-1].replace('.mp4', '') + task_suffix + '_origin.mp4'
        output_path = os.path.join(video_dir, samples_name)
        save_image_or_video(video, output_path, fps=4)
        gt_output_path = os.path.join(gt_video_dir, samples_name)
        save_image_or_video(gt_video, gt_output_path, fps=4)

        # evaling video quality
        if not production:
            gt_video = ((gt_video[0] + 1.0) / 2.0).clamp(0, 1)
            gt_video = rearrange((gt_video * 255), "c t h w -> t h w c") + 0.5
            eval_video = ((video[0] + 1.0) / 2.0).clamp(0, 1)
            eval_video = rearrange((eval_video * 255), "c t h w -> t h w c") + 0.5 # (F + 1, S, S, 3)
            eval_result = eval_videos_cuda(eval_video, gt_video)
            eval_result['latent_loss'] = loss

            # eval flow
            flow_result = compare_videos_with_flow(eval_video.float().detach().cpu().numpy().astype(np.uint8), gt_video.float().detach().cpu().numpy().astype(np.uint8))
            flow_dir = os.path.join(path_local, 'flows', str(iteration).zfill(5))
            gt_flow_dir = os.path.join(path_local, 'flows_gt', str(iteration).zfill(5))
            epe_vis_dir = os.path.join(path_local, 'epe_vis', str(iteration).zfill(5))
            os.makedirs(flow_dir, exist_ok=True)
            os.makedirs(gt_flow_dir, exist_ok=True)
            os.makedirs(epe_vis_dir, exist_ok=True)
            # samples_name = data_batch['__key__'][0].split('/')[-1]
            flow_out_path = os.path.join(flow_dir, samples_name)
            gt_flow_out_path = os.path.join(gt_flow_dir, samples_name)
            epe_out_path = os.path.join(epe_vis_dir, samples_name)
            easy_io.dump(flow_result['flow_a'], flow_out_path, file_format="mp4", format="mp4", fps=4)
            easy_io.dump(flow_result['flow_b'], gt_flow_out_path, file_format="mp4", format="mp4", fps=4)
            easy_io.dump(flow_result['epe'], epe_out_path, file_format="mp4", format="mp4", fps=4)

            # merge eval results
            eval_result['EPE_mean'] = flow_result['EPE_mean']
            eval_result['EPE_median'] = flow_result['EPE_median']
            eval_result['dir_cos_mean'] = flow_result['dir_cos_mean']
            eval_result['dir_cos_median'] = flow_result['dir_cos_median']
        else:
            eval_result = {}
            eval_result['latent_loss'] = loss
        return eval_result, samples_name
