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

from dataclasses import dataclass

import torch
import wandb
import torch.distributed as dist

from imaginaire.model import ImaginaireModel
from imaginaire.utils import distributed
from imaginaire.utils.callback import Callback

from imaginaire.utils import log
from imaginaire.utils.distributed import rank0_only

def _gather_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            yield p.grad

def compute_grad_norm_global(params, norm_type: float = 2.0, group=None) -> torch.Tensor:
    grads = []
    for p in params:
        if p.grad is not None:
            g = _to_local_tensor(p.grad)
            grads.append(g)

    if not grads:
        return torch.tensor(0.0, device="cuda" if torch.cuda.is_available() else "cpu")

    if norm_type == float("inf"):
        local_max = max((g.detach().abs().max().to(torch.float32) for g in grads), default=torch.tensor(0.0))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=group)
        return local_max
    else:
        p = norm_type
        local_sum = torch.zeros((), device=grads[0].device, dtype=torch.float32)
        for g in grads:
            if g.numel() > 0:
                n = g.detach().to(torch.float32).norm(p)
                local_sum = local_sum + n.pow(p)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=group)
        return local_sum.pow(1.0 / p)

def _is_dtensor(x) -> bool:
    return "DTensor" in type(x).__name__

def _to_local_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.to_local() if _is_dtensor(x) else x

def _to_float(x: torch.Tensor) -> float:
    if _is_dtensor(x):
        try:
            x = x.to_local()
        except Exception:
            x = x.to(torch.float32).cpu()
    if x.numel() == 0:
        return 0.0
    return float(x.detach().to(torch.float32).cpu().item())

@torch.jit.script
def _fused_nan_to_num(params: list[torch.Tensor]):
    for param in params:
        torch.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0, out=param)

@dataclass
class _MagnitudeRecord:
    state: float = 0
    iter_count: int = 0

    def reset(self) -> None:
        self.state = 0
        self.iter_count = 0

    def update(self, cur_state: torch.Tensor) -> None:
        self.state += cur_state
        self.iter_count += 1

    def get_stat(self) -> tuple[float, float]:
        if self.iter_count > 0:
            avg_state = self.state / self.iter_count
            avg_state = avg_state.item()
        else:
            avg_state = 0
        self.reset()
        return avg_state


class GradClip(Callback):
    def __init__(self, clip_norm=1.0, force_finite: bool = True):
        self.clip_norm = clip_norm
        self.force_finite = force_finite

        self.img_mag_log = _MagnitudeRecord()
        self.video_mag_log = _MagnitudeRecord()
        self._cur_state = None

    def on_training_step_start(
        self, model: ImaginaireModel, data_batch: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        if model.is_image_batch(data_batch):
            self._cur_state = self.img_mag_log
        else:
            self._cur_state = self.video_mag_log

    @rank0_only
    def _log_to_console(self, pre_norm, post_norm, clipped, iteration):
        log.info(f"[grad] iter={iteration} pre={pre_norm:.4f} post={post_norm:.4f} clip_max={self.clip_norm} clipped={bool(clipped)}")

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del scheduler
        model = model_ddp.module if isinstance(model_ddp, distributed.DistributedDataParallel) else model_ddp
        try:
            grad_scaler.unscale_(optimizer)
        except Exception:
            pass

        pre_norm_t = compute_grad_norm_global(model.parameters())
        if self.force_finite:
            params = [p.grad for p in model.parameters() if p.grad is not None]
            _fused_nan_to_num(params)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_norm)

        post_norm_t = compute_grad_norm_global(model.parameters())
        self._cur_state.update(post_norm_t)
        pre_norm  = _to_float(pre_norm_t)
        post_norm = _to_float(post_norm_t)
        clipped_happened = (pre_norm > self.clip_norm + 1e-12)
        self._log_to_console(pre_norm, post_norm, clipped_happened, iteration)

    # def on_before_optimizer_step(
    #     self,
    #     model_ddp: distributed.DistributedDataParallel,
    #     optimizer: torch.optim.Optimizer,
    #     scheduler: torch.optim.lr_scheduler.LRScheduler,
    #     grad_scaler: torch.amp.GradScaler,
    #     iteration: int = 0,
    # ) -> None:
    #     del optimizer, scheduler
    #     if isinstance(model_ddp, distributed.DistributedDataParallel):
    #         model = model_ddp.module
    #     else:
    #         model = model_ddp
    #     params = []
    #     if self.force_finite:
    #         for param in model.parameters():
    #             if param.grad is not None:
    #                 params.append(param.grad)
    #         _fused_nan_to_num(params)

    #     total_norm = model.clip_grad_norm_(self.clip_norm)

    #     self._cur_state.update(total_norm)
    #     if iteration % self.config.trainer.logging_iter == 0:
    #         avg_img_mag, avg_video_mag = self.img_mag_log.get_stat(), self.video_mag_log.get_stat()
    #         if wandb.run:
    #             wandb.log(
    #                 {
    #                     "clip_grad_norm/image": avg_img_mag,
    #                     "clip_grad_norm/video": avg_video_mag,
    #                     "iteration": iteration,
    #                 },
    #                 step=iteration,
    #             )
