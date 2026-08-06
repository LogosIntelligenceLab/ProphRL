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

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.configs.action_conditioned.paths import LIBERO_DATA
from cosmos_predict2.data.action_conditioned.action_conditioned_libero_dataset import ActionConditionedLiberoDataset
from imaginaire.lazy_config import LazyCall as L

base_path = LIBERO_DATA

libero_train_dataset = L(ActionConditionedLiberoDataset)(
    data_path=base_path,
    sequence_interval=1,
    num_frames=21, #13
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 256], # [480, 480]
    val_start_frame_interval=1,
    load_t5_embeddings=False, # False
    servo=True,
    mode="train",
)

libero_val_dataset = L(ActionConditionedLiberoDataset)(
    data_path=base_path,
    sequence_interval=1,
    num_frames=21, # 13
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 256], # [480, 480]
    val_start_frame_interval=50,
    load_t5_embeddings=False, # False
    servo=True,
    mode="val",
)


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


libero_train_dataloader = L(DataLoader)(
    dataset=libero_train_dataset,
    sampler=L(get_sampler)(dataset=libero_train_dataset),
    batch_size=24,
    num_workers=24,
    drop_last=True,
)

libero_val_dataloader = L(DataLoader)(
    dataset=libero_val_dataset,
    sampler=L(get_sampler)(dataset=libero_val_dataset),
    batch_size=1,
    drop_last=True,
)


def register_training_and_val_data_action_conditioned():
    cs = ConfigStore.instance()

    # for local dataset
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="libero_train",
        node=libero_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="libero_val",
        node=libero_val_dataloader,
    )
