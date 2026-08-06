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

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.configs.action_conditioned.paths import BRIDGE_DATA
from cosmos_predict2.data.action_conditioned.action_conditioned_bridge_dataset import ActionConditionedBridgeDataset
from imaginaire.lazy_config import LazyCall as L

base_path = BRIDGE_DATA
state_type = "pos_euler"

bridge_train_dataset = L(ActionConditionedBridgeDataset)(
    data_path=base_path,
    num_frames=21, #13
    video_size=[256, 256], # [480, 480]
    val_start_frame_interval=1,
    mode="train",
    state_type=state_type,
)

bridge_gen_dataset = L(ActionConditionedBridgeDataset)(
    data_path=base_path,
    num_frames=21, #13
    video_size=[256, 256], # [480, 480]
    val_start_frame_interval=1,
    mode="generate",
    finetune=False,
    state_type=state_type,
)

bridge_val_dataset = L(ActionConditionedBridgeDataset)(
    data_path=base_path,
    num_frames=21, #13
    video_size=[256, 256], # [480, 480]
    val_start_frame_interval=1,
    mode="val",
    state_type=state_type,
)


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


bridge_train_dataloader = L(DataLoader)(
    dataset=bridge_train_dataset,
    sampler=L(get_sampler)(dataset=bridge_train_dataset),
    batch_size=24,
    num_workers=24,
    drop_last=True,
)

bridge_val_dataloader = L(DataLoader)(
    dataset=bridge_val_dataset,
    sampler=L(get_sampler)(dataset=bridge_val_dataset),
    batch_size=1,
    drop_last=False,
)

bridge_gen_dataloader = L(DataLoader)(
    dataset=bridge_gen_dataset,
    sampler=L(get_sampler)(dataset=bridge_gen_dataset),
    batch_size=1,
    drop_last=False,
)

def register_training_and_val_data_action_conditioned():
    cs = ConfigStore.instance()

    # for local dataset
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="bridge_train",
        node=bridge_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="bridge_val",
        node=bridge_val_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="bridge_gen",
        node=bridge_gen_dataloader,
    )
