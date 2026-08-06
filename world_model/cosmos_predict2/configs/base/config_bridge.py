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

from typing import Any

import attrs

from cosmos_predict2.configs.action_conditioned.defaults.data_bridge import register_training_and_val_data_action_conditioned
from cosmos_predict2.configs.action_conditioned.defaults.model_prophet import register_prophet_model
from cosmos_predict2.configs.action_conditioned.paths import PRETRAIN_CHECKPOINT
from cosmos_predict2.configs.base.defaults.callbacks import register_callbacks
from cosmos_predict2.configs.base.defaults.checkpoint import register_checkpoint
from cosmos_predict2.configs.base.defaults.ema import register_ema
from cosmos_predict2.configs.base.defaults.optimizer import register_optimizer
from cosmos_predict2.configs.base.defaults.scheduler import register_scheduler
from imaginaire import config
from imaginaire.trainer import ImaginaireTrainer as Trainer
from imaginaire.utils.config_helper import import_all_modules_from_package


@attrs.define(slots=False)
class Config(config.Config):
    # default config groups that will be used unless overwritten
    # see config groups in registry.py
    defaults: list[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"dataloader_train": None},
            {"dataloader_val": None},
            {"optimizer": "fusedadamw"},
            {"scheduler": "constant"},
            {"model": None},
            {"callbacks": ["basic"]},
            {"net": None},
            {"ema": None},
            {"checkpoint": None},
            {"ckpt_type": None},
            # the list is with order, we need global experiment to be the last one
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # Specifying values through instances of attrs
    c.job.project = "prophrl_world_model"
    c.job.group = "finetune"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.max_iter = 30_000
    c.trainer.logging_iter = 10
    c.trainer.validation_iter = 2000
    c.trainer.max_val_iter = 20
    c.trainer.run_validation = True
    c.trainer.callbacks = None

    # change checkpointing and resume path
    c.checkpoint.save_iter = 2000
    c.checkpoint.load_training_state = False
    c.checkpoint.load_path = PRETRAIN_CHECKPOINT

    # Call this function to register config groups for advanced overriding. the order follows the default config groups
    register_optimizer()
    register_scheduler()

    register_ema()
    register_checkpoint()
    register_callbacks()

    # action conditional post-training config
    register_training_and_val_data_action_conditioned()
    register_prophet_model()

    # experiment config are defined in the experiment folder
    # call import_all_modules_from_package to register them
    import_all_modules_from_package("cosmos_predict2.configs.action_conditioned.experiment", reload=True)
    return c
