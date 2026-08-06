# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from hydra.core.config_store import ConfigStore


def _experiment(*, dataloader_train: str, dataloader_val: str | None, batch_size: int, group: str, name: str):
    defaults = [
        {"override /model": "prophet_2b_fsdp"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": dataloader_train},
    ]
    if dataloader_val is not None:
        defaults.append({"override /dataloader_val": dataloader_val})
    defaults.append("_self_")
    return {
        "defaults": defaults,
        "model": {"config": {"train_architecture": "base", "fsdp_shard_size": -1}},
        "job": {"group": group, "name": f"{name}_${{now:%Y-%m-%d}}_${{now:%H-%M-%S}}"},
        "model_parallel": {"context_parallel_size": 1},
        "dataloader_train": {"batch_size": batch_size},
        "trainer": {"distributed_parallelism": "fsdp"},
    }


PROPHET_FINETUNE_BRIDGE = _experiment(
    dataloader_train="bridge_train",
    dataloader_val="bridge_val",
    batch_size=24,
    group="finetune",
    name="prophet_finetune_bridge_2b",
)

PROPHET_FINETUNE_LIBERO = _experiment(
    dataloader_train="libero_train",
    dataloader_val="libero_val",
    batch_size=24,
    group="finetune",
    name="prophet_finetune_libero_2b",
)


def register_experiments() -> None:
    config_store = ConfigStore.instance()
    experiments = {
        "prophet_finetune_bridge": PROPHET_FINETUNE_BRIDGE,
        "prophet_finetune_libero": PROPHET_FINETUNE_LIBERO,
    }
    for name, experiment in experiments.items():
        config_store.store(group="experiment", package="_global_", name=name, node=experiment)


register_experiments()
