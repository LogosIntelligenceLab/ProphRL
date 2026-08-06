# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path


def env_path(name: str, default: str = "") -> str:
    value = os.environ.get(name, default)
    return str(Path(value).expanduser()) if value else ""


BRIDGE_DATA = env_path("PROPHRL_BRIDGE_DATA")
LIBERO_DATA = env_path("PROPHRL_LIBERO_DATA")
PRETRAIN_CHECKPOINT = env_path("PROPHRL_PRETRAIN_CKPT")
RESUME_CHECKPOINT = env_path("PROPHRL_RESUME_CKPT")
