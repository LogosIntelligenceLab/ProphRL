from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_ROOT = REPO_ROOT / "rl"
WORLD_MODEL_ROOT = REPO_ROOT / "world_model"


def test_world_model_wrapper_imports_from_repository_runtime() -> None:
    python_paths = [str(RL_ROOT), str(WORLD_MODEL_ROOT)]
    if openpi_client_src := os.environ.get("OPENPI_CLIENT_SRC"):
        python_paths.append(openpi_client_src)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from verl_vla.utils.wm_utils.cosmos_world_model "
                "import CosMosWorldModel; print(CosMosWorldModel.__name__)"
            ),
        ],
        cwd=RL_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "CosMosWorldModel" in result.stdout


def test_single_image_adapter_adds_only_an_internal_dummy_view() -> None:
    helper_path = RL_ROOT / "verl_vla/utils/wm_utils/bridge_single_image.py"
    spec = importlib.util.spec_from_file_location("bridge_single_image", helper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prepare_bridge_wm_inputs = module.prepare_bridge_wm_inputs

    observations = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)
    actions = np.arange(2 * 3 * 7, dtype=np.float32).reshape(2, 3, 7)

    videos, dual_actions = prepare_bridge_wm_inputs(
        observations,
        actions,
        total_frames=21,
    )

    assert videos.shape == (2, 42, 4, 5, 3)
    np.testing.assert_array_equal(videos[:, 0], observations)
    assert np.count_nonzero(videos[:, 1:]) == 0

    assert dual_actions.shape == (2, 3, 14)
    np.testing.assert_array_equal(dual_actions[..., :7], actions)
    assert np.count_nonzero(dual_actions[..., 7:]) == 0


def test_attention_accepts_per_view_history_memory() -> None:
    sys.path.insert(0, str(WORLD_MODEL_ROOT))
    from cosmos_predict2.models.text2image_dit import Attention

    attention = Attention(8, n_heads=2, head_dim=4, backend="torch")
    attention.q_norm = torch.nn.Identity()
    attention.k_norm = torch.nn.Identity()
    attention.v_norm = torch.nn.Identity()
    captured: dict[str, torch.Size] = {}

    def capture_attention(self, q, k, v, video_size=None):
        captured.update(q=q.shape, k=k.shape, v=v.shape)
        return torch.zeros(q.shape[0], q.shape[1], self.query_dim)

    attention.compute_attention = types.MethodType(capture_attention, attention)
    x = torch.randn(1, 4, 8)
    history = torch.randn(1, 2, 3, 8)

    output = attention(x, k_mem=history, v_mem=history)

    assert output.shape == (1, 4, 8)
    assert captured == {
        "q": torch.Size([1, 4, 2, 4]),
        "k": torch.Size([1, 10, 2, 4]),
        "v": torch.Size([1, 10, 2, 4]),
    }
