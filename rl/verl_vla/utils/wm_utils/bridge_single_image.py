from __future__ import annotations

import numpy as np


def prepare_bridge_wm_inputs(
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    total_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack one real Bridge view and one internal zero-valued dummy view."""
    if observations.ndim != 4 or observations.shape[-1] != 3:
        raise ValueError(
            f"observations must have shape [B, H, W, 3], got {observations.shape}"
        )
    if actions.ndim != 3:
        raise ValueError(f"actions must have shape [B, T, A], got {actions.shape}")
    if observations.shape[0] != actions.shape[0]:
        raise ValueError(
            "observations and actions must have the same batch size, got "
            f"{observations.shape[0]} and {actions.shape[0]}"
        )
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")

    batch_size, height, width, channels = observations.shape
    videos = np.zeros(
        (batch_size, 2 * total_frames, height, width, channels),
        dtype=observations.dtype,
    )
    videos[:, 0] = observations

    dummy_actions = np.zeros_like(actions)
    dual_actions = np.concatenate([actions, dummy_actions], axis=-1)
    return videos, dual_actions
