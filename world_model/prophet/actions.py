import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cosmos_predict2.data.action_conditioned.dataset_utils import euler2rotm, rotm2euler

ACTION_DIM_PER_ARM = 7
MAX_ARMS = 2
ACTION_DIM = ACTION_DIM_PER_ARM * MAX_ARMS
LIBERO_SERVO_SCALE = np.asarray(
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1.0] * MAX_ARMS,
    dtype=np.float32,
)


def _quaternions_to_euler(quaternions: np.ndarray) -> np.ndarray:
    shape = quaternions.shape
    rotations = Rotation.from_quat(quaternions.reshape(-1, 4))
    return rotations.as_euler("xyz").reshape(*shape[:-1], 3)


def normalize_action_shape(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions)
    if actions.ndim > 2:
        actions = actions.reshape(actions.shape[0], -1)
    if actions.ndim != 2:
        raise ValueError(f"Expected a 2D action array, got {actions.shape}")
    if actions.shape[-1] == ACTION_DIM_PER_ARM:
        actions = np.pad(actions, ((0, 0), (0, ACTION_DIM_PER_ARM)))
    if actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected 7D or 14D actions, got {actions.shape}")
    return actions.astype(np.float32, copy=False)


def libero_servo_actions(actions: np.ndarray) -> np.ndarray:
    actions = normalize_action_shape(actions).copy()
    actions[:, 6] = (1.0 - actions[:, 6]) / 2.0
    return actions * LIBERO_SERVO_SCALE


def actions_from_poses(data: np.lib.npyio.NpzFile) -> np.ndarray:
    positions = np.asarray(data["end_position"])
    rotations = _quaternions_to_euler(np.asarray(data["end_orientation"]))
    grippers = np.asarray(data["effector_position"])

    if positions.ndim == 2:
        positions = positions[:, None]
        rotations = rotations[:, None]
    if grippers.ndim == 1:
        grippers = grippers[:, None]

    num_frames = positions.shape[0]
    num_arms = min(positions.shape[1], MAX_ARMS)
    actions = np.zeros((num_frames - 1, ACTION_DIM), dtype=np.float32)
    for arm_index in range(num_arms):
        offset = arm_index * ACTION_DIM_PER_ARM
        for frame_index in range(1, num_frames):
            previous_rotation = euler2rotm(rotations[frame_index - 1, arm_index])
            current_rotation = euler2rotm(rotations[frame_index, arm_index])
            actions[frame_index - 1, offset : offset + 3] = previous_rotation.T @ (
                positions[frame_index, arm_index] - positions[frame_index - 1, arm_index]
            )
            actions[frame_index - 1, offset + 3 : offset + 6] = rotm2euler(
                previous_rotation.T @ current_rotation
            )
            actions[frame_index - 1, offset + 6] = np.asarray(
                grippers[frame_index, arm_index]
            ).reshape(-1)[0]

    gripper_columns = actions[:, [6, 13]]
    if gripper_columns.size and np.max(np.abs(gripper_columns)) <= 0.05:
        actions[:, [6, 13]] = np.clip(gripper_columns / 0.04, 0, 1)
    return actions


def load_actions(annotation_path: str | Path, action_format: str = "auto") -> np.ndarray:
    path = Path(annotation_path)
    if path.suffix.lower() == ".json":
        with path.open() as file:
            data = json.load(file)
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if action_format == "libero-servo":
                return libero_servo_actions(actions)
            return normalize_action_shape(actions)
        action_ee = np.asarray(data["action"])[:, :6] * 20
        gripper = np.asarray(data["continuous_gripper_state"])[1:, None]
        return normalize_action_shape(np.concatenate([action_ee, gripper], axis=1))

    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            if action_format == "pose-delta":
                return actions_from_poses(data)
            if "actions" in data:
                actions = np.asarray(data["actions"])
                if action_format == "libero-servo" or (action_format == "auto" and actions.ndim == 3):
                    return libero_servo_actions(actions)
                return normalize_action_shape(actions)
            if "action" in data:
                return normalize_action_shape(data["action"])
            required = {"end_position", "end_orientation", "effector_position"}
            missing = required.difference(data.files)
            if missing:
                raise KeyError(f"Missing action annotation keys: {sorted(missing)}")
            return actions_from_poses(data)

    raise ValueError(f"Unsupported action annotation format: {path.suffix}")
