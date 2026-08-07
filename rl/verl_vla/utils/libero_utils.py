"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
import numpy as np
try:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ImportError as exc:
    get_libero_path = OffScreenRenderEnv = None
    _LIBERO_IMPORT_ERROR = exc
else:
    _LIBERO_IMPORT_ERROR = None
import random
import torch

import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image
import io
# from experiments.robot.robot_utils import (
#     DATE,
#     DATE_TIME,


def get_libero_env(task, model_family, resolution=256, gpu_id=0):
    """Initializes and returns the LIBERO environment, along with the task description."""
    if _LIBERO_IMPORT_ERROR is not None:
        raise ImportError("LIBERO is required to create a simulation environment") from _LIBERO_IMPORT_ERROR

    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file, 
        "camera_heights": resolution, 
        "camera_widths": resolution,
        "render_gpu_device_id": gpu_id
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


# def resize_image(img, resize_size):
#     """
#     Takes numpy array corresponding to a single image and returns resized image as numpy array.

#     NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
#                     the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
#     """

#     # Resize to image size expected by model

def resize_image(img: np.ndarray, resize_size: tuple):
    """
    接受一个 numpy 数组图像，并返回调整大小后的 numpy 数组。
    这是一个简化的 PyTorch 版本，移除了 JPEG 编解码步骤以提高效率。
    """
    assert isinstance(resize_size, tuple), "resize_size 必须是一个元组"

    # 如果输入的 numpy 数组是 (1, H, W, C) 格式，先移除批次维度
    if img.ndim == 4 and img.shape[0] == 1:
        img = img.squeeze(0)

    # --- 图像尺寸调整 ---
    # 将 (H, W, C) 的 numpy 数组转换为 (C, H, W) 的 PyTorch 张量
    # F.resize 需要一个 float 类型的输入来进行抗锯齿处理
    tensor_img = torch.from_numpy(img.copy()).permute(2, 0, 1).to(torch.float32)

    # 使用 "lanczos3" 插值和抗锯齿调整图像大小
    # 此时的 tensor_img 保证是 3D 的 [C, H, W]
    resized_tensor = F.resize(
        tensor_img,
        list(resize_size),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True
    )

    # --- 后处理 ---
    # 将结果四舍五入，裁剪到 [0, 255] 范围，并转换回 uint8
    final_tensor = resized_tensor.round().clamp(0, 255).to(torch.uint8)

    # 将维度顺序从 (C, H, W) 转换回 (H, W, C) 以匹配 numpy 的格式
    numpy_img = final_tensor.permute(1, 2, 0).numpy()

    return numpy_img


def get_libero_image(obs, resize_size, flip=False):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    if flip:
        img = img[::-1]
    else:
        img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def get_libero_wrist_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1]
    img = resize_image(img, resize_size)
    return img

# def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
#     """Saves an MP4 replay of an episode."""
#     for img in rollout_images:
#     if log_file is not None:


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    if cfg.model_family == "openvla":
        resize_size = 224
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size






# def normalize_gripper_action(action, binarize=True):
#     """
#     Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
#     Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
#     Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
#     the dataset wrapper.

#     Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
#     """
#     # Just normalize the last action to [-1,+1].

#     if binarize:
#         # Binarize to -1 or +1.


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


# def invert_gripper_action(action):
#     """
#     Flips the sign of the gripper action (last dimension of action vector).
#     This is necessary for some environments where -1 = open, +1 = close, since
#     the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
#     """

def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] =inverted_action[..., -1] *  -1.0

    return inverted_action

def save_rollout_video(rollout_images, exp_name, task_name, step_idx, success, finish_step=-1):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{exp_name}" 
    os.makedirs(rollout_dir, exist_ok=True)
    ran_id = random.randint(1, 10000)
    if finish_step == -1:
        mp4_path = f"{rollout_dir}/step={step_idx}--task={task_name}--success={success}--ran={ran_id}.mp4"
    else:
        mp4_path = f"{rollout_dir}/step={step_idx}--task={task_name}--success={success}--finish_step={finish_step}--ran={ran_id}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path
