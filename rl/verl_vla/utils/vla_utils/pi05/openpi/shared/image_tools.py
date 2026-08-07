import functools

import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F  # noqa: N812

import openpi.shared.array_typing as at
from functools import reduce
from operator import mul
from typing import Tuple

@functools.partial(jax.jit, static_argnums=(1, 2, 3))
@at.typecheck
def resize_with_pad(
    images: at.UInt8[at.Array, "*b h w c"] | at.Float[at.Array, "*b h w c"],
    height: int,
    width: int,
    method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR,
) -> at.UInt8[at.Array, "*b {height} {width} c"] | at.Float[at.Array, "*b {height} {width} c"]:
    """Replicates tf.image.resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].
    """
    if images.ndim < 3:
        raise ValueError(f"Expected [..., height, width, channels], got shape {images.shape}")

    batch_shape = images.shape[:-3]
    cur_height, cur_width, channels = images.shape[-3:]
    flat_images = images.reshape((-1, cur_height, cur_width, channels))
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_images = jax.image.resize(
        flat_images, (flat_images.shape[0], resized_height, resized_width, channels), method=method
    )
    if images.dtype == jnp.uint8:
        # round from float back to uint8
        resized_images = jnp.round(resized_images).clip(0, 255).astype(jnp.uint8)
    elif images.dtype == jnp.float32:
        resized_images = resized_images.clip(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    padded_images = jnp.pad(
        resized_images,
        ((0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        constant_values=0 if images.dtype == jnp.uint8 else -1.0,
    )

    return padded_images.reshape((*batch_shape, height, width, channels))


# def resize_with_pad_torch(
#     images: torch.Tensor,
#     height: int,
#     width: int,
# ) -> torch.Tensor:
#     """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
#     by padding with black. If the image is float32, it must be in the range [-1, 1].

#     Args:
#         images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
#         height: Target height
#         width: Target width
#         mode: Interpolation mode ('bilinear', 'nearest', etc.)

#     Returns:
#         Resized and padded tensor with same shape format as input
#     """
#     # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
#     if images.shape[-1] <= 4:  # Assume channels-last format
#         # Convert to channels-first for torch operations
#         if images.dim() == 3:
#         if images.dim() == 3:


#     # Calculate resize ratio

#     # Resize

#     # Handle dtype-specific clipping
#     if images.dtype == torch.uint8:

#     # Calculate padding

#     # Pad
#         resized_images,

#     # Convert back to original format if needed
#     if channels_last:
#         if batch_size == 1 and images.shape[0] == 1:



def _infer_channels_last(x: torch.Tensor) -> bool:
    """猜测通道是否在最后一维。支持 >=3 维输入。"""
    if x.ndim < 3:
        raise ValueError(f"images must have at least 3 dims, got {x.ndim}")
    # 典型判断：最后一维是 1/3/4 且 倒数第三维不像通道
    if x.shape[-1] in (1, 3, 4):
        return True
    # 若通道在倒数第三维
    if x.shape[-3] in (1, 3, 4):
        return False
    # 默认按通道在最后一维处理
    return True

def _prod(shape) -> int:
    return reduce(mul, shape, 1)

def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    支持输入形状 [*b, H, W, C] 或 [*b, C, H, W]，返回与输入相同的格式与维度数。
    - uint8 认为范围是 [0,255]，pad 为 0
    - float（建议 float32）认为范围是 [-1,1]，pad 为 -1
    """
    if images.ndim < 3:
        raise ValueError(f"images must have at least 3 dims, got {images.ndim}")

    orig_dtype = images.dtype
    channels_last = _infer_channels_last(images)
    orig_shape = images.shape  # 保留以便还原

    if channels_last:
        # [*b, H, W, C] -> [*b, C, H, W]
        perm = list(range(images.ndim))
        # 把最后一维(C)移到倒数第三位
        perm = perm[:-3] + [images.ndim - 1, images.ndim - 3, images.ndim - 2]
        images_cf = images.permute(*perm)
    else:
        images_cf = images

    # 拆出前置 batch 维，展平为 Bflat
    bflat = _prod(images_cf.shape[:-3])
    C, H, W = images_cf.shape[-3], images_cf.shape[-2], images_cf.shape[-1]
    images_cf = images_cf.reshape(bflat, C, H, W)

    # interpolate 不支持 uint8，先转 float32
    work = images_cf.to(torch.float32)

    # 计算等比缩放后的尺寸
    ratio = max(W / float(width), H / float(height))
    new_h = max(int(round(H / ratio)), 1)
    new_w = max(int(round(W / ratio)), 1)

    # 选择 align_corners
    align = False if mode in ("bilinear", "bicubic") else None
    resized = F.interpolate(work, size=(new_h, new_w), mode=mode, align_corners=align)

    # 计算四边 padding（黑边/-1 边）
    pad_h0, rh = divmod(height - new_h, 2)
    pad_h1 = pad_h0 + rh
    pad_w0, rw = divmod(width - new_w, 2)
    pad_w1 = pad_w0 + rw

    pad_val = 0.0 if orig_dtype == torch.uint8 else -1.0
    padded = F.pad(resized, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=pad_val)

    # 恢复 dtype & 合理裁剪/量化
    if orig_dtype == torch.uint8:
        padded = torch.round(padded).clamp(0, 255).to(torch.uint8)
    else:
        # 假定 float 输入为 [-1,1]，保持范围
        padded = padded.clamp(-1.0, 1.0).to(orig_dtype)

    padded = padded.reshape(*orig_shape[:-3], C, height, width)

    # 若原来是 channels-last，则再换回去 [*b, H, W, C]
    if channels_last:
        # [*b, C, H, W] -> [*b, H, W, C]
        perm_back = list(range(padded.ndim))
        # 把倒数第三位(C)移到最后
        perm_back = perm_back[:-3] + [padded.ndim - 2, padded.ndim - 1, padded.ndim - 3]
        padded = padded.permute(*perm_back)

    return padded
