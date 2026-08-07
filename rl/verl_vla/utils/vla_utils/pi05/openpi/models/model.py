import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


def _load_pytorch_checkpoint(model: torch.nn.Module, weight_path: str) -> None:
    with safetensors.safe_open(weight_path, framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
        shared_tensors = checkpoint.metadata() or {}

    if keys and all(key.startswith("model.") for key in keys):
        state_dict = {
            key.removeprefix("model."): value
            for key, value in safetensors.torch.load_file(weight_path).items()
        }
        for missing_key, stored_key in shared_tensors.items():
            state_dict[missing_key.removeprefix("model.")] = state_dict[stored_key.removeprefix("model.")]
        model.load_state_dict(state_dict, strict=True)
        return

    safetensors.torch.load_model(model, weight_path)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
#     # Observation data.
#         ...  # Additional camera views
#     },
#         ...  # Masks for additional views
#     },
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#


def _to_float_chw(x):
    """Return tensor/ndarray in float32, channel-first, normalized to [-1,1]."""
    # ---- torch 张量 ----
    if torch.is_tensor(x):
        if x.is_sparse:
            x = x.to_dense()
        # 转 dtype
        needs_div255 = x.dtype == torch.uint8
        x = x.to(torch.float32)

        # 维度整理：支持 3D(HWC)、4D(BC*HW/NCHW/NHWC)、5D(BV*HW/BVCHW/BVHW*C)
        if x.ndim == 3:  # HWC -> CHW，加一个 batch 维度
            if x.shape[-1] in (1, 3):
                x = x.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
        elif x.ndim == 4:
            # NHWC -> NCHW
            if x.shape[-1] in (1, 3) and (x.shape[1] not in (1, 3)):
                x = x.permute(0, 3, 1, 2)
        elif x.ndim == 5:
            # B,V,H,W,C -> B,V,C,H,W
            if x.shape[-1] in (1, 3) and (x.shape[2] not in (1, 3)):
                x = x.permute(0, 1, 4, 2, 3)
        else:
            raise ValueError(f"Unsupported ndim for torch tensor: {x.ndim}")

        # 归一化：uint8 或看起来像 [0,255] 的 float -> [0,1]
        if needs_div255 or (x.numel() > 0 and torch.isfinite(x).any() and x.max() > 1.5):
            x = x / 255.0
        # [0,1] -> [-1,1]
        x = x * 2.0 - 1.0
        return x

    # ---- numpy 数组 ----
    if isinstance(x, np.ndarray):
        needs_div255 = x.dtype == np.uint8
        x = x.astype(np.float32, copy=False)

        if x.ndim == 3:  # HWC -> CHW，加一个 batch 维度
            if x.shape[-1] in (1, 3):
                x = np.moveaxis(x, -1, 0)[None, ...]  # 1,C,H,W
        elif x.ndim == 4:
            # NHWC -> NCHW
            if x.shape[-1] in (1, 3) and (x.shape[1] not in (1, 3)):
                x = np.moveaxis(x, -1, 1)
        elif x.ndim == 5:
            # B,V,H,W,C -> B,V,C,H,W
            if x.shape[-1] in (1, 3) and (x.shape[2] not in (1, 3)):
                x = np.moveaxis(x, -1, 2)
        else:
            raise ValueError(f"Unsupported ndim for numpy array: {x.ndim}")

        if needs_div255 or (np.isfinite(x).any() and x.max() > 1.5):
            x = x / 255.0
        x = x * 2.0 - 1.0
        return x

    raise TypeError(f"Unsupported image type: {type(x)}")

@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, torch.Tensor]
    # Image masks, with same keys as images.
    image_masks: dict[str, torch.Tensor]
    # Low-dimensional robot state.
    state: torch.Tensor

    # Tokenized prompt.
    tokenized_prompt: torch.Tensor | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: torch.Tensor | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: torch.Tensor | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: torch.Tensor | None = None

    # @classmethod
    # def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
    #     # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
    #     if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
    #     # If images are uint8, convert them to [-1, 1] float32.
    #     for key in data["image"]:
    #         if data["image"][key].dtype == np.uint8:
    #     return cls(

    @classmethod
    def from_dict(cls, data):
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")

        if "image" in data and isinstance(data["image"], dict):
            for key, img in list(data["image"].items()):
                data["image"][key] = _to_float_chw(img)

        return cls(
            images=data.get("image"),
            image_masks=data.get("image_mask"),
            state=data.get("state"),
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )
        
    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = torch.Tensor


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        _load_pytorch_checkpoint(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
