# Data Preparation and Layout

## Processed Layout

Each processed dataset root should contain paired video clips and NumPy annotations:

```text
dataset_root/
├── clips/
│   ├── episode_or_clip_00000.mp4
│   └── ...
├── metadata/
│   ├── episode_or_clip_00000.npz
│   └── ...
├── blacks/                  # optional projected action-frame videos
│   ├── episode_or_clip_00000_black.mp4
│   └── ...
├── dataset.json             # required by the BRIDGE task-index builder
└── task_info.json           # generated/consumed by BRIDGE and LIBERO tooling
```

The relative stem under `clips/` and `metadata/` must match. Optional action-frame videos use the same stem plus `_black.mp4` under `blacks/`.

## Action Format

Prophet uses a shared dual-arm action representation. Each arm has seven values per step:

```text
[delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper_open]
```

The combined action is 14D. Single-arm datasets are padded with seven zeros for the missing arm. Gripper values are normalized to `[0, 1]`, where `0` is closed and `1` is open.

For a 20-frame prediction chunk, the model receives 20 actions and the DiT action-conditioning width is `14 × 20 = 280`.

## Annotation Keys

The dataset adapters support dataset-specific fields. The common public inference path accepts:

- `actions`: precomputed 7D or 14D action sequence.
- `action`: alternative precomputed action key.
- `end_position`: absolute end-effector positions with shape `[T, N, 3]`.
- `end_orientation`: quaternions with shape `[T, N, 4]` in SciPy `xyzw` order.
- `effector_position`: gripper state or width with shape `[T, N, ...]`.

LIBERO training annotations are expected to include `actions` and pose/camera fields used by optional action-frame construction. BRIDGE annotations use `obs_state` and/or `action` with position-and-Euler end-effector states.

## Environment Variables

```bash
export PROPHRL_BRIDGE_DATA=/datasets/bridge_processed
export PROPHRL_LIBERO_DATA=/datasets/libero_processed
```

The launch scripts validate required roots before starting distributed training.

## Convert BRIDGE

The BRIDGE converter reads the official TFDS/RLDS builder directory and writes one paired MP4/NPZ sample per eligible episode:

```bash
python -m pip install tensorflow==2.15.1 tensorflow-datasets==4.9.4

prophet-prepare-bridge \
  --input /path/to/bridge_orig/1.0.0 \
  --output /path/to/processed/bridge_orig

prophet-validate-data /path/to/processed/bridge_orig
```

Use `--limit 10` for a conversion smoke test. The converter uses `image_0` as the primary camera, matching the released BRIDGE recipe.

## Convert LIBERO

The official LIBERO HDF5 demonstrations already contain agent-view RGB frames, servo actions, end-effector poses, and gripper state. Convert one file or a directory recursively:

```bash
prophet-prepare-libero \
  --input /path/to/libero/demonstrations \
  --output /path/to/processed/libero \
  --clip-length 64 \
  --stride 16

prophet-validate-data /path/to/processed/libero
```

The public LIBERO fine-tuning recipe uses the same servo preprocessing as the selected training run: translational and rotational actions are multiplied by `0.05`, and the first-arm gripper command is mapped from `[-1, 1]` to `[1, 0]`.

The converter produces the data needed by that servo fine-tuning recipe. It does not synthesize the optional projected action-frame videos under `blacks/`; those require the LIBERO simulator camera calibration. If publishing the historical processed dataset, preserve its existing `blacks/` directory.

`dataset.json` entries produced by both converters use repository-relative clip paths, so moving or uploading the processed root does not leave private absolute paths behind.
