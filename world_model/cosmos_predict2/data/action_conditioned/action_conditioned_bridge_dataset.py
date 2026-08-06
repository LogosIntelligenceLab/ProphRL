import json
import os
import random
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from cosmos_predict2.data.action_conditioned.dataset_utils import (
    ToTensorVideo,
    euler2rotm,
    rotm2euler,
)


def save_video_tensor(
    video_thwc: torch.Tensor,
    path: str,
    fps: int = 10,
    codec: str = "libx264",
    bitrate: str | None = None,
) -> str:
    assert video_thwc.ndim == 4 and video_thwc.shape[-1] == 3, \
        f"expect (T,H,W,3), got {tuple(video_thwc.shape)}"

    x = video_thwc.detach()
    if x.is_floating_point():
        x = x.clamp(0, 255).round()
        x = x.to(torch.uint8)
    if x.is_cuda:
        x = x.cpu()
    frames = x.contiguous().numpy()  # (T,H,W,3), uint8, RGB

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    writer_kwargs = dict(fps=fps, codec=codec, macro_block_size=None)
    if bitrate is not None:
        writer_kwargs["bitrate"] = bitrate

    with imageio.get_writer(path, **writer_kwargs) as w:
        for f in frames:
            w.append_data(f)  # f: (H,W,3) uint8 RGB

    return path


def quats_to_euler(
    quats: np.ndarray,
    order: str = "xyz",
    degrees: bool = False,
    eps: float = 1e-8,
    missing: str = "identity",   # "identity" | "nan" | "zero"
    return_mask: bool = False,
):
    q = np.asarray(quats, dtype=np.float64)
    assert q.ndim == 3 and q.shape[-1] == 4 and q.shape[1] == 2, "need to be (N, 2, 4)"
    N = q.shape[0]

    norms = np.linalg.norm(q, axis=-1)             # (N, 2)
    mask_missing = (norms < eps) | ~np.isfinite(norms)

    q_flat = q.reshape(-1, 4)
    norms_flat = norms.reshape(-1)
    bad_flat = mask_missing.reshape(-1)
    good_idx = np.where(~bad_flat)[0]

    euler_flat = np.empty((q_flat.shape[0], 3), dtype=np.float64)

    if good_idx.size > 0:
        qn = q_flat[good_idx] / np.clip(norms_flat[good_idx, None], eps, None)
        euler_flat[good_idx] = R.from_quat(qn).as_euler(order, degrees=degrees)

    if bad_flat.any():
        if missing == "identity":
            id_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)[None, :]
            id_euler = R.from_quat(id_quat).as_euler(order, degrees=degrees)[0]
            euler_flat[bad_flat] = id_euler
        elif missing == "nan":
            euler_flat[bad_flat] = np.nan
        elif missing == "zero":
            euler_flat[bad_flat] = 0.0
        else:
            raise ValueError("missing must be one of {'identity','nan','zero'}")

    euler = euler_flat.reshape(N, 2, 3)
    return (euler, mask_missing) if return_mask else euler

class ResizePadCrop:
    def __init__(self, target_h=240, target_w=320, interp="bilinear"):
        self.target_h = int(target_h)
        self.target_w = int(target_w)
        self.interp = interp

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        assert vid.ndim == 4, f"expect (T,C,H,W), got {tuple(vid.shape)}"
        T, C, H, W = vid.shape
        if H == 0 or W == 0:
            raise ValueError(f"invalid video size: H={H}, W={W}")

        orig_dtype = vid.dtype
        x = vid.to(dtype=torch.float32)

        scale = self.target_h / float(H)
        new_w = max(1, int(round(W * scale)))
        x = F.interpolate(
            x, size=(self.target_h, new_w),
            mode=self.interp, align_corners=False if self.interp in ("bilinear", "bicubic") else None
        )
        if new_w < self.target_w:
            pad_total = self.target_w - new_w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x = F.pad(x, (pad_left, pad_right, 0, 0), mode="constant", value=0.0)
        elif new_w > self.target_w:
            start = (new_w - self.target_w) // 2
            end = start + self.target_w
            x = x[:, :, :, start:end]

        assert x.shape[2] == self.target_h and x.shape[3] == self.target_w, x.shape
        if orig_dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
            x = x.clamp(0, 255).to(orig_dtype)
        return x

def is_all_black_video(path: str, sample_every: int = 10, thresh: int = 3) -> bool:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return True
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total == 0:
        cap.release()
        return True
    idxs = range(0, total, max(1, sample_every))
    ok_all_black = True
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if np.max(frame) > thresh:
            ok_all_black = False
            break
    cap.release()
    return ok_all_black

def build_task_info(data_path: str, save_name: str = "task_info.json") -> dict:
    task_path = os.path.join(data_path, "dataset.json")
    with open(task_path, "r") as f:
        task_file = json.load(f)

    task_info = {}
    dropped_missing, dropped_black = 0, 0

    for item in tqdm(task_file):
        caption = item["caption"]
        media_path = item["media_path"]
        if not os.path.isabs(media_path):
            media_path = os.path.join(data_path, media_path)
        meta_path = media_path.replace("clips", "metadata").replace(".mp4", ".npz")

        episode = meta_path.split("_")[-3]
        caption_with_episode = f"{caption}-{episode}"
        if ("wrist" in meta_path) or ("hand" in meta_path):
            continue

        if not os.path.exists(media_path):
            dropped_missing += 1
            continue
        if is_all_black_video(media_path):
            dropped_black += 1
            continue

        task_info.setdefault(caption_with_episode, []).append(meta_path)

    save_path = os.path.join(data_path, save_name)
    with open(save_path, "w") as f:
        json.dump(task_info, f, ensure_ascii=False, indent=2)

    print(
        f"[build_task_info] kept_groups={len(task_info)} "
        f"dropped_missing={dropped_missing} dropped_black={dropped_black} -> {save_path}"
    )
    return task_info

# action augmentation
def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi

@dataclass
class AugCfgFixedLen:
    p_noise: float = 1.0
    p_drift: float = 0.5
    p_speed: float = 0.5
    p_delay: float = 0.3
    p_drop: float = 0.3
    p_freeze: float = 0.3
    p_impulse: float = 0.3
    p_saturate: float = 0.5
    p_g_sticky: float = 0.5
    p_g_spurious: float = 0.3
    p_g_failclose: float = 0.3
    pos_noise_std: float = 2e-3
    ang_noise_std: float = np.deg2rad(1.0)
    pos_drift_std: float = 5e-4
    ang_drift_std: float = np.deg2rad(0.5)
    speed_range: tuple = (0.6, 1.3)
    delay_range: tuple = (1, 3)
    drop_span: tuple = (3, 8)
    freeze_span: tuple = (3, 8)
    pos_impulse_std: float = 2e-2
    ang_impulse_std: float = np.deg2rad(5.0)
    pos_sat_abs: float = 3e-3
    ang_sat_abs: float = np.deg2rad(1.5)
    sticky_delay: tuple = (1, 4)
    spurious_window: tuple = (5, 18)

def augment_delta(
    delta20: np.ndarray,   # (20, 7)
    cfg: AugCfgFixedLen = AugCfgFixedLen(),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    assert delta20.shape == (20, 7), f"expect (20,7), got {delta20.shape}"
    if rng is None:
        rng = np.random.default_rng()
    out = delta20.copy()
    Tm1 = 20

    def coin(p): return rng.random() < p
    def span(lohi):
        lo, hi = lohi
        hi = max(hi, lo)
        return int(rng.integers(lo, hi + 1))

    if coin(cfg.p_noise):
        out[:, :3] += rng.normal(0, cfg.pos_noise_std, size=(Tm1, 3))
        out[:, 3:6] = wrap_to_pi(out[:, 3:6] + rng.normal(0, cfg.ang_noise_std, size=(Tm1, 3)))

    if coin(cfg.p_drift):
        steps = np.arange(Tm1).reshape(-1, 1)
        out[:, :3] += steps * rng.normal(0, cfg.pos_drift_std, size=(1, 3))
        out[:, 3:6] = wrap_to_pi(out[:, 3:6] + steps * rng.normal(0, cfg.ang_drift_std, size=(1, 3)))

    if coin(cfg.p_speed):
        s = rng.uniform(*cfg.speed_range)
        out[:, :6] *= s

    if coin(cfg.p_delay):
        d = span(cfg.delay_range)
        if d > 0:
            out[:-d, :6] = out[d:, :6]
            out[-d:, :6] = 0.0

    if coin(cfg.p_drop):
        L = min(span(cfg.drop_span), Tm1)
        s = int(rng.integers(0, Tm1 - L + 1))
        out[s:s+L, :6] = 0.0

    if coin(cfg.p_freeze):
        L = min(span(cfg.freeze_span), Tm1)
        s = int(rng.integers(0, Tm1 - L + 1))
        ref = out[max(0, s-1), :6].copy()
        out[s:s+L, :6] = ref

    if coin(cfg.p_impulse):
        t = int(rng.integers(0, Tm1))
        out[t, :3] += rng.normal(0, cfg.pos_impulse_std, size=(3,))
        out[t, 3:6] = wrap_to_pi(out[t, 3:6] + rng.normal(0, cfg.ang_impulse_std, size=(3,)))

    if coin(cfg.p_saturate):
        out[:, :3] = np.clip(out[:, :3], -cfg.pos_sat_abs, cfg.pos_sat_abs)
        out[:, 3:6] = np.clip(out[:, 3:6], -cfg.ang_sat_abs, cfg.ang_sat_abs)

    g = out[:, 6].copy()

    if coin(cfg.p_g_sticky):
        delay = span(cfg.sticky_delay)
        if delay > 0:
            g_del = g.copy()
            g_del[delay:] = g[:-delay]
            g_del[:delay] = g[0]
            g = g_del

    if coin(cfg.p_g_spurious):
        lo, hi = cfg.spurious_window
        lo = max(0, min(lo, Tm1-1))
        hi = max(lo+1, min(hi, Tm1))
        t = int(rng.integers(lo, hi))
        g[t] = 1.0 - g[t]

    if coin(cfg.p_g_failclose):
        prev = np.concatenate([g[:1], g[:-1]], axis=0)
        falling = (prev > 0.5) & (g < 0.5)
        idx = np.nonzero(falling)[0]
        if len(idx) > 0:
            j = int(rng.integers(0, len(idx)))
            g[idx[j]] = 1.0

    out[:, 6] = g
    return out

class ActionConditionedBridgeDataset(Dataset):
    def __init__(
        self,
        data_path='',
        sequence_interval=1,
        num_frames=21,
        cam_ids=[0],
        accumulate_action=False,
        video_size=[256, 256],
        val_start_frame_interval=1,
        debug=False,
        normalize=False,
        do_evaluate=False,
        load_t5_embeddings=False,
        load_action=True,
        load_history_video=True,
        history_video_length=60,
        mode="train",
        state_type='none',
        finetune=False,
        action_key='action',
    ):
        """Dataset class for loading 3D robot action-conditioned data.

        This dataset loads robot trajectories consisting of RGB video frames, robot states (arm positions and gripper states),
        and computes relative actions between consecutive frames.

        Args:
            train_annotation_path (str): Path to training annotation files
            val_annotation_path (str): Path to validation annotation files
            test_annotation_path (str): Path to test annotation files
            video_path (str): Base path to video files
            sequence_interval (int): Interval between sampled frames in a sequence
            num_frames (int): Number of frames to load per sequence
            cam_ids (list): List of camera IDs to sample from
            accumulate_action (bool): Whether to accumulate actions relative to first frame
            video_size (list): Target size [H,W] for video frames
            val_start_frame_interval (int): Frame sampling interval for validation/test
            debug (bool, optional): If True, only loads subset of data. Defaults to False.
            normalize (bool, optional): Whether to normalize video frames. Defaults to False.
            pre_encode (bool, optional): Whether to pre-encode video frames. Defaults to False.
            do_evaluate (bool, optional): Whether in evaluation mode. Defaults to False.
            load_t5_embeddings (bool, optional): Whether to load T5 embeddings. Defaults to False.
            load_action (bool, optional): Whether to load actions. Defaults to True.
            mode (str, optional): Dataset mode - 'train', 'val' or 'test'. Defaults to 'train'.

        The dataset loads robot trajectories and computes:
        - RGB video frames from specified camera views
        - Robot arm states (xyz position + euler angles)
        - Gripper states (binary open/closed)
        - Relative actions between consecutive frames

        Actions are computed as relative transforms between frames:
        - Translation: xyz offset in previous frame's coordinate frame
        - Rotation: euler angles of relative rotation
        - Gripper: binary gripper state

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - action: Action tensor [T-1,7]
            - video_name: Dict with episode/frame metadata
            - latent: Pre-encoded video features if pre_encode=True
        """

        super().__init__()
        if mode == "train":
            self.start_frame_interval = 5
        elif mode == "val":
            self.start_frame_interval = val_start_frame_interval
        elif mode == 'generate':
            self.start_frame_interval = val_start_frame_interval
            self.aug_cfg = AugCfgFixedLen()

        self.data_path = data_path
        self.video_path = os.path.join(self.data_path, 'clips')
        self.ann_path = os.path.join(self.data_path, 'metadata')
        self.sequence_interval = sequence_interval
        self.mode = mode
        self.state_type = state_type
        self.sequence_length = num_frames
        self.normalize = normalize
        self.load_action = load_action
        self.pre_encode = False
        self.load_history_video = load_history_video
        self.history_video_length = history_video_length

        self.cam_ids = cam_ids
        self.accumulate_action = accumulate_action
        self.load_t5_embeddings = load_t5_embeddings
        self.finetune = finetune
        self.action_key = action_key
        if self.finetune:
            # if not os.path.exists(os.path.join(self.data_path, 'task_info.json')):
            #     task_info = build_task_info(self.data_path, save_name="task_info.json")
            # else:
            #     with open(os.path.join(self.data_path, 'task_info.json'), 'r') as f:
            #         task_info = json.load(f)
            task_info_path = os.path.join(self.data_path, 'dataset.json')
            with open(task_info_path, "r") as f:
                task_file = json.load(f)
            task_info = {}
            for info in task_file:
                caption, media_path = info['caption'], info['media_path']
                media_name = media_path.split('/')[-1].replace('.mp4', '')
                task_info[media_name] = caption
            self.task_info = task_info

        if self.mode == 'generate':
            task_info_path = os.path.join(self.data_path, 'dataset.json')
            task_info = {}
            with open(task_info_path, "r") as f:
                task_file = json.load(f)
            for info in task_file:
                caption, media_path = info['caption'], info['media_path']
                media_name = media_path.split('/')[-1].replace('.mp4', '')
                task_info[media_name] = caption

            task_for_rl_path = os.path.join(self.data_path, 'bridge_for_rl.json')
            task_rl = {}
            with open(task_for_rl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_info = json.loads(line.strip())
                    task_type = line_info["task_type"][0]
                    task_language = line_info["task_language"][0]
                    task_rl[task_language] = task_type
            self.task_rl = task_rl
            self.task_info = task_info

        self.action_dim = 14  # ee xyz (3) + ee euler (3) + gripper(1) * 2 for 2 arms
        self.c_act_scaler = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.c_act_scaler = np.array(self.c_act_scaler, dtype=float)
        self.ann_files = self._init_anns(self.ann_path)

        # print(f"{len(self.ann_files)} trajectories in total")
        self.samples = self._init_sequences(self.ann_files)

        self.samples = sorted(self.samples, key=lambda x: (x["ann_file"], x["frame_ids"][0]))
        if debug and not do_evaluate:
            self.samples = self.samples[0:10]
        # print(f"{len(self.ann_files)} trajectories in total")
        # print(f"{len(self.samples)} samples in total")
        # with open('./samples_16.pkl','wb') as file:
        #     pickle.dump(self.samples,file)
        self.wrong_number = 0
        self.transform = T.Compose([T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
        self.training = False
        self.preprocess = T.Compose(
            [
                ToTensorVideo(),
                ResizePadCrop(video_size[0], video_size[1]),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.not_norm_preprocess = T.Compose([ToTensorVideo(), ResizePadCrop(video_size[0], video_size[1])])

    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.data_path}"

    def _init_anns(self, data_dir):
        ann_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if (f.endswith(".npz") and ('wrist' not in f) and ('hand' not in f) )]
        total_num = len(ann_files)
        if not self.finetune:
            if self.mode == 'train':
                return ann_files #[:total_num * 15 // 16]
            elif self.mode == 'val':
                return ann_files[total_num * 15 // 16: ]
            elif self.mode == 'generate':
                filtered_ann_files = []
                for ann_file in ann_files:
                    media_name = ann_file.split('/')[-1].replace('.npz', '')
                    task = self.task_info[media_name]
                    if task in self.task_rl.keys():
                        filtered_ann_files.append(ann_file)
                return filtered_ann_files
        else:
            # filtered_ann_files = []
            # for ann_file in ann_files:
            #     media_name = ann_file.split('/')[-1].replace('.npz', '')
            #     task = self.task_info[media_name]
            #     # if task == 'sweep into pile':
            #     #     filtered_ann_files.append(ann_file)
            #     # if 'fold the cloth' in task or 'fold cloth' in task:
            #     #     filtered_ann_files.append(ann_file)
            #     # if 'close the drawer' in task or 'close drawer' in task:
            #     #     filtered_ann_files.append(ann_file)
            #     if 'lid ' in task:
            #         filtered_ann_files.append(ann_file)
            # filtered_num = len(filtered_ann_files)
            # if self.mode == 'train':
            #     return filtered_ann_files[:150]
            # else:
            #     return filtered_ann_files[150:]

            # failure
            # train_ann_files, val_ann_files = [], []
            # for ann_file in ann_files:
            #     media_name = ann_file.split('/')[-1].replace('.npz', '')
            #     task = self.task_info[media_name]
            #     if task.startswith("close ") or task.startswith("closed "):
            #         train_ann_files.append(ann_file)
            #     if task.startswith("open ") or task.startswith("opened "):
            #         val_ann_files.append(ann_file)
            # if self.mode == 'train':
            #     return train_ann_files[:150]
            # else:
            #     return val_ann_files

            # train_ann_files, val_ann_files = [], []
            # for ann_file in ann_files:
            #     media_name = ann_file.split('/')[-1].replace('.npz', '')
            #     task = self.task_info[media_name]
            #     if task.startswith("move the carrot"):
            #         val_ann_files.append(ann_file)
            #     elif task.startswith("move the"):
            #         train_ann_files.append(ann_file)
            # if self.mode == 'train':
            #     return train_ann_files[:150]
            # else:
            #     return val_ann_files

            # train_ann_files, val_ann_files = [], []
            # for ann_file in ann_files:
            #     media_name = ann_file.split('/')[-1].replace('.npz', '')
            #     task = self.task_info[media_name]
            #     if task.startswith("put the yellow cube"):
            #         val_ann_files.append(ann_file)
            #     elif 'cube' in task and 'put' in task:
            #         train_ann_files.append(ann_file)
            # if self.mode == 'train':
            #     return train_ann_files[:150]
            # else:
            #     return val_ann_files

            # in progress
            # train_ann_files, val_ann_files = [], []
            # for ann_file in ann_files:
            #     media_name = ann_file.split('/')[-1].replace('.npz', '')
            #     task = self.task_info[media_name]
            #     if task.startswith("pick up pot") or task.startswith("pick up the pot"):
            #         val_ann_files.append(ann_file)
            #     elif 'pick up' in task:
            #         train_ann_files.append(ann_file)
            # if self.mode == 'train':
            #     return train_ann_files[:150]
            # else:
            #     return val_ann_files

            train_ann_files, val_ann_files = [], []
            for ann_file in ann_files:
                media_name = ann_file.split('/')[-1].replace('.npz', '')
                task = self.task_info[media_name]
                if task.startswith("open fridge"):
                    val_ann_files.append(ann_file)
                elif task.startswith("open"):
                    train_ann_files.append(ann_file)
            if self.mode == 'train':
                return train_ann_files[:150]
            else:
                return val_ann_files

            # we need to verify the finetuning through small smaples
            # strategy 1. each text+episode need 1 clip for training
            # ann_files = []
            # for task_name in self.task_info:
            #     if self.mode == 'train':
            #         ann_files.append(self.task_info[task_name][0])
            #     elif self.mode == 'val':
            #         clip_num = len(self.task_info[task_name])
            #         for i in range(1, clip_num):
            #             ann_files.append(self.task_info[task_name][i]) # 1301 train; 6360 in val

            # stragety 2. uniformly sample tasks
            # tasks_stat = {}
            # for task_name in self.task_info:
            #     text = task_name.split('-')[0].split(' ')[0].lower()
            #     if text not in tasks_stat:
            #         tasks_stat[text] = 1
            #     else:
            #         tasks_stat[text] += 1

            # train_ann_files, val_ann_files = [], []
            # tasks_count = {}
            # for task_name in self.task_info:
            #     text = task_name.split('-')[0].split(' ')[0].lower()
            #     if tasks_stat[text] <= 1:
            #         continue
            #     if text not in tasks_count:
            #         tasks_count[text] = 1
            #         for clip_path in self.task_info[task_name]:
            #             train_ann_files.append(clip_path)
            #     elif tasks_count[text] >= (tasks_stat[text] // 10 + 1):
            #         for clip_path in self.task_info[task_name]:
            #             if 'image_0' in clip_path:
            #                 val_ann_files.append(clip_path)
            #     elif tasks_count[text] < (tasks_stat[text] // 10 + 1):
            #         tasks_count[text] += 1
            #         for clip_path in self.task_info[task_name]:
            #             if 'image_0' in clip_path:
            #                 train_ann_files.append(clip_path)
            # if self.mode == 'train':
            #     ann_files = train_ann_files
            # else:
            #     ann_files = val_ann_files
            # return ann_files # ratio 10, 777 in trainset, 6863 in valset


    def _init_sequences(self, ann_files):
        samples = []
        with ThreadPoolExecutor(32) as executor:
            future_to_ann_file = {
                executor.submit(self._load_and_process_ann_file, ann_file): ann_file for ann_file in ann_files
            }
            # for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files)):
            for future in as_completed(future_to_ann_file):
                samples.extend(future.result())
        return samples

    def _load_and_process_ann_file(self, ann_file):
        samples = []
        ann = np.load(ann_file)

        n_frames = ann['obs_state'].shape[0]

        if self.mode == 'generate':
            start_index = n_frames - self.sequence_length
        else:
            start_index = 0

        for frame_i in range(start_index, n_frames, self.start_frame_interval):
            sample = dict()
            sample["ann_file"] = ann_file
            sample["frame_ids"] = []
            if self.load_history_video:
                sample["history_frame_ids"] = []

            curr_frame_i = frame_i
            while True:
                if curr_frame_i > (n_frames - 1):
                    break
                sample["frame_ids"].append(curr_frame_i)
                if len(sample["frame_ids"]) == self.sequence_length:
                    break
                curr_frame_i += self.sequence_interval

            if len(sample["frame_ids"]) != self.sequence_length:
                continue

            if self.load_history_video:
                L = int(self.history_video_length)
                anchor = sample["frame_ids"][0]
                start = max(0, anchor - L)
                history = list(range(start, anchor))
                if len(history) < L:
                    pad_cnt = L - len(history)
                    history = [0] * pad_cnt + history
                sample["history_frame_ids"] = history

            samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_video(self, video_path, frame_ids):
        from decord import VideoReader, cpu  # Importing here due to malloc errors on ARM when importing on top level

        current_frame_ids, history_frame_ids = frame_ids[0], frame_ids[1]
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(current_frame_ids) < len(vr)).all()
        assert (np.array(current_frame_ids) >= 0).all()
        if history_frame_ids is not None:
            assert (np.array(history_frame_ids) < len(vr)).all()
            assert (np.array(history_frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(current_frame_ids).asnumpy()
        if history_frame_ids is not None:
            history_frame_data = vr.get_batch(history_frame_ids).asnumpy()
        else:
            history_frame_data = None
        return frame_data, history_frame_data

    def _get_frames(self, video_path, frame_ids, cam_id, pre_encode):
        if pre_encode:
            raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")
        else:
            frames, history_frames = self._load_video(video_path, frame_ids)
            frames = frames.astype(np.uint8)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
            if history_frames is not None and self.load_history_video:
                history_frames = history_frames.astype(np.uint8)
                history_frames = torch.from_numpy(history_frames).permute(0, 3, 1, 2)

            def printvideo(videos, filename):
                t_videos = rearrange(videos, "f c h w -> f h w c")
                t_videos = (
                    ((t_videos / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous().numpy()
                )
                writer = imageio.get_writer(filename, fps=10)  # fps
                for frame in t_videos:
                    writer.append_data(frame)  # 1 4 13 23 # fp16 24 76 456 688

            if self.normalize:
                frames = self.preprocess(frames)
                if history_frames is not None and self.load_history_video:
                    history_frames = self.preprocess(history_frames)
            else:
                frames = self.not_norm_preprocess(frames)
                frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
                if history_frames is not None and self.load_history_video:
                    history_frames = self.not_norm_preprocess(history_frames)
                    history_frames = torch.clamp(history_frames * 255.0, 0, 255).to(torch.uint8)
        return frames, history_frames

    def _get_obs(self, video_path, frame_ids, cam_id, pre_encode):
        if cam_id is None:
            temp_cam_id = random.choice(self.cam_ids)
        else:
            temp_cam_id = cam_id
        frames, history_frames = self._get_frames(video_path, frame_ids, cam_id=temp_cam_id, pre_encode=pre_encode)
        return frames, history_frames, temp_cam_id

    def _get_robot_states(self, label, frame_ids):
        if self.action_key != 'action':
            end_position = np.array(label['obs_state'])[..., 0:3] # (N, 2, 3)
        else:
            end_position = np.array(label['action'])[..., 0:3]

        if self.action_key != 'action':
            if self.state_type == 'pos_euler':
                end_rotation = np.array(label['obs_state'])[..., 3:6]
            elif self.state_type == 'pos_quat':
                end_rotation = quats_to_euler(np.array(label['obs_state'])[..., 3:7]) # (N, 2, 4) -> (N, 2, 3)
            else:
                raise NotImplementedError
        else:
            if self.state_type == 'pos_euler':
                end_rotation = np.array(label['action'])[..., 3:6]
            elif self.state_type == 'pos_quat':
                end_rotation = quats_to_euler(np.array(label['action'])[..., 3:7]) # (N, 2, 4) -> (N, 2, 3)
            else:
                raise NotImplementedError

        if self.action_key != 'action':
            effector_position = np.array(label['obs_state'])[..., 7] # (N, 2)
            # effector_position = (effector_position > 0.01) * 1.0
        else:
            effector_position = np.array(label['action'][..., -1])

        end_p = end_position[frame_ids]
        end_s = end_rotation[frame_ids]
        # openx 0 for open, 1 for close, so we need to inverse it for aligning with agibot
        cont_gripper_states = effector_position[frame_ids]
        arm_states = np.concatenate([end_p, end_s], -1)
        return arm_states, cont_gripper_states  #[:, :, 0]

    def _get_all_robot_states(self, label, frame_ids):
        end_position = np.array(label['end_position']) # (N, 2, 3)
        end_rotation = quats_to_euler(np.array(label['end_orientation'])) # (N, 2, 4) -> (N, 2, 3)
        effector_position = np.array(label['effector_position']) # (N, 2)

        end_p = end_position[frame_ids]
        end_s = end_rotation[frame_ids]
        cont_gripper_states = effector_position[frame_ids]
        arm_states = np.concatenate([end_p, end_s], -1)
        return arm_states, cont_gripper_states

    def _get_actions(self, arm_states, gripper_states, accumulate_action):
        action = np.zeros((self.sequence_length - 1, self.action_dim))
        if accumulate_action:
            for i in range(arm_states.shape[1]):
                first_xyz = arm_states[0, i, 0:3]
                first_rpy = arm_states[0, i, 3:6]
                first_rotm = euler2rotm(first_rpy)
                for k in range(1, self.sequence_length):
                    curr_xyz = arm_states[k, i, 0:3]
                    curr_rpy = arm_states[k, i, 3:6]
                    curr_gripper = gripper_states[k, i]
                    curr_rotm = euler2rotm(curr_rpy)
                    rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
                    rel_rotm = first_rotm.T @ curr_rotm
                    rel_rpy = rotm2euler(rel_rotm)
                    action[k - 1, i*7: i*7 + 3] = rel_xyz
                    action[k - 1, i*7 + 3: i*7 + 6] = rel_rpy
                    action[k - 1, i*7 + 6] = curr_gripper
        else:
            for i in range(arm_states.shape[1]):
                for k in range(1, self.sequence_length):
                    prev_xyz = arm_states[k - 1, i, 0:3]
                    prev_rpy = arm_states[k - 1, i, 3:6]
                    prev_rotm = euler2rotm(prev_rpy)
                    curr_xyz = arm_states[k, i, 0:3]
                    curr_rpy = arm_states[k, i, 3:6]
                    curr_gripper = gripper_states[k, i]
                    curr_rotm = euler2rotm(curr_rpy)
                    rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                    rel_rotm = prev_rotm.T @ curr_rotm
                    rel_rpy = rotm2euler(rel_rotm)
                    action[k - 1, i*7: i*7 + 3] = rel_xyz
                    action[k - 1, i*7 + 3: i*7 + 6] = rel_rpy
                    action[k - 1, i*7 + 6] = curr_gripper
        return torch.from_numpy(action)  # (l - 1, act_dim)

    def __getitem__(self, index, cam_id=None, return_video=False):
        try:
            if self.mode != "train":
                np.random.seed(index)
                random.seed(index)

            sample = self.samples[index]
            ann_file = sample["ann_file"]
            frame_ids = sample["frame_ids"]
            data = dict()
            if self.load_history_video:
                history_frame_ids = sample["history_frame_ids"]
            else:
                history_frame_ids = None
            label = np.load(ann_file)
            video_path = ann_file.replace('metadata', 'clips').replace('.npz', '.mp4')
            black_path = ann_file.replace('metadata', 'blacks').replace('.npz', '_black.mp4')

            if self.action_key != 'action':
                arm_states, gripper_states = self._get_robot_states(label, frame_ids)
                # actions = np.concatenate([arm_states, gripper_states[..., None]], axis=-1)[:-1].reshape(self.sequence_length - 1, self.action_dim)
                actions = self._get_actions(arm_states, gripper_states, self.accumulate_action)
                actions *= self.c_act_scaler
            else:
                if self.mode == 'generate':
                    use_action_bias = True # bool(random.randint(0, 1))
                    data['use_action_bias'] = use_action_bias
                else:
                    use_action_bias = False
                arm_states, gripper_states = self._get_robot_states(label, frame_ids)
                actions = np.zeros((self.sequence_length - 1, self.action_dim))
                if not use_action_bias:
                    actions[:, :6] = arm_states[:-1, 0, :6]
                    actions[:, 6] = gripper_states[:-1, 0] # 1 means open, 0 means close
                else:
                    # rng = np.random.default_rng(123)
                    actions[:, :7] = augment_delta(actions[:, :7], self.aug_cfg)
                actions[:, 7:13] = arm_states[:-1, 1, :6]
                actions[:, 13] = gripper_states[:-1, 1]
                actions = torch.from_numpy(actions)
                actions *= self.c_act_scaler

            if self.load_action:
                data["action"] = actions.float()

            if self.pre_encode:
                raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")
            else:
                video, history_video, cam_id = self._get_obs(video_path, [frame_ids, history_frame_ids], cam_id, pre_encode=False)
                video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
                data["video"] = video.to(dtype=torch.uint8)
                data['first_frame'] = video[:, 0, ...]
                if self.load_history_video and history_video is not None:
                    history_video = history_video.permute(1, 0, 2, 3)
                    data['history'] = history_video.to(torch.uint8)
                if False and os.path.exists(black_path):
                    black, _, _ = self._get_obs(black_path, [frame_ids, None], cam_id, pre_encode=False)
                    black = black.permute(1, 0, 2, 3)
                    data["blacks"] = black.to(dtype=torch.uint8)

            data["annotation_file"] = ann_file
            # NOTE: __key__ is used to uniquely identify the sample, required for callback functions
            if "episode_id" in label:
                data["__key__"] = label["episode_id"]
            else:
                data["__key__"] = video_path

            # add task info
            # media_name = video_path.split('/')[-1].replace('.mp4', '')
            # data["__task__"] = self.task_rl[self.task_info[media_name]]

            # Just add these to fit the interface
            if self.load_t5_embeddings:
                t5_embeddings_path = ann_file.replace('metadata', '.precomputed/conditions/clips').replace('.npz', '.pt')
                t5_embeddings_dict = torch.load(t5_embeddings_path)
                t5_embeddings = t5_embeddings_dict['prompt_embeds'][:, :1024] # (256, 4096)
                data["t5_text_embeddings"] = t5_embeddings#.cuda()
                data["t5_text_mask"] = t5_embeddings_dict['prompt_attention_mask'].to(torch.int64)#.cuda() # (256, )
            else:
                data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16)#.cuda()
                data["t5_text_mask"] = torch.ones(512, dtype=torch.int64)#.cuda()
            data["fps"] = 10
            data["image_size"] = 256 * torch.ones(4)#.cuda()  # TODO: Does this matter?
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 256, 256)#.cuda()

            return data
        except Exception:
            warnings.warn(  # noqa: B028
                f"Invalid data encountered: {self.samples[index]['ann_file']}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")  # noqa: B028
            warnings.warn(traceback.format_exc())  # noqa: B028
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.samples))]
