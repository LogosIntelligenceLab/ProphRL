import json
import os
import random
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import imageio
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# import sys
from cosmos_predict2.data.action_conditioned.dataset_utils import (
    Resize_Preprocess,
    ToTensorVideo,
    euler2rotm,
    rotm2euler,
)

def quats_to_euler(quats, order="xyz", degrees=False):
    q_flat = quats.reshape(-1, 4)
    r = R.from_quat(q_flat)
    euler_flat = r.as_euler(order, degrees=degrees)
    return euler_flat.reshape(quats.shape[0], quats.shape[1], 3)

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

def quats_to_euler(quats, order="xyz", degrees=False):
    q_flat = quats.reshape(-1, 4)
    r = R.from_quat(q_flat)
    euler_flat = r.as_euler(order, degrees=degrees)
    return euler_flat.reshape(quats.shape[0], quats.shape[1], 3)

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

caption_to_suite = {
    "pick up the black bowl between the plate and the ramekin and place it on the plate": "libero_spatial",
    "pick up the black bowl next to the ramekin and place it on the plate": "libero_spatial",
    "pick up the black bowl from table center and place it on the plate": "libero_spatial",
    "pick up the black bowl on the cookie box and place it on the plate": "libero_spatial",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": "libero_spatial",
    "pick up the black bowl on the ramekin and place it on the plate": "libero_spatial",
    "pick up the black bowl next to the cookie box and place it on the plate": "libero_spatial",
    "pick up the black bowl on the stove and place it on the plate": "libero_spatial",
    "pick up the black bowl next to the plate and place it on the plate": "libero_spatial",
    "pick up the black bowl on the wooden cabinet and place it on the plate": "libero_spatial",
    "pick up the alphabet soup and place it in the basket": "libero_object",
    "pick up the cream cheese and place it in the basket": "libero_object",
    "pick up the salad dressing and place it in the basket": "libero_object",
    "pick up the bbq sauce and place it in the basket": "libero_object",
    "pick up the ketchup and place it in the basket": "libero_object",
    "pick up the tomato sauce and place it in the basket": "libero_object",
    "pick up the butter and place it in the basket": "libero_object",
    "pick up the milk and place it in the basket": "libero_object",
    "pick up the chocolate pudding and place it in the basket": "libero_object",
    "pick up the orange juice and place it in the basket": "libero_object",
    "open the middle drawer of the cabinet": "libero_goal",
    "put the bowl on the stove": "libero_goal",
    "put the wine bottle on top of the cabinet": "libero_goal",
    "open the top drawer and put the bowl inside": "libero_goal",
    "put the bowl on top of the cabinet": "libero_goal",
    "push the plate to the front of the stove": "libero_goal",
    "put the cream cheese in the bowl": "libero_goal",
    "turn on the stove": "libero_goal",
    "put the bowl on the plate": "libero_goal",
    "put the wine bottle on the rack": "libero_goal",
    "close the top drawer of the cabinet": "libero_90",
    "close the top drawer of the cabinet and put the black bowl on top of it": "libero_90",
    "put the black bowl in the top drawer of the cabinet": "libero_90",
    "put the butter at the back in the top drawer of the cabinet and close it": "libero_90",
    "put the butter at the front in the top drawer of the cabinet and close it": "libero_90",
    "put the chocolate pudding in the top drawer of the cabinet and close it": "libero_90",
    "open the bottom drawer of the cabinet": "libero_90",
    "open the top drawer of the cabinet": "libero_90",
    "open the top drawer of the cabinet and put the bowl in it": "libero_90",
    "put the black bowl on the plate": "libero_90",
    "put the black bowl on top of the cabinet": "libero_90",
    "put the black bowl at the back on the plate": "libero_90",
    "put the black bowl at the front on the plate": "libero_90",
    "put the middle black bowl on the plate": "libero_90",
    "put the middle black bowl on top of the cabinet": "libero_90",
    "stack the black bowl at the front on the black bowl in the middle": "libero_90",
    "stack the middle black bowl on the back black bowl": "libero_90",
    "put the frying pan on the stove": "libero_90",
    "put the moka pot on the stove": "libero_90",
    "turn on the stove and put the frying pan on it": "libero_90",
    "close the bottom drawer of the cabinet": "libero_90",
    "close the bottom drawer of the cabinet and open the top drawer": "libero_90",
    "put the black bowl in the bottom drawer of the cabinet": "libero_90",
    "put the wine bottle in the bottom drawer of the cabinet": "libero_90",
    "put the wine bottle on the wine rack": "libero_90",
    "put the ketchup in the top drawer of the cabinet": "libero_90",
    "close the microwave": "libero_90",
    "put the yellow and white mug to the front of the white mug": "libero_90",
    "open the microwave": "libero_90",
    "put the white bowl on the plate": "libero_90",
    "put the white bowl to the right of the plate": "libero_90",
    "put the right moka pot on the stove": "libero_90",
    "turn off the stove": "libero_90",
    "put the frying pan on the cabinet shelf": "libero_90",
    "put the frying pan on top of the cabinet": "libero_90",
    "put the frying pan under the cabinet shelf": "libero_90",
    "put the white bowl on top of the cabinet": "libero_90",
    "pick up the alphabet soup and put it in the basket": "libero_90",
    "pick up the cream cheese box and put it in the basket": "libero_90",
    "pick up the ketchup and put it in the basket": "libero_90",
    "pick up the tomato sauce and put it in the basket": "libero_90",
    "pick up the butter and put it in the basket": "libero_90",
    "pick up the milk and put it in the basket": "libero_90",
    "pick up the orange juice and put it in the basket": "libero_90",
    "pick up the alphabet soup and put it in the tray": "libero_90",
    "pick up the butter and put it in the tray": "libero_90",
    "pick up the cream cheese and put it in the tray": "libero_90",
    "pick up the ketchup and put it in the tray": "libero_90",
    "pick up the tomato sauce and put it in the tray": "libero_90",
    "pick up the black bowl on the left and put it in the tray": "libero_90",
    "pick up the chocolate pudding and put it in the tray": "libero_90",
    "pick up the salad dressing and put it in the tray": "libero_90",
    "stack the left bowl on the right bowl and place them in the tray": "libero_90",
    "stack the right bowl on the left bowl and place them in the tray": "libero_90",
    "put the red mug on the left plate": "libero_90",
    "put the red mug on the right plate": "libero_90",
    "put the white mug on the left plate": "libero_90",
    "put the yellow and white mug on the right plate": "libero_90",
    "put the chocolate pudding to the left of the plate": "libero_90",
    "put the chocolate pudding to the right of the plate": "libero_90",
    "put the red mug on the plate": "libero_90",
    "put the white mug on the plate": "libero_90",
    "pick up the book and place it in the front compartment of the caddy": "libero_90",
    "pick up the book and place it in the left compartment of the caddy": "libero_90",
    "pick up the book and place it in the right compartment of the caddy": "libero_90",
    "pick up the yellow and white mug and place it to the right of the caddy": "libero_90",
    "pick up the book and place it in the back compartment of the caddy": "libero_10",
    "pick up the red mug and place it to the right of the caddy": "libero_90",
    "pick up the white mug and place it to the right of the caddy": "libero_90",
    "pick up the book in the middle and place it on the cabinet shelf": "libero_90",
    "pick up the book on the left and place it on top of the shelf": "libero_90",
    "pick up the book on the right and place it on the cabinet shelf": "libero_90",
    "pick up the book on the right and place it under the cabinet shelf": "libero_90",
    "put both the alphabet soup and the tomato sauce in the basket": "libero_10",
    "put both the cream cheese box and the butter in the basket": "libero_10",
    "turn on the stove and put the moka pot on it": "libero_10",
    "put the black bowl in the bottom drawer of the cabinet and close it": "libero_10",
    "put the white mug on the left plate and put the yellow and white mug on the right plate": "libero_10",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate": "libero_10",
    "put both the alphabet soup and the cream cheese box in the basket": "libero_10",
    "put both moka pots on the stove": "libero_10",
    "put the yellow and white mug in the microwave and close it": "libero_10"
  }

class ActionConditionedLiberoDataset(Dataset):
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
        use_self_forcing=False,
        history_video_length=60,
        mode="train",
        servo=False,
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
            self.start_frame_interval = 10
        elif mode == "val":
            self.start_frame_interval = val_start_frame_interval
        elif mode == "test":
            self.start_frame_interval = val_start_frame_interval
        self.data_path = data_path
        self.video_path = os.path.join(self.data_path, 'clips')
        self.ann_path = os.path.join(self.data_path, 'metadata')
        self.sequence_interval = sequence_interval
        self.mode = mode
        self.sequence_length = num_frames
        self.normalize = normalize
        self.load_action = load_action
        self.pre_encode = False
        self.use_self_forcing = use_self_forcing
        if self.use_self_forcing:
            self.load_history_video = True
            self.history_video_length = num_frames + history_video_length - 1
        else:
            self.load_history_video = load_history_video
            self.history_video_length = history_video_length
        self.servo = servo

        # task info
        task_info = {}
        task_info_path = os.path.join(self.data_path, 'dataset.json')
        with open(task_info_path, "r") as f:
            task_file = json.load(f)
        for info in task_file:
            caption, media_path = info['caption'], info['media_path']
            media_name = media_path.split('/')[-1].replace('.mp4', '')
            if 'eye_in_hand' in media_name:
                continue
            task = caption_to_suite[caption]
            task_info[media_name] = task
        self.task_info = task_info

        self.cam_ids = cam_ids
        self.accumulate_action = accumulate_action
        self.load_t5_embeddings = load_t5_embeddings

        self.action_dim = 14  # ee xyz (3) + ee euler (3) + gripper(1) * 2 for 2 arms
        if self.servo:
            self.c_act_scaler = [5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 1, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 1]
        else:
            self.c_act_scaler = [1, 1, 1, 1, 1, 1, 1/0.04, 1, 1, 1, 1, 1, 1, 1/0.04]
        self.c_act_scaler = np.array(self.c_act_scaler, dtype=float)
        self.ann_files = self._init_anns(self.ann_path)

        print(f"{len(self.ann_files)} trajectories in total")
        self.samples = self._init_sequences(self.ann_files)

        self.samples = sorted(self.samples, key=lambda x: (x["ann_file"], x["frame_ids"][0]))
        if debug and not do_evaluate:
            self.samples = self.samples[0:10]
        print(f"{len(self.ann_files)} trajectories in total")
        print(f"{len(self.samples)} samples in total")
        # with open('./samples_16.pkl','wb') as file:
        #     pickle.dump(self.samples,file)
        self.wrong_number = 0
        self.transform = T.Compose([T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
        self.training = False
        self.preprocess = T.Compose(
            [
                ToTensorVideo(),
                Resize_Preprocess(tuple(video_size)),  # 288 512
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.not_norm_preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])

    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.data_path}"

    def _init_anns(self, data_dir):
        ann_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if (f.endswith(".npz") and 'agentview' in f)]
        total_num = len(ann_files)
        if self.mode == 'train':
            return ann_files[:total_num * 15 // 16]
        elif self.mode == 'test':
            return ann_files[total_num * 14 // 16: total_num * 15 // 16]
        elif self.mode == 'val':
            # return ann_files[total_num * 15 // 16: ]
            val_ann_files = ann_files[total_num * 15 // 16: ]
            filtered_ann_files = []
            for ann_file in val_ann_files:
                media_name = ann_file.split('/')[-1].replace('.npz', '')
                if self.task_info[media_name] == 'libero_10':
                    filtered_ann_files.append(ann_file)
            return filtered_ann_files

    def _init_sequences(self, ann_files):
        samples = []
        with ThreadPoolExecutor(32) as executor:
            future_to_ann_file = {
                executor.submit(self._load_and_process_ann_file, ann_file): ann_file for ann_file in ann_files
            }
            for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files)):
                samples.extend(future.result())
        return samples

    def _load_and_process_ann_file(self, ann_file):
        samples = []
        ann = np.load(ann_file)

        n_frames = ann['end_position'].shape[0]
        for frame_i in range(0, n_frames, self.start_frame_interval):
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
                if self.use_self_forcing:
                    history = history + [anchor]
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
        extrinsics = np.array(label['extrinsics']) # (N, 4, 4)
        intrinsics = np.array(label['intrinsics']) # (3, 3)
        end_position = np.array(label['end_position']) # (N, 2, 3)
        end_rotation = quats_to_euler(np.array(label['end_orientation'])) # (N, 2, 4) -> (N, 2, 3)
        effector_position = np.array(label['effector_position'])[..., 0] # (N, 2)
        # effector_position = (effector_position > 0.01) * 1.0

        end_p = end_position[frame_ids]
        end_s = end_rotation[frame_ids]
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
            if self.load_history_video:
                history_frame_ids = sample["history_frame_ids"]
            else:
                history_frame_ids = None
            label = np.load(ann_file)
            video_path = ann_file.replace('metadata', 'clips').replace('.npz', '.mp4')
            black_path = ann_file.replace('metadata', 'blacks').replace('.npz', '_black.mp4')
            if self.servo:
                # using servo actions
                # In the raw servo action -1 means open, 1 mean close, so we need to change it
                actions = np.array(label['actions'])[frame_ids][:-1].reshape(self.sequence_length - 1, self.action_dim)
                actions[:, 6] = (1 - actions[:, 6]) / 2
                actions = torch.from_numpy(actions)
                if self.use_self_forcing:
                    history_actions = np.array(label['actions'])[history_frame_ids[-self.sequence_length:]][:-1].reshape(self.sequence_length - 1, self.action_dim)
                    history_actions[:, 6] = (1 - history_actions[:, 6]) / 2
                    history_actions = torch.from_numpy(history_actions)
                    history_actions *= self.c_act_scaler
            else:
                # using delta actions
                arm_states, gripper_states = self._get_robot_states(label, frame_ids)
                # actions = np.concatenate([arm_states, gripper_states[..., None]], axis=-1)[:-1].reshape(self.sequence_length - 1, self.action_dim)
                actions = self._get_actions(arm_states, gripper_states, self.accumulate_action)
                if self.use_self_forcing:
                    history_arm_states, history_gripper_states = self._get_robot_states(label, history_frame_ids[-self.sequence_length:])
                    history_actions = self._get_actions(history_arm_states, history_gripper_states, self.accumulate_action)
                    history_actions *= self.c_act_scaler
            actions *= self.c_act_scaler

            data = dict()
            if self.load_action:
                data["action"] = actions.float()
                if self.use_self_forcing:
                    data['history_action'] = history_actions.float()

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
                if not self.servo and os.path.exists(black_path):
                    if not self.use_self_forcing:
                        black, _, _ = self._get_obs(black_path, [frame_ids, None], cam_id, pre_encode=False)
                        black = black.permute(1, 0, 2, 3)
                        data["blacks"] = black.to(dtype=torch.uint8)
                    else:
                        black, history_black, _ = self._get_obs(black_path, [frame_ids, history_frame_ids], cam_id, pre_encode=False)
                        black = black.permute(1, 0, 2, 3)
                        data["blacks"] = black.to(dtype=torch.uint8)
                        history_black = history_black.permute(1, 0, 2, 3)
                        data['history_black'] = history_black.to(torch.uint8)

            data["annotation_file"] = ann_file
            # NOTE: __key__ is used to uniquely identify the sample, required for callback functions
            if "episode_id" in label:
                data["__key__"] = label["episode_id"]
            else:
                data["__key__"] = video_path

            # add task info
            media_name = video_path.split('/')[-1].replace('.mp4', '')
            data["__task__"] = self.task_info[media_name]

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
            if self.use_self_forcing:
                # data['self_forcing'] = True
                data['history_video_length'] = self.history_video_length - self.sequence_length + 1

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

if __name__ == '__main__':
    dataset = ActionConditionedLiberoDataset(mode='val') # 170005 samples in trainset
    # data = dataset[2]
    import pdb; pdb.set_trace()
