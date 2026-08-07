
# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import asyncio
import logging
import random

from functools import partial
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from .base import BaseVLAEnvironment
from .venv import SubprocVectorEnv

logger = logging.getLogger(__name__)

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ImportError as exc:
    benchmark = get_libero_path = OffScreenRenderEnv = None
    _LIBERO_IMPORT_ERROR = exc
else:
    _LIBERO_IMPORT_ERROR = None


class LIBEROAdapter(BaseVLAEnvironment):
    """
    An adapter for the LIBERO benchmark suite that wraps it in a vectorized,
    asynchronous interface conforming to BaseVLAEnvironment.

    This class manages a pool of LIBERO environments running in separate processes,
    handling task sampling, state initialization, and batched stepping.

    Note: While the interface is `async`, the underlying environment calls are
    blocking. This implementation uses `asyncio.to_thread` to run blocking
    I/O without blocking the event loop, making it compatible with async frameworks.
    """

    def __init__(self,
                 task_suite_name: str,
                 num_envs: int,
                 max_steps: int,
                 num_steps_wait: int = 10,
                 model_family: str = "openvla",
                 flip: bool = False,
                 gpu_ids: List[int] = [0],
                 seed: int = 0):
        """
        Initializes the LIBERO Adapter.

        Args:
            task_suite_name (str): The name of the LIBERO task suite to use (e.g., "libero_10").
            num_envs (int): The number of parallel environments to run.
            num_steps_wait (int): Number of dummy steps to wait for stabilization after reset.
            model_family (str): The model family, affects action space format.
            gpu_ids (List[int]): A list of GPU device IDs to distribute environments across.
            seed (int): The base random seed.
        """
        if _LIBERO_IMPORT_ERROR is not None:
            raise ImportError("LIBERO is required to create a LIBEROAdapter") from _LIBERO_IMPORT_ERROR

        logger.debug(
            f"Initializing with {num_envs} environments for task suite '{task_suite_name}'.")

        self.task_suite_name = task_suite_name
        self.env_num = num_envs
        self.seed = seed
        self.max_steps = max_steps
        self.num_steps_wait = num_steps_wait
        self.model_family = model_family
        self.gpu_ids = gpu_ids
        self.flip = flip

        self.env: SubprocVectorEnv = None
        self.step_count = None

        self.benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = self.benchmark_dict[self.task_suite_name]()

    def _blocking_reset(self, task_ids: Optional[List[int]] = None, trial_ids: Optional[List[int]] = None, init_state = None, init_state_len = None) -> List[Dict[str, Any]]:
        """Synchronous implementation of the reset logic."""

        # Use provided task_ids or sample new ones
        if task_ids is None:
            logger.warning(
                f"No task_ids provided, sampling {self.env_num} new tasks.")
            task_ids = random.sample(
                range(self.task_suite.n_tasks), self.env_num)
        else:
            assert len(
                task_ids) <= self.env_num, "Provided task_ids length must less or equal num_envs"
        logger.info(f"Resetting with task IDs: {task_ids}")

        num_active_envs = len(task_ids)
        active_env_ids = list(range(num_active_envs))

        task_descriptions = []
        initial_states_list = []
        env_creators = []
        resolution = 256

        for i, task_id in enumerate(task_ids):
            task = self.task_suite.get_task(task_id)
            task_descriptions.append(task.language)
            task_initial_states = self.task_suite.get_task_init_states(task_id)
            initial_states_list.append(task_initial_states)

            assigned_gpu = self.gpu_ids[i % len(self.gpu_ids)]
            env_creators.append(
                partial(LIBEROAdapter._get_libero_env, task, assigned_gpu, resolution))

        if self.env is None or (len(self.env.workers) != len(active_env_ids)):
            # First time reset, create the SubprocVectorEnv
            self.env = SubprocVectorEnv(env_creators)
        else:
            self.env.reinit_envs(env_creators, id=active_env_ids)

        # Reset only the active environments.
        self.env.reset(id=active_env_ids)

        initial_states_to_set = []
        initial_state_ids = []
        # Use provided trial_ids or sample new ones
        if trial_ids is None:
            logger.warning(f"No trial_ids provided, sampling new trials.")
            trial_ids = [random.randint(
                0, len(initial_states_list[i]) - 1) for i in range(len(task_ids))]
        else:
            assert len(
                trial_ids) == len(task_ids), "Provided trial_ids length must equal task_ids length"

        for i in range(len(trial_ids)):
            state_id = trial_ids[i]
            initial_state_ids.append(state_id)
            if init_state is not None:
                if init_state_len is not None:
                    init_s = init_state[i][:init_state_len[i][0]]
                else:
                    init_s = init_state[i]
                initial_states_to_set.append(init_s)
            else:
                initial_states_to_set.append(initial_states_list[i][state_id])

        # Set initial state only for the active environments.
        # if self.delta_action:
        obs_np_list = self.env.set_init_state(initial_states_to_set, id=active_env_ids)

        for _ in range(self.num_steps_wait):
            dummy_actions = [self._get_dummy_action()
                            for _ in range(len(trial_ids))]
            # Step only the active environments.
            obs_np_list, _, _, _ = self.env.step(dummy_actions, id=active_env_ids)

        self.step_count = np.zeros(len(trial_ids), dtype=int)

        results = []
        for i in range(len(task_ids)):
            task_id = task_ids[i]
            trial_id = initial_state_ids[i]
            results.append({
                'type': 'init',
                'obs': obs_np_list[i],
                "task_description": task_descriptions[i],
                'valid_images': [obs_np_list[i]["agentview_image"][::-1] if self.flip else obs_np_list[i]["agentview_image"][::-1, ::-1]],
                'task_file_name': f"{self.task_suite_name}_task_{task_id}_trial_{trial_id}",
                'active': True,
                'complete': False,
                'finish_step': 0
            })
        return results

    async def reset(self, task_ids: Optional[List[int]] = None, trial_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Asynchronously resets all parallel environments."""
        return await asyncio.to_thread(self._blocking_reset, task_ids=task_ids, trial_ids=trial_ids)

    def _blocking_step(self, action: Dict[str, Any], use_vlm_rm=False) -> List[Dict[str, Any]]:
        """Synchronous implementation of the step logic for an action chunk."""

        actions = action["actions"]
        active_indices_set = set(action["indices"])
        active_indices_set_step = set(action["indices"])
        batch_size = actions.shape[0]
        results = [None] * batch_size
        step_images = [None] * batch_size
        done_raw = [False] * batch_size

        active_indices_list = sorted(list(active_indices_set))
        active_indices_list_step = sorted(list(active_indices_set_step))
        chunk_buf = [None] * batch_size

        for j in range(actions.shape[1]):
            normalized_actions = []
            active_indices_list = sorted(list(active_indices_set))
            if len(active_indices_list) == 0:
                break
            for act_idx in active_indices_list:
                normalized_action = self._normalize_gripper_action(
                    actions[act_idx][j], binarize=True)
                inverted_action = self._invert_gripper_action(normalized_action)
                normalized_actions.append(inverted_action.tolist())

            step_return = self.env.step(normalized_actions, active_indices_list)
            if len(step_return) == 4:
                obs, rew, dones, infos = step_return
            else:  # new API
                obs, rew, terminateds, truncateds, infos = step_return
                dones = np.logical_or(terminateds, truncateds)

            self.step_count[active_indices_list_step] += 1

            for i in range(len(active_indices_list)):
                act_idx = active_indices_list[i]
                if step_images[act_idx] is None:
                    step_images[act_idx] = []
                if self.flip:
                    step_images[act_idx].append(obs[i]["agentview_image"][::-1])
                else:
                    step_images[act_idx].append(obs[i]["agentview_image"][::-1, ::-1])

                if chunk_buf[act_idx] is None:
                    chunk_buf[act_idx] = dict(
                        agentview_images=[],
                        eye_in_hand_images=[],
                        gripper_qpos=[],
                        joint_pos=[],
                        eef_pos=[],
                        eef_quat=[],
                        sim_state=[],
                        actions_applied=[],
                    )

                obsi = obs[i]
                # 原始图像（不翻转；保存时再统一旋转 180°）
                if "agentview_image" in obsi:
                    chunk_buf[act_idx]["agentview_images"].append(obsi["agentview_image"])
                if "robot0_eye_in_hand_image" in obsi:
                    chunk_buf[act_idx]["eye_in_hand_images"].append(obsi["robot0_eye_in_hand_image"])

                # 低维
                if "robot0_gripper_qpos" in obsi:
                    chunk_buf[act_idx]["gripper_qpos"].append(np.asarray(obsi["robot0_gripper_qpos"]).reshape(-1))
                if "robot0_joint_pos" in obsi:
                    chunk_buf[act_idx]["joint_pos"].append(np.asarray(obsi["robot0_joint_pos"]).reshape(-1))
                if "robot0_eef_pos" in obsi:
                    chunk_buf[act_idx]["eef_pos"].append(np.asarray(obsi["robot0_eef_pos"]).reshape(3))
                if "robot0_eef_quat" in obsi:
                    chunk_buf[act_idx]["eef_quat"].append(np.asarray(obsi["robot0_eef_quat"]).reshape(4))  # wxyz

                # sim_state 若 obs/infos 提供，也收集
                if "sim_state" in obsi:
                    chunk_buf[act_idx]["sim_state"].append(np.asarray(obsi["sim_state"]).reshape(-1))
                elif isinstance(infos, (list, tuple)) and i < len(infos) and isinstance(infos[i], dict) and "sim_state" in infos[i]:
                    chunk_buf[act_idx]["sim_state"].append(np.asarray(infos[i]["sim_state"]).reshape(-1))

                # 记录实际送入 env 的动作
                chunk_buf[act_idx]["actions_applied"].append(np.asarray(normalized_actions[i]).reshape(-1))



                if use_vlm_rm:
                    if dones[i] or self.step_count[act_idx] >= self.max_steps:
                        if act_idx in active_indices_list_step:
                            done_raw[i] = dones[i]
                            active_indices_list_step.remove(act_idx)

                else:
                    if dones[i] or self.step_count[act_idx] >= self.max_steps:
                        results[act_idx] = {
                            'type': 'step',
                            'obs': obs[i],
                            'active': False,
                            'complete': dones[i],
                            'finish_step': self.step_count[act_idx],
                            'valid_images': step_images[act_idx],
                            'traj_chunk': chunk_buf[act_idx],
                        }
                        active_indices_set.remove(act_idx)

        for i in range(len(active_indices_list)):
            act_idx = active_indices_list[i]
            if results[act_idx] is None:
                results[act_idx] = {
                    'type': 'step',
                    'obs': obs[i],
                    'active': not(dones[i] or self.step_count[act_idx] >= self.max_steps),
                    'complete': dones[i] if not use_vlm_rm else done_raw[i],
                    'finish_step': self.step_count[act_idx],
                    'valid_images': step_images[act_idx],
                    'traj_chunk': chunk_buf[act_idx],
                }

        return results

    async def step(self, action: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Asynchronously steps all parallel environments.
        Note: The return types are batched for vectorized operation.
        """
        return await asyncio.to_thread(self._blocking_step, action)

    def close(self):
        """Closes all environments and shuts down subprocesses."""
        logger.debug("Closing all environments.")
        if self.env is not None:
            self.env.close()

    @staticmethod
    def _get_libero_env(task, gpu_id, resolution=256):
        """Initializes and returns the LIBERO environment."""
        task_bddl_file = os.path.join(get_libero_path(
            "bddl_files"), task.problem_folder, task.bddl_file)
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": resolution,
            "camera_widths": resolution,
            "render_gpu_device_id": gpu_id
        }
        env = OffScreenRenderEnv(**env_args)
        # IMPORTANT: seed seems to affect object positions even when using fixed initial state
        env.seed(0)
        return env

    def _get_dummy_action(self) -> List[float]:
        """Returns a neutral or no-op action for the specified model family."""
        return [0, 0, 0, 0, 0, 0, -1]

    def _normalize_gripper_action(self, action: np.ndarray, binarize: bool = True) -> np.ndarray:
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
        normalized_action[..., -1] = 2 * \
            (normalized_action[..., -1] - orig_low) / \
            (orig_high - orig_low) - 1
        if binarize:
            # Binarize to -1 or +1
            normalized_action[..., -1] = np.sign(normalized_action[..., -1])
        return normalized_action

    def _invert_gripper_action(self, action: np.ndarray) -> np.ndarray:
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
        inverted_action[..., -1] = inverted_action[..., -1] * -1.0
        return inverted_action
