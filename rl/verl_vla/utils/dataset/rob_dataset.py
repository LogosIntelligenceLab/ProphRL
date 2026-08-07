# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from omegaconf import ListConfig
import os
from typing import List, Union
import h5py
import pandas as pd
import json
from PIL import Image 

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

from verl_vla import DataProto
from verl_vla.utils.fs import copy_local_path_from_hdfs

from verl_vla.utils.model import compute_position_id_with_mask
import verl_vla.utils.torch_functional as verl_F
try:
    from libero.libero import benchmark
except ImportError as exc:
    benchmark = None
    _LIBERO_IMPORT_ERROR = exc
else:
    _LIBERO_IMPORT_ERROR = None


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class LIBERO_Dataset(Dataset):
    def __init__(self,
                 task_suite_name,
                 num_trials_per_task = 50,
                 use_world_model = False,
                 train_val = "train",
                 libero_raw_data_dir = "",
                 ):
        if _LIBERO_IMPORT_ERROR is not None:
            raise ImportError("LIBERO is required to create a LIBERO dataset") from _LIBERO_IMPORT_ERROR

        
        self.task_suite_name = task_suite_name  
        self.num_trials_per_task = num_trials_per_task  
        self.use_world_model = use_world_model
        self.train_val = train_val
        self.libero_raw_data_dir = libero_raw_data_dir
        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        dataframes = []
        
        if self.task_suite_name in ["libero_10", "libero_90", "libero_goal",  "libero_object",  "libero_spatial"]:
            for task_id in range(num_tasks_in_suite):
                # if task_id != 4 and self.train_val == "valid":
                if self.train_val == "train":
                    if self.libero_raw_data_dir == "":
                        task = task_suite.get_task(task_id)
                        trials_range = list(range(0, int(self.num_trials_per_task))) 
                        init_state_list = []
                        init_state_len_list = []
                        initial_states = task_suite.get_task_init_states(task_id)
                        task_name_list = []
                        for i in trials_range:
                            init_state = initial_states[i]
                            state_len = len(init_state)
                            padded_state = np.pad(init_state, (0, 200 - state_len), 'constant')
                            
                            init_state_list.append(padded_state)
                            task_name_list.append(task.language)
                            init_state_len_list.append(state_len)
                    else:
                        task = task_suite.get_task(task_id)
                        orig_data_path = os.path.join(self.libero_raw_data_dir, self.task_suite_name, f"{task.name}_demo.hdf5")
                        orig_data_file = h5py.File(orig_data_path, "r")
                        orig_data = orig_data_file["data"]
                        init_state_list, task_name_list = [], []
                        init_state_len_list = []
                        for i in range(len(orig_data.keys())):
                            demo_data = orig_data[f"demo_{i}"]
                            orig_states = demo_data["states"][()]
                            init_state = orig_states[0]
                            state_len = len(init_state)
                            padded_state = np.pad(init_state, (0, 200 - state_len), "constant")

                            init_state_list.append(padded_state)
                            task_name_list.append(task.language)
                            init_state_len_list.append(state_len)
                        
                        trials_range = list(range(0, len(orig_data.keys())))
                    
                elif self.train_val == "valid":
                    trials_range = list(range(0, int(self.num_trials_per_task))) 
                    init_state_list = []
                    init_state_len_list = []
                    for i in trials_range:
                        initial_states = task_suite.get_task_init_states(task_id)
                        init_state = initial_states[i]
                        state_len = len(init_state)
                        padded_state = np.pad(init_state, (0, 200 - state_len), 'constant')
                        
                        init_state_list.append(padded_state)
                        init_state_len_list.append(state_len)
                else:
                    raise ValueError
                for i in trials_range:
                    data = {
                        "task_suite_name": self.task_suite_name,
                        "task_id": torch.tensor(task_id, dtype=torch.int64).unsqueeze(0),
                        "trial_id": torch.tensor(i, dtype=torch.int64).unsqueeze(0),
                        "init_state": torch.from_numpy(init_state_list[i].copy()),
                        "init_state_len": torch.tensor(init_state_len_list[i]).unsqueeze(0),
                    }
                    if self.train_val == "train":
                        data.update({"task_lang": task_name_list[i]})
                    dataframes.append(data)
            self.dataframe = dataframes
            print(f'dataset len: {len(self.dataframe)}')
            
        else:
            raise ValueError
     

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        return self.dataframe[item]


class Bridge_Dataset(Dataset):
    def __init__(self,
                 task_suite_name,
                 bridge_scene = 'joint',
                 use_world_model = False,
                 train_val = "train",
                 data_dir = "",
                 jsonl_path="",
                 ):
        
        self.task_suite_name = task_suite_name  
        self.bridge_scene = bridge_scene
        self.use_world_model = use_world_model
        self.train_val = train_val
        self.data_dir = data_dir
        self.jsonl_path = jsonl_path
        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        dataframes = []
        if self.jsonl_path == "":
            dataset_path  = os.path.join(self.data_dir, "dataset.jsonl")
        else:
            dataset_path = self.jsonl_path
        data_list = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line))
        
        for data_item in data_list:
            task_language = data_item['task_language']
            if self.bridge_scene != "joint":
                if self.bridge_scene not in data_item['task_type']:
                    continue
            init_state_path = os.path.join(self.data_dir,  data_item['init_state_path'][0])
            image_pil = Image.open(init_state_path).convert("RGB")
            image = np.array(image_pil)
            task_id = -1
            trial_id = -1
            init_state_len = 1
            data = {
                "task_suite_name": self.task_suite_name,
                "task_id": torch.tensor(task_id, dtype=torch.int64).unsqueeze(0),
                "trial_id": torch.tensor(trial_id, dtype=torch.int64).unsqueeze(0),
                "init_state": torch.from_numpy(image.copy()),
                "init_state_len": torch.tensor(init_state_len).unsqueeze(0),
                "task_lang": task_language[0],
            }
            dataframes.append(data)
        
        self.dataframe = dataframes
        print(f'dataset len: {len(self.dataframe)}')


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        return self.dataframe[item]



class BufferedDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.buffer = []
        self.dataloader_iter = None

    def start_new_epoch(self):
        """Reset for new epoch"""
        self.dataloader_iter = iter(self.dataloader)

    def get_next_batch(self):
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            raise StopIteration

    def __len__(self):
        return len(self.dataloader)

    def add_to_buffer(self, samples):
        if len(self.buffer) == 0:
            self.buffer = samples
        else:
            self.buffer = DataProto.concat([self.buffer, samples])

    def get_from_buffer(self, count, dp_size):
        if count > self.buffer_size():
            count = (self.buffer_size() // dp_size) * dp_size
        samples = self.buffer.slice(range(0, count))
        self.buffer = self.buffer.slice(range(count, self.buffer_size()))
        return samples

    def buffer_size(self):
        return len(self.buffer)
