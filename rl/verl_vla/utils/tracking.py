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
"""
A unified tracking interface that supports logging data to different backend
"""

from typing import List, Union
import warnings

class Tracking(object):
    supported_backend = ['wandb', 'console', 'tensorboard', 'swanlab']

    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = 'console', config=None, local_dir=None, wandb_mode='online'):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == 'tracking':
                import warnings
                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning)
            else:
                assert backend in self.supported_backend, f'{backend} is not supported'

        self.logger = {}

        if 'tracking' in default_backend or 'wandb' in default_backend:
            import wandb
            wandb.init(project=project_name, name=experiment_name, mode=wandb_mode, config=config)
            self.logger['wandb'] = wandb
        
        if 'swanlab' in default_backend:
            import swanlab
            swanlab.init(project=project_name, name=experiment_name, mode=wandb_mode, config=config)
            self.logger['wandb'] = swanlab

        if 'console' in default_backend:
            from verl_vla.utils.logger.aggregate_logger import LocalLogger
            self.console_logger = LocalLogger(print_to_console=True, log_dir=local_dir)
            self.logger['console'] = self.console_logger
            
        if 'tensorboard' in default_backend:
            from torch.utils.tensorboard import SummaryWriter
            # TensorBoard logs are typically saved in a runs/ directory
            # You might want to customize the log_dir for TensorBoard
            log_dir = local_dir if local_dir else f"runs/{project_name}/{experiment_name}"
            self.tensorboard_writer = SummaryWriter(log_dir=log_dir)
            self.logger['tensorboard'] = self.tensorboard_writer

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                if default_backend == 'tensorboard':
                    # For TensorBoard, we need to iterate through the data dictionary
                    # and log each item separately.
                    for key, value in data.items():
                        if isinstance(value, (int, float)):
                            logger_instance.add_scalar(key, value, step)
                        # You can add more types if needed, e.g., images, histograms
                        else:
                            warnings.warn(f"TensorBoard does not support direct logging of type {type(value)} for key {key}. Skipping.")
                else:
                    logger_instance.log(data=data, step=step)

    def close(self):
        """
        Close any open loggers, especially important for TensorBoard.
        """
        if 'tensorboard' in self.logger:
            self.logger['tensorboard'].close()
        # Add similar closing logic for other loggers if they have it (e.g., wandb.finish())
        if 'wandb' in self.logger:
            self.logger['wandb'].finish()
        if 'swanlab' in self.logger:
            self.logger['swanlab'].finish()
