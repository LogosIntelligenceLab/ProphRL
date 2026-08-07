import torch
import torch.nn as nn
import numpy as np
import os

class DummyWorldModel(nn.Module):
    """
    一个用于调试的、功能上为空的World Model。
    
    它遵循了真实模型可能有的结构（继承自 nn.Module），
    但其内部没有任何计算，因此启动快、不占资源。
    """
    def __init__(self, model_config):
        """
        初始化一个空的World Model。
        
        Args:
            model_config: 这是一个配置对象或字典，我们模仿真实模型的接口，
                          但实际上并不会使用它的内容。
        """
        super().__init__()
        
        # 我们可以打印一些信息来确认配置被正确传递了
        print(f"--- [DummyWorldModel] Initialized ---")
        print(f"--- PID: {os.getpid()} ---")
        print(f"--- Received model_config: {model_config} ---")
        
        self.dummy_parameter = nn.Parameter(torch.ones(1))

    def load_state_dict(self, state_dict, strict=True):
        """
        重写加载权重的方法，使其什么都不做。
        
        这样，调用 `load_state_dict` 的代码无需修改，但实际上不会有任何加载操作。
        """
        print(f"--- [DummyWorldModel] load_state_dict called. Skipping actual loading. ---")
        pass

    @torch.no_grad()
    def step(self, current_obs_batch: np.ndarray, action_batch: np.ndarray) -> np.ndarray:
        """
        模拟环境的一步。
        
        为了确保数据流的正确性，这个方法会直接返回输入的观测，
        模拟一个“状态没有发生任何变化”的环境。

        Args:
            current_obs_batch (np.ndarray): 当前的批量观测。
            action_batch (np.ndarray): 当前的批量动作。

        Returns:
            np.ndarray: 下一步的批量观测。
        """
        print(f"--- [DummyWorldModel] step called on PID {os.getpid()} ---")
        print(f"--- Input obs shape: {current_obs_batch.shape}, dtype: {current_obs_batch.dtype} ---")
        print(f"--- Input action shape: {action_batch.shape}, dtype: {action_batch.dtype} ---")

        return current_obs_batch.copy() # 使用 .copy() 是一个好习惯，可以避免潜在的内存视图问题
    