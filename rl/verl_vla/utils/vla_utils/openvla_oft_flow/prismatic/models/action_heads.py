"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.models.diffusion_transformer import  DiT_SingleTokenAction_OneCtx

def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)

class FlowMatchingActionHead(nn.Module):
    """
    MLP-based action head that generates continuous actions via flow matching.
    
    Flow matching is an alternative to diffusion models that directly learns the continuous-time flow
    between a simple distribution (e.g., standard normal) and the target distribution.
    
    Inspired by: https://arxiv.org/abs/2210.02747 and PI0: A Vision-Language-Action Flow Model
    """
    
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_flow_steps=10,
        num_actions=20,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_flow_steps = num_flow_steps
        
        self.flow_predictor = FlowPredictionDiT_V1(
            transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=512, action_dim=action_dim, num_actions=num_actions,
        )

        # Time encoder for positional encoding of timesteps
        self.time_encoder = nn.Identity()

    def sample_noise(self, shape, device):
        """Sample noise from a standard normal distribution."""
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.bfloat16,
            device=device,
        )
        return noise
    

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.bfloat16, device=device)
        
    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the flow prediction network. Returns noise, noisy actions, and the
        corresponding flow timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        
        # Sample random noise with shape equal to actions
        noise = self.sample_noise((batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device)
        
        # Sample random flow timesteps (one for each action in batch)
        timesteps = self.sample_time(batch_size, device)
        
        # In flow matching, we interpolate between noise and ground truth based on timestep
        time_expanded = timesteps.view(-1, 1, 1)
        noisy_actions = (1 - time_expanded) * noise + time_expanded * ground_truth_actions
        u_t = noise - ground_truth_actions

        timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        timestep_embeddings = timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)
        
        return_dict = dict(
            noise=noise,
            flow=u_t,
            noisy_actions=noisy_actions,
            timestep_embeddings=timestep_embeddings,
        )
        
        return return_dict
    
    def predict_flow(self, actions_hidden_states, noisy_actions=None, timestep_embeddings=None, 
                    noisy_action_projector=None, proprio=None, proprio_projector=None):
        """
        Given a batch of last hidden Transformer layer embeddings, predicts the flow field
        that transforms the noisy actions to the target actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        
        if noisy_actions is not None:
            noisy_actions_flat = noisy_actions.reshape(batch_size, -1).unsqueeze(-1).to(torch.bfloat16)  # (bsz, chunk_len * action_dim, 1)
            noise_actions_hidden_states = noisy_action_projector(noisy_actions_flat)  # (B, chunk_len * action_dim, llm_dim)
        else:
            noise_actions_hidden_states = torch.zeros_like(actions_hidden_states)
        
        if proprio is not None and proprio_projector is not None:
            proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
            proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
            proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
        else:
            proprio_features = None
        
        rearranged_actions_hidden_states = noise_actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        
        flow_pred = self.flow_predictor(
            obs=rearranged_actions_hidden_states,
            hidden_states=actions_hidden_states,
            time_step=timestep_embeddings,
            proprio_states=proprio_features
        )
        
        return flow_pred
    
    def sample_actions(self, actions_hidden_states, num_steps=10, noisy_action_projector=None, 
                      proprio=None, proprio_projector=None):
        """
        Samples actions by integrating the flow field from noise to the target distribution.
        """
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        
        # Start from standard normal noise
        x = self.sample_noise((batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device)
        
        # Discretize the flow into num_steps
        dt = -1.0 / num_steps  # Negative step size for reverse flow
        
        # Start from t=1 and move backward to t=0
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        # Euler integration of the flow field
        while time >= -dt / 2:
            # Current timestep in [0, 1]
            current_t = time.expand(batch_size)
            
            # Get timestep embeddings
            timestep_embeddings = self.time_encoder(current_t).to(device)
            timestep_embeddings = timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)
            
            # Predict flow at current position
            flow = self.predict_flow(
                actions_hidden_states, 
                noisy_actions=x, 
                timestep_embeddings=timestep_embeddings,
                noisy_action_projector=noisy_action_projector,
                proprio=proprio,
                proprio_projector=proprio_projector
            )
            
            # Update position using Euler step
            x = x + dt * flow
            time += dt
        
        return x
    

class FlowPredictionDiT_V1(nn.Module):
    """
    Diffusion flow prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a flow prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
        num_actions=20,
    ):
        super().__init__()
        self.dit = DiT_SingleTokenAction_OneCtx(in_channels=transformer_hidden_dim, out_channels=action_dim, depth=8, hidden_size=hidden_dim, num_heads=8, ctx_every=2, num_actions=num_actions)

    def forward(
        self,
        obs, hidden_states = None, time_step=None, proprio_states = None
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        # output: predicted noise

        output = self.dit(x = obs, context = hidden_states, timesteps = time_step, proprio = proprio_states)
        return output
    
    
class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        action = self.model(rearranged_actions_hidden_states)
        return action


class NoisePredictionModel(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.mlp_resnet = MLPResNet(
            num_blocks=2,
            input_dim=transformer_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(
        self,
        obs,
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        output = self.mlp_resnet(obs)
        return output


class DiffusionActionHead(nn.Module):
    """
    Simple MLP-based action head that generates continuous actions via conditional denoising diffusion process.

    Loosely inspired by: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_diffusion_steps_train=50,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.noise_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim
        )
        self.num_diffusion_steps_train = num_diffusion_steps_train
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps_train, beta_schedule="squaredcos_cap_v2")
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)

    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the noise prediction network. Returns noise, noisy actions, and the
        corresponding diffusion timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        # Sample random noise with shape equal to actions, used for closed-form forward diffusion.
        noise = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device, dtype=ground_truth_actions.dtype)  # (B, chunk_len, action_dim)
        # Sample random diffusion timesteps (one for each action in batch).
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_size,), device=device
        )
        # Add noise to clean actions according to the magnitude at each diffusion timestep via
        # closed-form forward diffusion.
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)  # (B, chunk_len, action_dim)

        # Get diffusion timestep embeddings as well
        diffusion_timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        return_dict = dict(
            noise=noise,
            noisy_actions=noisy_actions,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
        )

        return return_dict

    def predict_noise(self, actions_hidden_states):
        """
        Given a batch of last hidden Transformer layer embeddings (which fuse the vision-language observation embeddings,
        noisy action embeddings, and diffusion timestep embedding), predicts the noise applied to the actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)
        # Get diffusion model's noise prediction.
        noise_pred = self.noise_predictor(rearranged_actions_hidden_states)
        return noise_pred
