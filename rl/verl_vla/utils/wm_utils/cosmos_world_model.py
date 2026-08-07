import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import math
from typing import Dict, List, Tuple, Literal
from einops import rearrange
import mediapy as mp
from collections import deque

from verl_vla.utils.wm_utils.bridge_single_image import prepare_bridge_wm_inputs

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from megatron.core import parallel_state


# ============================================================================
# History Sampling Utilities (from video2world_action_multiview_state.py)
# ============================================================================
Strategy = Literal["arc_uniform", "fps", "log_time"]

COSMOS_TOKENIZER_REPO_ID = "nvidia/Cosmos-Predict2-2B-Video2World"
COSMOS_TOKENIZER_FILENAME = "tokenizer/tokenizer.pth"
COSMOS_TOKENIZER_REVISION = "f50c09f5d8ab133a90cac3f4886a6471e9ba3f18"


def resolve_cosmos_tokenizer_path(model_config) -> str:
    """Resolve the Cosmos Video2World tokenizer from a local override or Hugging Face."""
    from huggingface_hub import hf_hub_download

    tokenizer_path = str(model_config.get("tokenizer_path", "") or "").strip()
    if tokenizer_path:
        if not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(f"Cosmos tokenizer not found: {tokenizer_path}")
        return tokenizer_path

    repo_id = str(model_config.get("tokenizer_repo_id", COSMOS_TOKENIZER_REPO_ID))
    filename = str(model_config.get("tokenizer_filename", COSMOS_TOKENIZER_FILENAME))
    revision = str(model_config.get("tokenizer_revision", COSMOS_TOKENIZER_REVISION))
    cache_dir = model_config.get("tokenizer_cache_dir") or None
    endpoint = str(model_config.get("tokenizer_endpoint", "https://huggingface.co"))
    local_files_only = bool(model_config.get("tokenizer_local_files_only", False))

    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            endpoint=endpoint,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to obtain Cosmos tokenizer {repo_id}/{filename}. "
            "Accept the NVIDIA model terms and run `HF_ENDPOINT=https://huggingface.co hf auth login`, or set "
            "actor_rollout_ref.world_model.tokenizer_path to a local tokenizer.pth."
        ) from exc


def sample_frame_id_with_state_from_pose(
    poses: List[np.ndarray],
    current_delta: np.ndarray,
    history_length: int,
    *,
    strategy: Strategy = "arc_uniform",
    w_trans: float = 1.0,
    w_rot: float = 0.3,
) -> Tuple[List[int], np.ndarray]:
    """
    Sample history frame indices and compute state (delta-to-terminal) for each sampled frame.

    Inputs
    ------
    poses: list length n>=1, each (6,) pose already accumulated.
           poses[0] is zeros; poses[-1] is current conditional frame pose.
    current_delta: (>=6,) minus action for current conditional frame.
                   terminal pose = poses[-1] + current_delta[:6]
    history_length: desired history length

    Returns
    -------
    sampled_history_frame_ids: list[int], length==history_length, chronological, padded with 0 in front.
    state_to_terminal: np.ndarray, (history_length, 6)
        For each sampled frame i:  state[i] = terminal_pose - pose[i]
    """
    if history_length < 1:
        raise ValueError("history_length must be >= 1")
    if len(poses) < 1:
        raise ValueError("poses must be non-empty, min [zeros(6)]")

    P = np.asarray(poses, dtype=np.float64)  # (n, 6)
    if P.ndim != 2 or P.shape[1] != 6:
        raise ValueError(f"poses should be list of (6,) arrays -> (n,6), got {P.shape}")

    n = P.shape[0]
    cur = np.asarray(current_delta, dtype=np.float64).reshape(-1)
    if cur.shape[0] < 6:
        raise ValueError(f"current_delta must have at least 6 dims, got {cur.shape}")
    cur6 = cur[:6]

    idx0 = 0
    idx_last = n - 1

    # If only 1 requested, cannot include both ends: return frame0 only
    if history_length == 1:
        terminal_pose = P[idx_last] + cur6
        state_to_terminal = (terminal_pose[None, :] - P[idx0:idx0 + 1]).astype(np.float64)
        return [idx0], state_to_terminal

    def pose_dist(p: np.ndarray, q: np.ndarray) -> float:
        d = p - q
        dt = d[:3]
        dr = d[3:]
        return float(math.sqrt(w_trans * float(dt @ dt) + w_rot * float(dr @ dr)))

    lo, hi = idx0, idx_last
    need_mid = history_length - 2

    if hi <= lo or need_mid <= 0:
        chosen = [idx0] + [idx0] * need_mid + [idx_last]
        chosen = chosen[:history_length - 1] + [idx_last]
        if len(chosen) < history_length:
            chosen = [idx0] * (history_length - len(chosen)) + chosen

        terminal_pose = P[idx_last] + cur6
        state_to_terminal = (terminal_pose[None, :] - P[chosen]).astype(np.float64)
        return chosen, state_to_terminal

    candidates = [i for i in range(lo + 1, hi)]
    mids: List[int] = []

    if need_mid > 0 and len(candidates) > 0:
        if strategy == "arc_uniform":
            seg = np.zeros(n, dtype=np.float64)
            for i in range(1, n):
                seg[i] = pose_dist(P[i], P[i - 1])
            arc = np.cumsum(seg)
            arc0, arch = arc[lo], arc[hi]

            if abs(arch - arc0) < 1e-12:
                targets = np.linspace(lo, hi, num=need_mid + 2)[1:-1]
                mids = [int(round(x)) for x in targets]
            else:
                targets = np.linspace(arc0, arch, num=need_mid + 2)[1:-1]
                mids = []
                for s in targets:
                    j = int(np.argmin(np.abs(arc - s)))
                    mids.append(j)

            mids = sorted({i for i in mids if lo < i < hi})
            if len(mids) < need_mid:
                remaining = [i for i in candidates if i not in mids]
                remaining.sort()
                mids += remaining[: (need_mid - len(mids))]
            mids = mids[:need_mid]

        elif strategy == "fps":
            min_d = np.full(n, np.inf, dtype=np.float64)
            for i in range(lo, hi + 1):
                min_d[i] = min(pose_dist(P[i], P[lo]), pose_dist(P[i], P[hi]))
            min_d[lo] = 0.0
            min_d[hi] = 0.0

            mids = []
            while len(mids) < min(need_mid, len(candidates)):
                cand = max(candidates, key=lambda i: min_d[i])
                mids.append(cand)
                for i in candidates:
                    d = pose_dist(P[i], P[cand])
                    if d < min_d[i]:
                        min_d[i] = d
                min_d[cand] = 0.0

            mids = sorted(mids)
            if len(mids) < need_mid:
                remaining = [i for i in candidates if i not in mids]
                remaining.sort()
                mids += remaining[: (need_mid - len(mids))]
            mids = mids[:need_mid]

        elif strategy == "log_time":
            span = hi - lo
            picks = set()
            k = 0
            while len(picks) < need_mid and (2**k) < span + 1:
                j = hi - (2**k)
                if lo < j < hi:
                    picks.add(j)
                k += 1
            mids = sorted(picks)
            if len(mids) < need_mid:
                remaining = [i for i in candidates if i not in mids]
                remaining.sort()
                mids += remaining[: (need_mid - len(mids))]
            mids = mids[:need_mid]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    chosen = [idx0] + sorted(mids) + [idx_last]

    if len(chosen) < history_length:
        chosen = [idx0] * (history_length - len(chosen)) + chosen
    if len(chosen) > history_length:
        chosen = chosen[-history_length:]
        chosen[-1] = idx_last

    # State semantics: delta-to-terminal
    terminal_pose = P[idx_last] + cur6
    state_to_terminal = (terminal_pose[None, :] - P[chosen]).astype(np.float64)

    return chosen, state_to_terminal


def pose_based_history_sampling_with_path_collection(
    history_frame_ids: List[int],           # length Th, may include repeated 0
    history_arm_states: np.ndarray,         # (Th, 6) minus action; last is hist_last -> forcing[0]
    history_gripper_states: np.ndarray,     # (Th,) absolute gripper state at each history frame (0 close, 1 open)
    frame_ids: List[int],                   # length Tg, frames to generate/predict
    arm_states: np.ndarray,                 # (Tg, >=6) minus action; use [:Tg-1, :6] for forcing[i]->forcing[i+1]
    gripper_states: np.ndarray,             # (Tg,) absolute gripper state at each forcing frame
    sampled_history_length: int,            # Ts
    *,
    strategy: Strategy = "arc_uniform",
    w_trans: float = 1.0,
    w_rot: float = 0.3,
    rot_scale: float = 1.0,                         # optional: scale rotation dims inside pose_dist for sampling
    eps: float = 1e-12,
    sum6_tanh_scale: Tuple[float, float] = (0.05, 0.25),  # (trans_scale, rot_scale) for tanh-normalizing sum6 features
) -> Tuple[List[int], np.ndarray]:
    """
    Sample Ts history frames from a length-Th history list (chronological, padding with frame0),
    MUST include:
      - earliest index where history_frame_ids[i] == 0
      - last history index i == Th-1

    Then compute action-path tensor AP of shape (Tg, Ts, f).
    AP[i, j] is the path feature between forcing_frame_ids[i] (target) and the j-th sampled history frame.

    Requirements matched to your request:
      1) Net pose change (6D) between history->target
      2) Pose magnitude over the path (use per-step mean abs + RMS, both 6D, robust to long horizons)
      3) Length ratio: frame distance / (Th + Tg)  (1D, bounded)
      4) Gripper info: g_start, g_end (2D)
      5) Gripper toggles ratio: toggles(h->t) / toggles_total(whole timeline) (1D, bounded)

    Returns
    -------
    sampled_history_frame_ids : List[int], length Ts, chronological, padded with 0 at FRONT if needed
    AP : np.ndarray, shape (Tg, Ts, f)
         f = 6 (net pose) + 6 (mean abs) + 6 (rms) + 1 (len_ratio) + 2 (grip start/end) + 1 (toggle_ratio)
           = 22
    """
    # ---------------- checks ----------------
    if sampled_history_length < 1:
        raise ValueError("sampled_history_length (Ts) must be >= 1")
    if len(history_frame_ids) < 1:
        raise ValueError("history_frame_ids must be non-empty")
    if len(frame_ids) < 1:
        raise ValueError("forcing_frame_ids must be non-empty")

    H_ids = list(history_frame_ids)
    Th = len(H_ids)
    Tg = len(frame_ids)

    H_arm = np.asarray(history_arm_states, dtype=np.float64)
    if H_arm.shape != (Th, 6):
        raise ValueError(f"history_arm_states must have shape ({Th},6), got {H_arm.shape}")

    H_grip = np.asarray(history_gripper_states, dtype=np.float64).reshape(-1)
    if H_grip.shape[0] != Th:
        raise ValueError(f"history_gripper_states must have length {Th}, got {H_grip.shape[0]}")

    F_arm = np.asarray(arm_states, dtype=np.float64)
    if F_arm.ndim != 2 or F_arm.shape[0] < Tg or F_arm.shape[1] < 6:
        raise ValueError(f"arm_states must be (>=Tg, >=6), got {F_arm.shape}, Tg={Tg}")
    F_arm6 = F_arm[:Tg, :6]

    F_grip = np.asarray(gripper_states, dtype=np.float64).reshape(-1)
    if F_grip.shape[0] != Tg:
        raise ValueError(f"gripper_states must have length {Tg}, got {F_grip.shape[0]}")

    # ---------------- sampling (pose-aware) ----------------
    # idx0 = earliest history index where frame_id == 0, else 0
    try:
        idx0 = H_ids.index(0)
    except ValueError:
        idx0 = 0
    idx_last = Th - 1

    # integrate relative pose (wrt idx0) for sampling only
    pose = np.zeros((Th, 6), dtype=np.float64)
    for i in range(idx0 + 1, Th):
        pose[i] = pose[i - 1] + H_arm[i - 1]
    for i in range(idx0 - 1, -1, -1):
        pose[i] = pose[i + 1] - H_arm[i]

    def pose_dist(p: np.ndarray, q: np.ndarray) -> float:
        d = p - q
        dt = d[:3]
        dr = d[3:] * rot_scale
        return float(math.sqrt(w_trans * float(dt @ dt) + w_rot * float(dr @ dr)))

    lo, hi = idx0, idx_last
    need_mid = sampled_history_length - 2

    # If only 1 requested: return frame0 only (cannot include both ends)
    if sampled_history_length == 1:
        chosen = [idx0]
    # trivial span
    elif hi <= lo or need_mid <= 0:
        chosen = [idx0] + [idx0] * max(0, need_mid) + [idx_last]
        chosen = chosen[:sampled_history_length - 1] + [idx_last]
    else:
        candidates = list(range(lo + 1, hi))  # exclude lo/hi
        mids: List[int] = []

        if need_mid > 0 and len(candidates) > 0:
            if strategy == "arc_uniform":
                seg = np.zeros(Th, dtype=np.float64)
                for i in range(1, Th):
                    seg[i] = pose_dist(pose[i], pose[i - 1])
                arc = np.cumsum(seg)
                arc0, arch = arc[lo], arc[hi]

                if abs(arch - arc0) < eps:
                    # no motion -> time-uniform
                    targets = np.linspace(lo, hi, num=need_mid + 2)[1:-1]
                    mids = [int(round(x)) for x in targets]
                else:
                    targets = np.linspace(arc0, arch, num=need_mid + 2)[1:-1]
                    arc_slice = arc[lo:hi + 1]
                    mids = []
                    for s in targets:
                        j_local = int(np.argmin(np.abs(arc_slice - s)))
                        mids.append(lo + j_local)

                mids = sorted({i for i in mids if lo < i < hi})
                # fill if short: prefer earlier (cleaner) frames
                if len(mids) < need_mid:
                    remaining = [i for i in candidates if i not in mids]
                    remaining.sort()
                    mids += remaining[: (need_mid - len(mids))]
                mids = mids[:need_mid]

            elif strategy == "fps":
                min_d = np.full(Th, np.inf, dtype=np.float64)
                for i in range(lo, hi + 1):
                    min_d[i] = min(pose_dist(pose[i], pose[lo]), pose_dist(pose[i], pose[hi]))
                min_d[lo] = 0.0
                min_d[hi] = 0.0

                mids = []
                while len(mids) < min(need_mid, len(candidates)):
                    cand = max(candidates, key=lambda i: min_d[i])
                    mids.append(cand)
                    for i in candidates:
                        d = pose_dist(pose[i], pose[cand])
                        if d < min_d[i]:
                            min_d[i] = d
                    min_d[cand] = 0.0

                mids = sorted(mids)
                if len(mids) < need_mid:
                    remaining = [i for i in candidates if i not in mids]
                    remaining.sort()
                    mids += remaining[: (need_mid - len(mids))]
                mids = mids[:need_mid]

            elif strategy == "log_time":
                span = hi - lo
                picks = set()
                kpow = 0
                while len(picks) < need_mid and (2 ** kpow) < span + 1:
                    j = hi - (2 ** kpow)
                    if lo < j < hi:
                        picks.add(j)
                    kpow += 1
                mids = sorted(picks)
                if len(mids) < need_mid:
                    remaining = [i for i in candidates if i not in mids]
                    remaining.sort()
                    mids += remaining[: (need_mid - len(mids))]
                mids = mids[:need_mid]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        chosen = [idx0] + sorted(mids) + [idx_last]

        # pad with idx0 at FRONT
        if len(chosen) < sampled_history_length:
            chosen = [idx0] * (sampled_history_length - len(chosen)) + chosen
        # trim from FRONT, keep last
        if len(chosen) > sampled_history_length:
            chosen = chosen[-sampled_history_length:]
        chosen[-1] = idx_last

    sampled_history_frame_ids = [H_ids[i] for i in chosen]  # length Ts

    # ---------------- build unified timeline for paths ----------------
    # nodes: history nodes + target nodes
    node_grip = np.concatenate([H_grip, F_grip], axis=0)  # (Th+Tg,)
    N_nodes = Th + Tg

    # edges: history edges (Th edges, last goes hist_last -> target[0]) + target edges (Tg-1)
    edges = np.concatenate([H_arm, F_arm6[: max(0, Tg - 1)]], axis=0)  # (Th+Tg-1, 6)
    if edges.shape[0] != N_nodes - 1:
        raise RuntimeError("Edge count mismatch; check shapes and definitions.")

    # prefix accumulators for O(1) path stats
    prefix_sum = np.zeros((N_nodes, 6), dtype=np.float64)
    prefix_abs = np.zeros((N_nodes, 6), dtype=np.float64)
    prefix_sq  = np.zeros((N_nodes, 6), dtype=np.float64)
    for i in range(N_nodes - 1):
        a = edges[i]
        prefix_sum[i + 1] = prefix_sum[i] + a
        prefix_abs[i + 1] = prefix_abs[i] + np.abs(a)
        prefix_sq[i + 1]  = prefix_sq[i]  + a * a

    # total toggles on full timeline (bounded ratio)
    toggle_prefix = np.zeros((N_nodes,), dtype=np.float64)
    for i in range(N_nodes - 1):
        toggle_prefix[i + 1] = toggle_prefix[i] + (1.0 if node_grip[i + 1] != node_grip[i] else 0.0)
    toggles_total = float(toggle_prefix[-1])
    toggles_total = max(toggles_total, 1.0)  # avoid div-by-zero; if none, ratio will be 0 anyway

    # ---------------- build AP ----------------
    # f = 22 as described
    Fdim = 22
    AP = np.zeros((Tg, sampled_history_length, Fdim), dtype=np.float32)

    # normalize net pose to be bounded (optional but recommended)
    trans_tanh, rot_tanh = sum6_tanh_scale

    for ti in range(Tg):
        tgt_node = Th + ti  # node index in unified timeline

        for j, h_node in enumerate(chosen):
            if tgt_node < h_node:
                continue  # should not happen

            L = int(tgt_node - h_node)
            denom = float(max(L, 1))

            # 1) net pose change
            sum6 = prefix_sum[tgt_node] - prefix_sum[h_node]  # (6,)

            # optional: boundedize sum6 to reduce OOD
            sum6_b = sum6.copy()
            sum6_b[:3] = np.tanh(sum6_b[:3] / max(trans_tanh, 1e-9))
            sum6_b[3:] = np.tanh(sum6_b[3:] / max(rot_tanh, 1e-9))

            # 2) pose magnitude over path: mean_abs6 + rms6 (stable w.r.t. long horizons)
            abs6 = prefix_abs[tgt_node] - prefix_abs[h_node]
            sq6  = prefix_sq[tgt_node]  - prefix_sq[h_node]
            mean_abs6 = abs6 / denom
            rms6 = np.sqrt(np.maximum(sq6 / denom, 0.0))

            # 3) length ratio: frame distance / (Th + Tg)
            len_ratio = float(L) / float(max(Th + Tg, 1))

            # 4) gripper info + 5) toggle ratio on the path
            g_start = float(node_grip[h_node])
            g_end   = float(node_grip[tgt_node])
            toggles_path = float(toggle_prefix[tgt_node] - toggle_prefix[h_node])
            toggle_ratio = toggles_path / toggles_total  # in [0,1] approximately

            feat = np.concatenate(
                [
                    sum6_b,                  # 6
                    mean_abs6,               # 6
                    rms6,                    # 6
                    np.array([len_ratio], dtype=np.float64),
                    np.array([g_start, g_end], dtype=np.float64),
                    np.array([toggle_ratio], dtype=np.float64),
                ],
                axis=0,
            ).astype(np.float32)

            AP[ti, j] = feat

    return sampled_history_frame_ids, AP

from cosmos_predict2.configs.base.config_multiview_action import get_cosmos_predict2_multiview_state_pred_pipeline as get_cosmos_predict2_multiview_action_pipeline
from cosmos_predict2.models.video2world_multiview_action_state_pred_dit_11 import (
    ActionConditionedMultiViewStatePredDiT as BridgeStatePredDiT,
)
from cosmos_predict2.pipelines.video2world_multiview_action_state_pred import MultiviewVideo2WorldActionConditionedStatePredPipeline
from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedBatchPipeline, Video2WorldActionConditionedPipeline
from imaginaire.constants import (
    CosmosPredict2ActionConditionedModelSize,
    get_cosmos_predict2_action_conditioned_checkpoint,
)
from cosmos_predict2.data.action_conditioned.dataset_utils import (
    Resize_Preprocess,
    ToTensorVideo,
    euler2rotm,
    rotm2euler,
)
from imaginaire.utils import distributed, log, misc


class CosMosWorldModel(nn.Module):
    def __init__(self, model_config):
        """
        """
        super().__init__()

        print(f"--- [CosMosWorldModel] Initialized ---")
        print(f"--- PID: {os.getpid()} ---")
        print(f"--- Received model_config: {model_config} ---")

        self.config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_size = self.config.get("chunk_size", 20)  # Fixed: integer default
        self.unnorm_key = self.config.unnorm_key
        print(f"--- [CosMosWorldModel] Initializing on PID {os.getpid()} ---")

        args = argparse.Namespace(
            model_size=self.config.get("model_size", "2B"),
            dit_path=self.config.get("dit_path", ""),
            load_ema=self.config.get("load_ema", False),
            seed=self.config.get("seed", 0),
            guidance=self.config.get("guidance", 7.0),
            num_gpus=1,
            disable_guardrail=True,
            disable_prompt_refiner=True,
        )

        resolution = '480'
        fps = 10
        pipeline_config = get_cosmos_predict2_multiview_action_pipeline(model_size="2B", resolution="720", fps=10, views=3, frames=21)

        # The public Bridge checkpoint predates the Flexiv-specific replacement
        # used by the current default config. Select the retained Bridge DiT so
        # attention modules and state-bias dimensions match the checkpoint.
        pipeline_config.net._target_ = BridgeStatePredDiT
        pipeline_config.tokenizer.vae_pth = resolve_cosmos_tokenizer_path(self.config)
        pipeline_config.state_t = self.config.get("state_t", 6)
        # step() concatenates the real and dummy-view actions before inference.
        pipeline_config.net.action_dim = (
            2 * self.config.get("action_dim", 14) * (self.config.get("pred_len", 21) - 1)
        )
        pipeline_config.net.use_black = self.config.get("use_black", False)
        pipeline_config.net.use_history = self.config.get("use_history", False)
        pipeline_config.guardrail_config.enabled = False
        pipeline_config.prompt_refiner_config.enabled = False
        self.use_history = self.config.get("use_history", False)

        if self.use_history:
            self.history_video_length = self.config.get("history_video_length", 60)
            self.history_latent_length = self.config.get("history_latent_length", 20)
            self.history_frames = deque(maxlen=self.history_video_length)
            self.history_frame_poses = deque(maxlen=self.history_video_length)
            self.history_frame_grippers = deque(maxlen=self.history_video_length)
        
        if hasattr(args, "dit_path") and args.dit_path:
            dit_path = args.dit_path
        else:
            dit_path = get_cosmos_predict2_action_conditioned_checkpoint(
                model_size=args.model_size, resolution=resolution, fps=fps
            )
        
        misc.set_random_seed(seed=args.seed, by_rank=True)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        print(f"--- [CosMosWorldModel] Initializing Video2WorldPipeline... ---")
        # self.pipe = Video2WorldActionConditionedBatchPipeline.from_config(
        self.pipe = MultiviewVideo2WorldActionConditionedStatePredPipeline.from_config(
            config=pipeline_config,
            dit_path=dit_path,
            use_text_encoder=False,
            device=self.device,
            torch_dtype=torch.bfloat16,
            load_ema_to_reg=args.load_ema,
            load_prompt_refiner=False,
        )
        self.pipe.eval()
        
        print(f"--- [CosMosWorldModel] Initialization complete on device {self.device} ---")


    def load_state_dict(self, state_dict, strict=True):
        """
        """
        print(f"--- [CosMosWorldModel] load_state_dict called. Skipping actual loading. ---")
        pass

    def forward(self, current_obs_batch: np.ndarray, action_batch: np.ndarray, step) -> np.ndarray:
        """Run one world-model step through ``nn.Module.__call__``.

        FSDP installs its parameter all-gather hooks on ``forward``. Callers
        must therefore invoke the wrapped module instead of bypassing those
        hooks through a direct ``step`` method call.
        """
        return self.step(current_obs_batch, action_batch, step)

    @torch.no_grad()
    def step(self, current_obs_batch: np.ndarray, action_batch: np.ndarray, step) -> np.ndarray:
        """        
        Args:
            current_obs_batch (np.ndarray): 形状为 [B, H, W, 3]，uint8，范围 [0, 255]
            action_batch (np.ndarray): 形状为 [B, action_dim]

        Returns:
            np.ndarray: 下一步的批量观测，形状与输入相同 [B, H, W, 3]，uint8
        """
        B = int(current_obs_batch.shape[0])
        n_views = 2
        total_frames = 21
        latent_view_indices = torch.tensor([2, 0], device=self.device).repeat_interleave(
            self.pipe.config.state_t
        )

        # Standardize actions to [B, T, A]
        if action_batch.ndim == 2:
            actions = action_batch[:, None, :]          # [B, 1, A]
        elif action_batch.ndim == 3:
            actions = action_batch                      # [B, T, A]
        else:
            raise ValueError(f"action_batch must be [B, A] or [B, T, A], got {action_batch.shape}")

        T, A = int(actions.shape[1]), int(actions.shape[2])
        if T == 0:
            raise ValueError("Empty action sequence (T=0).")

        # --------- 1) Preprocess actions (match your current logic) ---------
        actions = actions.copy()
        # Normalize the last action dimension to [-1, +1], then sign and flip
        # For wm, 0 is open, 1 is close.
        # For libero, 1 is open, -1 is close.
        # For vla, 0 is open, 1 is close.
        
        actions[..., -1] = (actions[..., -1] >= 0.5)*1  # (B, chunk, 7)
        if self.unnorm_key == 'bridge_orig':
            c_act_scaler = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])[None, None].reshape(1, 1, 7)
        else:
            c_act_scaler = np.array([5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 1.0])[None, None].reshape(1, 1, 7)
        actions = actions * c_act_scaler
        # --------- 2) Chunking plan ---------
        max_len = int(self.config.get("wm_max_steps", 20))  # WM一次支持的最大帧数（默认20）
        if max_len <= 0:
            raise ValueError(f"wm_max_steps must be positive, got {max_len}")

        # First frames for the first chunk
        cur_first_frames: List[np.ndarray] = [frame for frame in current_obs_batch]  # B x (H,W,3), uint8
        out_chunks: List[np.ndarray] = []

        # --------- 3) Loop over chunks ---------
        for s in range(0, T, max_len):
            e = min(s + max_len, T)                         # [s, e) chunk interval
            chunk_len = e - s
            padded_actions_list: List[np.ndarray] = []
            for b in range(B):
                chunk = actions[b, s:e]     # [chunk_len, A]
                if chunk_len < max_len:
                    pad = np.zeros(
                        (max_len - chunk_len, actions.shape[-1]),
                        dtype=actions.dtype
                    )
                    chunk = np.concatenate([chunk, pad], axis=0)  # [max_len, A]
                padded_actions_list.append(chunk)

            video_inputs, dual_action_inputs = prepare_bridge_wm_inputs(
                np.stack(cur_first_frames),
                np.stack(padded_actions_list),
                total_frames=total_frames,
            )

            # --------- Pose-based history sampling ---------
            sampled_state = None
            sampled_history_frame_ids = None
            if self.use_history and len(self.history_frame_poses) > 0:
                history_poses = np.asarray(self.history_frame_poses, dtype=np.float64)
                history_grippers = np.asarray(self.history_frame_grippers, dtype=np.float64)
                history_actions = np.zeros_like(history_poses)
                history_actions[:-1] = np.diff(history_poses, axis=0)

                target_actions = np.zeros((max_len + 1, 6), dtype=np.float64)
                target_actions[:chunk_len] = actions[0, s:e, :6]
                target_grippers = np.empty(max_len + 1, dtype=np.float64)
                target_grippers[0] = history_grippers[-1]
                target_grippers[1:] = actions[0, e - 1, -1]
                target_grippers[1 : chunk_len + 1] = actions[0, s:e, -1]

                sampled_history_frame_ids, sampled_state = pose_based_history_sampling_with_path_collection(
                    list(range(len(history_poses))),
                    history_actions,
                    history_grippers,
                    list(range(len(history_poses), len(history_poses) + max_len + 1)),
                    target_actions,
                    target_grippers,
                    self.history_latent_length,
                )
                # sampled_state: [target frames, sampled history frames, 22]

            # Run the WM pipeline for this chunk
            frames_uint8_list: List[np.ndarray] = []
            generated_frames_full_list: List[np.ndarray] = []
            for b in range(B):
                video_tensor = torch.from_numpy(video_inputs[b]).permute(3, 0, 1, 2).to(self.device)

                history_tensor = None
                state_tensor = None
                if self.use_history:
                    history_obs_batch = np.array(self.history_frames)

                    if sampled_history_frame_ids is not None and len(history_obs_batch) > 0:
                        # Use pose-based sampled indices
                        # Clamp indices to valid range
                        valid_ids = [min(idx, len(history_obs_batch) - 1) for idx in sampled_history_frame_ids]
                        view1_history = history_obs_batch[valid_ids, b]  # (history_latent_length, H, W, 3)
                    else:
                        # Fallback: use last history_latent_length frames
                        view1_history = history_obs_batch[-self.history_latent_length:, b]

                    view2_history = np.zeros_like(view1_history)
                    history_np = np.concatenate([view1_history, view2_history], axis=0)
                    history_tensor = torch.from_numpy(history_np).permute(3, 0, 1, 2).unsqueeze(0)

                    # Prepare the action-path features used by the Bridge attention bias.
                    if sampled_state is not None:
                        state_tensor = sampled_state[None, ...]
                    else:
                        state_tensor = None

                video = self.pipe(
                    video_tensor=video_tensor,
                    input_path=None,
                    actions=dual_action_inputs[b],
                    state=state_tensor,
                    blacks=None,
                    history=history_tensor,
                    latent_view_indices=latent_view_indices,
                    num_conditional_frames=1,
                    guidance=0.0,
                    n_views=n_views,
                    seed=self.config.get("seed", 0),
                    num_sampling_step=self.config.get("num_sampling_step", 10),
                    use_cuda_graphs=self.config.get("use_cuda_graphs", False),
                )

                video_frames = (
                    (video[0].permute(1, 2, 3, 0).cpu().numpy() / 2 + 0.5)
                    .clip(0, 1)
                    * 255
                ).astype(np.uint8)
                view1_frames = video_frames[:total_frames]
                generated_frames_full = view1_frames[1:]  # drop conditional frame
                generated_frames_full_list.append(generated_frames_full)
                frames_uint8_list.append(generated_frames_full[:chunk_len])

            frames_uint8 = np.stack(frames_uint8_list, axis=0)
            out_chunks.append(frames_uint8)

            # --------- Update history frames and poses ---------
            if self.use_history:
                for t in range(chunk_len):
                    batch_frame = np.stack(
                        [generated_frames_full_list[b][t] for b in range(B)], axis=0
                    )
                    self.history_frames.append(batch_frame)

                # Update history_frame_poses with accumulated poses
                # Use batch 0's actions as reference
                for k in range(chunk_len):
                    action_idx = s + k
                    if action_idx < T:
                        latest_pose = self.history_frame_poses[-1] if self.history_frame_poses else np.zeros(6)
                        # Accumulate pose: current_pose = latest_pose + action[:6]
                        current_pose = latest_pose + actions[0, action_idx, :6]
                        self.history_frame_poses.append(current_pose)
                        self.history_frame_grippers.append(actions[0, action_idx, -1])
                    
            # Next chunk starts from the last predicted frame of this chunk
            cur_first_frames = [generated_frames_full_list[b][-1] for b in range(B)]
        
        # --------- 4) Concatenate along time ---------
        out = np.concatenate(out_chunks, axis=1)   # [B, T, H, W, 3]
        return out  # B, chunk_size, H, W, C
    
    def reset(self, current_obs_batch_np, task_text=None):
        """Reset the world model history with initial observation.

        Args:
            current_obs_batch_np: Initial observation batch
            task_text: Task description text (unused in CosMosWorldModel, kept for interface compatibility)
        """
        if self.use_history:
            self.history_frames = deque(maxlen=self.history_video_length)
            self.history_frames.append(current_obs_batch_np)
            self.history_frame_poses = deque([np.zeros(6)], maxlen=self.history_video_length)
            self.history_frame_grippers = deque([0.0], maxlen=self.history_video_length)
    
    # def init_env(self, init_state):
    #     if self.use_history:
    #         while len(self.history_frames) < self.history_video_length:
    
    def save_video_grid(self, video_batch: np.ndarray, save_path: str, fps: int = 10):
        """
        将一个批量的视频数据 (B, F, H, W, C) 拼接成一个网格视频并保存。
        最终的视频将会有 F 帧，每一帧都是一个 B 行 1 列的图像网格。

        Args:
            video_batch (np.ndarray): 视频数据，形状为 (B, F, H, W, C)。
            save_path (str): 保存视频的路径。
            fps (int): 视频的帧率。
        """
        B, F, H, W, C = video_batch.shape

        # 1. 交换 Batch 和 Frame 维度，方便我们遍历每一帧
        # (B, F, H, W, C) -> (F, B, H, W, C)
        video_frames_first = video_batch.transpose(1, 0, 2, 3, 4)

        # 2. 为视频的每一帧创建一个网格图像
        grid_frames = []
        for i in range(F):
            # 获取当前时间步的所有 B 个样本帧
            frame_batch_at_t = video_frames_first[i]  # 形状: (B, H, W, C)
            
            # 使用 np.vstack 将这 B 张图片垂直堆叠起来
            # 这会创建一个 B 行 1 列的网格
            grid_frame = np.vstack(list(frame_batch_at_t)) # 形状: (B*H, W, C)
            grid_frames.append(grid_frame)

        # 3. 使用 mediapy 将这些网格帧保存为视频
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        print(f"正在将 {len(grid_frames)} 帧视频保存到: {save_path}")
        mp.write_video(save_path, grid_frames, fps=fps)
        print(f"网格视频成功保存。")

    def save_trajectory_grid_image(self, video_batch: np.ndarray, save_path: str):
        """
        将一个批量的视频数据 (B, F, H, W, C) 拼接成一个静态的网格大图并保存。
        最终的图片将会是 B 行 F 列的图像网格。

        Args:
            video_batch (np.ndarray): 视频数据，形状为 (B, F, H, W, C)。
            save_path (str): 保存图片的路径 (例如: "output/grid.png")。
        """
        B, F, H, W, C = video_batch.shape

        # 1. 将一个 batch 样本的所有 F 帧在水平方向拼接成一个长条
        #    `[np.hstack(video_batch[b]) for b in range(B)]` 会生成 B 个 (H, F*W, C) 的数组
        rows = [np.hstack(video_batch[b]) for b in range(B)]

        # 2. 将所有 B 个长条在垂直方向拼接成一个巨大的网格图
        grid_image = np.vstack(rows) # 形状: (B*H, F*W, C)
        
        # 3. 使用 mediapy 保存这张图片
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        print(f"正在将形状为 {grid_image.shape} 的网格图片保存到: {save_path}")
        mp.write_image(save_path, grid_image)
        print(f"网格图片成功保存。")
        
    def process_action(self, action_sequences):
        action_processed_list = []
        for old_action in action_sequences:
            old_action = old_action.reshape(-1, 2, 7)
            end_position = old_action[..., 0:3]
            end_rotation = old_action[..., 3:6]
            effector_position = old_action[..., -1]
            frame_num = end_position.shape[0]
            action = np.zeros((frame_num - 1, 14))
            for i in range(end_position.shape[1]):
                for k in range(1, frame_num):
                    prev_xyz = end_position[k - 1, i, 0:3]
                    prev_rpy = end_rotation[k - 1, i, 0:3]
                    prev_rotm = euler2rotm(prev_rpy)
                    curr_xyz = end_position[k, i, 0:3]
                    curr_rpy = end_rotation[k, i, 0:3]
                    curr_gripper = effector_position[k, i]
                    curr_rotm = euler2rotm(curr_rpy)
                    rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                    rel_rotm = prev_rotm.T @ curr_rotm
                    rel_rpy = rotm2euler(rel_rotm)
                    action[k - 1, i*7: i*7 + 3] = rel_xyz
                    action[k - 1, i*7 + 3: i*7 + 6] = rel_rpy
                    action[k - 1, i*7 + 6] = curr_gripper * 0.03
            action = action * np.array([20, 20, 20, 20, 20, 20, 1e-5, 20, 20, 20, 20, 20, 20, 1e-5])
            action = np.concatenate([action, action[-2:-1]], axis=0)
            action_processed_list.append(action)
        return action_processed_list
            
