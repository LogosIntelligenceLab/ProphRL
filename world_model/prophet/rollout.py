#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import cv2
import mediapy as mp
import numpy as np
import torch
from megatron.core import parallel_state

from cosmos_predict2.configs.action_conditioned.config_prophet import get_prophet_pipeline
from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedPipeline
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video
from prophet.actions import load_actions
from prophet.checkpoints import EXPECTED_ACTION_DIM, inspect_checkpoint

ACTION_CHUNK_SIZE = 20
HISTORY_LENGTH = 60

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Prophet world-model rollout")
    parser.add_argument("--checkpoint", "--dit-path", dest="checkpoint", required=True)
    parser.add_argument("--input", "--input-video", dest="input_video", required=True)
    parser.add_argument("--actions", "--input-annotation", dest="actions", required=True)
    parser.add_argument("--output", "--save-path", dest="output", default="outputs/prophet_rollout.mp4")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--autoregressive", action="store_true")
    parser.add_argument("--guidance", type=float, default=0)
    parser.add_argument("--num-sampling-steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-frames", action="store_true")
    parser.add_argument(
        "--action-format",
        choices=("auto", "libero-servo", "pose-delta", "precomputed"),
        default="auto",
        help="Action conversion applied to the annotation file",
    )
    parser.add_argument("--guardrail", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preflight", action="store_true", help="Validate inputs without allocating the model")
    return parser.parse_args()


def setup_pipeline(args: argparse.Namespace) -> Video2WorldActionConditionedPipeline:
    checkpoint = inspect_checkpoint(args.checkpoint)
    config = get_prophet_pipeline(model_size="2B", resolution="480", fps=10)
    config.state_t = 6
    config.net.action_dim = EXPECTED_ACTION_DIM
    config.net.use_black = True
    config.net.use_history = True
    config.guardrail_config.enabled = args.guardrail
    config.prompt_refiner_config.enabled = False

    misc.set_random_seed(seed=args.seed, by_rank=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.num_gpus > 1:
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)

    return Video2WorldActionConditionedPipeline.from_config(
        config=config,
        dit_path=checkpoint.path,
        use_text_encoder=False,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=False,
        load_prompt_refiner=False,
    )


def read_first_frame(path: str) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Unable to read image: {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.read_video(path)[0]


def resize_video(video: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.stack([cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR) for frame in video])


def load_action_frames(annotation_path: str, size: tuple[int, int]) -> np.ndarray:
    path = annotation_path.replace("metadata", "blacks").replace(".npz", "_black.mp4")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Projected action-frame video not found: {path}")
    return resize_video(mp.read_video(path), size)


def generate_rollout(args: argparse.Namespace, pipeline: Video2WorldActionConditionedPipeline) -> None:
    actions = load_actions(args.actions, args.action_format)
    if len(actions) < ACTION_CHUNK_SIZE:
        raise ValueError(f"Need at least {ACTION_CHUNK_SIZE} actions, got {len(actions)}")

    target_size = (args.width, args.height)
    first_frame = cv2.resize(read_first_frame(args.input_video), target_size, interpolation=cv2.INTER_AREA)
    action_frames = load_action_frames(args.actions, target_size) if args.action_frames else None
    history_frames = [first_frame] * HISTORY_LENGTH if args.history else []
    chunk_starts = range(0, len(actions), ACTION_CHUNK_SIZE) if args.autoregressive else range(1)
    generated_chunks = []

    for chunk_index, start in enumerate(chunk_starts):
        action_chunk = actions[start : start + ACTION_CHUNK_SIZE]
        if len(action_chunk) < ACTION_CHUNK_SIZE:
            break
        action_frame_chunk = None
        if action_frames is not None:
            action_frame_chunk = action_frames[start : start + ACTION_CHUNK_SIZE + 1]
            if len(action_frame_chunk) != ACTION_CHUNK_SIZE + 1:
                raise ValueError("Action-frame video is shorter than the requested action chunk")
            action_frame_chunk = action_frame_chunk[None].transpose(0, 4, 1, 2, 3)
        history_chunk = None
        if args.history:
            history_chunk = np.asarray(history_frames[-HISTORY_LENGTH:])[None].transpose(0, 4, 1, 2, 3)

        video = pipeline(
            first_frame,
            action_chunk,
            action_frame_chunk,
            history_chunk,
            num_conditional_frames=1,
            guidance=args.guidance,
            num_sampling_step=args.num_sampling_steps,
            seed=args.seed + chunk_index,
        )
        if video is None:
            raise RuntimeError("Generated video was rejected by the enabled guardrail")
        generated_chunks.append(video)
        frames = ((video[0].permute(1, 2, 3, 0).cpu().numpy() / 2 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        first_frame = frames[-1]
        history_frames.extend(frames[1:])

    if not generated_chunks:
        raise RuntimeError("No rollout chunks were generated")
    output = torch.cat([generated_chunks[0]] + [chunk[:, :, 1:] for chunk in generated_chunks[1:]], dim=2)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image_or_video(output, str(output_path), fps=10)
    log.success(f"Saved Prophet rollout to {output_path}")


def run_preflight(args: argparse.Namespace) -> None:
    checkpoint = inspect_checkpoint(args.checkpoint)
    actions = load_actions(args.actions, args.action_format)
    frame = read_first_frame(args.input_video)
    if len(actions) < ACTION_CHUNK_SIZE:
        raise ValueError(f"Need at least {ACTION_CHUNK_SIZE} actions, got {len(actions)}")
    if args.action_frames:
        load_action_frames(args.actions, (args.width, args.height))
    print(f"Checkpoint tensors: {checkpoint.tensor_count}")
    print(f"Input frame shape: {frame.shape}")
    print(f"Action shape: {actions.shape}")
    print("Prophet rollout preflight passed")


def cleanup_distributed() -> None:
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main() -> None:
    arguments = parse_args()
    if arguments.preflight:
        run_preflight(arguments)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Prophet rollout requires a CUDA GPU")
    try:
        model = setup_pipeline(arguments)
        generate_rollout(arguments, model)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
