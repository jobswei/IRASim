import argparse
import json
import os
from pathlib import Path

import imageio
import numpy as np
import torch
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DDPMScheduler, PNDMScheduler
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

from dataset.dataset_libero import Dataset_Libero
from models import get_models
from sample.pipeline_trajectory2videogen import Trajectory2VideoGenPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="LIBERO checkpoint inference.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/libero/frame_ada.yaml",
        help="Training config used to build the LIBERO model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained .pt checkpoint saved by main.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to work_dirs/inference/libero/<checkpoint_stem>.",
    )
    parser.add_argument(
        "--inference-mode",
        type=str,
        default="slice",
        choices=["episode", "slice"],
        help=(
            "`slice` runs fixed-window dataset-slice inference. "
            "`episode` runs autoregressive chunked inference on full episodes."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split mode used to build LIBERO inputs.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First episode/slice index to run inference on.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="How many episodes/slices to run. Default -1 means all remaining items.",
    )
    parser.add_argument(
        "--conditioning-frames",
        type=int,
        default=1,
        help="Number of prefix frames kept fixed during each inference chunk.",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=None,
        help="Optional override for LIBERO slice stride.",
    )
    parser.add_argument(
        "--cam-id",
        action="append",
        default=None,
        help="Optional camera id/name override. Repeat to keep multiple views.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Optional override for the diffusion sampling step count.",
    )
    parser.add_argument(
        "--encode-chunk-size",
        type=int,
        default=8,
        help="How many frames to VAE-encode at once.",
    )
    parser.add_argument(
        "--decode-chunk-size",
        type=int,
        default=8,
        help="How many frames to VAE-decode at once.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="FPS used when writing mp4 results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for deterministic inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs instead of skipping them.",
    )
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--world-size", "--world_size", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_config(config_path):
    data_config = OmegaConf.load("configs/base/data.yaml")
    diffusion_config = OmegaConf.load("configs/base/diffusion.yaml")
    task_config = OmegaConf.load(config_path)
    args = OmegaConf.merge(data_config, diffusion_config, task_config)
    args.base_dir = ""
    args.do_evaluate = True
    args.debug = False
    args.use_wandb = False
    return args


def resolve_runtime_context(cli_args):
    env_rank = os.environ.get("RANK")
    env_world_size = os.environ.get("WORLD_SIZE")
    env_local_rank = os.environ.get("LOCAL_RANK")

    rank = cli_args.rank if cli_args.rank is not None else int(env_rank) if env_rank is not None else 0
    world_size = (
        cli_args.world_size
        if cli_args.world_size is not None
        else int(env_world_size)
        if env_world_size is not None
        else 1
    )
    local_rank = (
        cli_args.local_rank
        if cli_args.local_rank is not None
        else int(env_local_rank)
        if env_local_rank is not None
        else rank
    )
    return rank, world_size, local_rank


def resolve_device(device_arg, local_rank):
    if device_arg == "cuda":
        return torch.device(f"cuda:{local_rank}")
    return torch.device(device_arg)


def build_selected_indices(total_count, start_index, max_samples):
    start_index = max(int(start_index), 0)
    if max_samples < 0:
        end_index = total_count
    else:
        end_index = min(total_count, start_index + int(max_samples))

    if start_index >= end_index:
        raise ValueError(
            f"Empty inference range: start_index={start_index}, end_index={end_index}, total_count={total_count}."
        )
    return list(range(start_index, end_index))


def shard_indices(indices, rank, world_size):
    return indices[rank::world_size]


def build_scheduler(args):
    if args.sample_method == "PNDM":
        scheduler_cls = PNDMScheduler
    elif args.sample_method == "DDPM":
        scheduler_cls = DDPMScheduler
    else:
        raise ValueError(f"Unsupported sample_method: {args.sample_method}")

    return scheduler_cls.from_pretrained(
        args.scheduler_path,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
    )


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        state_dict = checkpoint["ema"]
        source_name = "ema"
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        source_name = "model"
    else:
        state_dict = checkpoint
        source_name = "raw"

    model_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    ignored_keys = sorted(set(state_dict.keys()) - set(filtered_state_dict.keys()))
    model_dict.update(filtered_state_dict)
    model.load_state_dict(model_dict)

    print(
        f"Loaded {len(filtered_state_dict)}/{len(model_dict)} keys from "
        f"{checkpoint_path} ({source_name})."
    )
    if ignored_keys:
        print(f"Ignored {len(ignored_keys)} unexpected checkpoint keys.")


def expand_views(sample):
    video = sample["video"]
    action = sample["action"]
    video_name = sample["video_name"]

    if video.dim() == 4:
        single_name = dict(video_name)
        single_name.setdefault("cam_name", single_name.get("cam_id", "unknown"))
        return [{
            "video": video,
            "action": action,
            "video_name": single_name,
        }]

    views = []
    cam_name_list = video_name.get("cam_name", video_name["cam_id"])
    for view_idx in range(video.shape[0]):
        views.append({
            "video": video[view_idx],
            "action": action[view_idx],
            "video_name": {
                "episode_id": video_name["episode_id"][view_idx],
                "start_frame_id": video_name["start_frame_id"][view_idx],
                "cam_id": video_name["cam_id"][view_idx],
                "cam_name": cam_name_list[view_idx],
            },
        })
    return views


def load_episode_actions(dataset, ann):
    action_path = dataset.data_root / ann["actions"]
    actions = np.load(action_path)
    expected_actions = max(int(ann["num_frames"]) - 1, 0)
    actions = actions[:expected_actions]
    if actions.shape[0] < expected_actions:
        padding = np.zeros(
            (expected_actions - actions.shape[0], dataset.action_dim),
            dtype=actions.dtype,
        )
        actions = np.concatenate([actions, padding], axis=0)

    action_tensor = torch.from_numpy(actions).float()
    action_tensor = action_tensor * torch.as_tensor(dataset.c_act_scaler, dtype=torch.float32)
    return action_tensor


def resolve_episode_views(dataset, ann):
    if dataset.use_all_views:
        view_ids = dataset.cam_ids or ann.get("camera_order", list(ann["videos"].keys()))
    elif dataset.cam_ids:
        view_ids = [dataset.cam_ids[0]]
    else:
        view_ids = [0]

    total_frames = int(ann["num_frames"])
    episode_id = dataset._get_ann_id(ann)
    episode_actions = load_episode_actions(dataset, ann)

    views = []
    for view_id in view_ids:
        video, cam_name, cam_index = dataset._get_frames(ann, 0, total_frames, view_id)
        views.append({
            "video": video.float(),
            "action": episode_actions.clone(),
            "video_name": {
                "episode_id": episode_id,
                "start_frame_id": "0",
                "cam_id": str(cam_index),
                "cam_name": cam_name,
            },
        })
    return views


def pad_actions(actions, target_length):
    if actions.shape[0] >= target_length:
        return actions[:target_length]

    padding = torch.zeros(
        target_length - actions.shape[0],
        actions.shape[1],
        dtype=actions.dtype,
    )
    return torch.cat([actions, padding], dim=0)


def encode_video(video, vae, device, chunk_size):
    video = video.to(device=device, dtype=torch.float32)
    batch_size, video_length = video.shape[:2]
    frames = rearrange(video, "b f c h w -> (b f) c h w").contiguous()

    encoded_chunks = []
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, frames.shape[0], chunk_size):
        chunk = frames[start:start + chunk_size]
        encoded = vae.encode(chunk).latent_dist.sample().mul_(vae.config.scaling_factor)
        encoded_chunks.append(encoded)

    latents = torch.cat(encoded_chunks, dim=0)
    return rearrange(latents, "(b f) c h w -> b f c h w", b=batch_size, f=video_length)


def to_uint8_video(video):
    return ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(torch.uint8).cpu()


def write_video(video_path, video_array, fps):
    writer = imageio.get_writer(video_path, fps=fps)
    for frame in video_array:
        writer.append_data(frame)
    writer.close()


def split_episode_id(episode_id):
    task_name, episode_num = str(episode_id).rsplit("_", 1)
    return task_name, episode_num


def sanitize_name(name):
    return str(name).replace("/", "_").replace(" ", "_")


def build_sample_paths(output_dir, video_name):
    task_name, episode_num = split_episode_id(video_name["episode_id"])
    task_dir = output_dir / sanitize_name(task_name)
    sample_dir = task_dir / f"{episode_num}_{video_name['start_frame_id']}"
    return task_dir, sample_dir


def trim_saved_outputs(pred_video, gt_video, pred_latents, conditioning_frames):
    frame_offset = max(int(conditioning_frames), 0)
    if frame_offset >= pred_video.shape[1]:
        raise ValueError(
            f"conditioning_frames={frame_offset} must be smaller than generated frames={pred_video.shape[1]}."
        )

    pred_video = pred_video[:, frame_offset:]
    gt_video = gt_video[:, frame_offset:frame_offset + pred_video.shape[1]]
    pred_latents = pred_latents[:, frame_offset:frame_offset + pred_video.shape[1]]
    return pred_video, gt_video, pred_latents


def to_numpy_video(video):
    return to_uint8_video(video[0]).permute(0, 2, 3, 1).contiguous().numpy()


def save_sample_prediction(sample_dir, sample_name, view_results, fps):
    sample_dir.mkdir(parents=True, exist_ok=True)

    comparison_columns = []
    sorted_view_names = sorted(
        view_results.keys(),
        key=lambda name: int(view_results[name]["metadata"]["cam_id"])
        if str(view_results[name]["metadata"]["cam_id"]).isdigit()
        else str(view_results[name]["metadata"]["cam_id"]),
    )
    for view_name in sorted_view_names:
        view_data = view_results[view_name]
        gt_array = view_data["gt_array"]
        pred_array = view_data["pred_array"]
        np.save(sample_dir / f"{view_name}.gt.npy", gt_array)
        np.save(sample_dir / f"{view_name}.pred.npy", pred_array)
        comparison_columns.append(np.concatenate([gt_array, pred_array], axis=1))

    comparison_array = comparison_columns[0]
    if len(comparison_columns) > 1:
        comparison_array = np.concatenate(comparison_columns, axis=2)
    write_video(sample_dir / "comparison.mp4", comparison_array, fps=fps)

    print(f"Saved {sample_name} to {sample_dir}")


def run_slice_inference(
    dataset,
    pipeline,
    vae,
    device,
    args,
    cli_args,
    checkpoint_path,
    output_dir,
    rank,
    world_size,
):
    selected_indices = build_selected_indices(len(dataset), cli_args.start_index, cli_args.max_samples)
    local_indices = shard_indices(selected_indices, rank, world_size)
    conditioning_frames = max(int(cli_args.conditioning_frames), 1)

    if rank == 0:
        print(
            f"[Inference] mode=slice total_slices={len(selected_indices)} "
            f"dataset_size={len(dataset)} world_size={world_size}"
        )
    print(f"[Inference] rank={rank} assigned_slices={len(local_indices)}")

    progress = tqdm(
        local_indices,
        total=len(local_indices),
        desc=f"Slice Inference Rank {rank}",
        position=rank,
        leave=True,
    )
    with torch.no_grad():
        for dataset_index in progress:
            sample = dataset[dataset_index]
            views = expand_views(sample)
            if not views:
                continue

            sample_video_name = views[0]["video_name"]
            _, sample_dir = build_sample_paths(output_dir, sample_video_name)
            sample_name = f"{sample_video_name['episode_id']}/{sample_dir.name}"
            comparison_path = sample_dir / "comparison.mp4"
            if comparison_path.exists() and not cli_args.overwrite:
                print(f"Skip existing sample {sample_name}")
                continue

            view_results = {}
            for view in views:
                video_name = view["video_name"]
                gt_video = view["video"].unsqueeze(0)
                actions = view["action"].unsqueeze(0).to(device=device, dtype=torch.float32)
                latent_video = encode_video(
                    gt_video,
                    vae=vae,
                    device=device,
                    chunk_size=cli_args.encode_chunk_size,
                )
                current_conditioning_frames = min(conditioning_frames, latent_video.shape[1])
                mask_x = latent_video[:, :current_conditioning_frames]

                pred_videos, pred_latents = pipeline(
                    actions,
                    mask_x=mask_x,
                    video_length=args.num_frames,
                    height=args.video_size[0],
                    width=args.video_size[1],
                    num_inference_steps=args.infer_num_sampling_steps,
                    guidance_scale=args.guidance_scale,
                    device=device,
                    return_dict=False,
                    output_type="both",
                    decode_chunk_size=cli_args.decode_chunk_size,
                )

                saved_pred_videos, saved_gt_video, saved_pred_latents = trim_saved_outputs(
                    pred_videos,
                    gt_video,
                    pred_latents,
                    current_conditioning_frames,
                )

                view_name = sanitize_name(video_name.get("cam_name", video_name["cam_id"]))
                metadata = {
                    "inference_mode": "slice",
                    "task_name": split_episode_id(video_name["episode_id"])[0],
                    "episode_num": split_episode_id(video_name["episode_id"])[1],
                    "sample_name": sample_name,
                    "view_name": view_name,
                    "dataset_index": dataset_index,
                    "episode_id": video_name["episode_id"],
                    "start_frame_id": video_name["start_frame_id"],
                    "cam_id": video_name["cam_id"],
                    "cam_name": video_name.get("cam_name", video_name["cam_id"]),
                    "checkpoint": str(checkpoint_path),
                    "config": cli_args.config,
                    "mode": args.mode,
                    "num_frames": int(args.num_frames),
                    "sample_interval": int(args.sample_interval),
                    "conditioning_frames": int(current_conditioning_frames),
                    "num_inference_steps": int(args.infer_num_sampling_steps),
                    "encode_chunk_size": int(cli_args.encode_chunk_size),
                    "decode_chunk_size": int(cli_args.decode_chunk_size),
                    "saved_pred_frames": int(saved_pred_videos.shape[1]),
                }
                view_results[view_name] = {
                    "gt_array": to_numpy_video(saved_gt_video),
                    "pred_array": to_numpy_video(saved_pred_videos),
                    "metadata": metadata,
                }

            if view_results:
                save_sample_prediction(
                    sample_dir=sample_dir,
                    sample_name=sample_name,
                    view_results=view_results,
                    fps=cli_args.fps,
                )


def run_episode_inference(
    dataset,
    pipeline,
    vae,
    device,
    args,
    cli_args,
    checkpoint_path,
    output_dir,
    rank,
    world_size,
):
    if int(args.num_frames) <= 1:
        raise ValueError(f"`num_frames` must be > 1, got {args.num_frames}.")

    requested_conditioning_frames = max(int(cli_args.conditioning_frames), 1)
    requested_conditioning_frames = min(requested_conditioning_frames, int(args.num_frames) - 1)
    selected_indices = build_selected_indices(len(dataset.annotations), cli_args.start_index, cli_args.max_samples)
    local_indices = shard_indices(selected_indices, rank, world_size)

    if rank == 0:
        print(
            f"[Inference] mode=episode total_episodes={len(selected_indices)} "
            f"episode_count={len(dataset.annotations)} world_size={world_size}"
        )
    print(f"[Inference] rank={rank} assigned_episodes={len(local_indices)}")

    progress = tqdm(
        local_indices,
        total=len(local_indices),
        desc=f"Episode Inference Rank {rank}",
        position=rank,
        leave=True,
    )
    with torch.no_grad():
        for episode_index in progress:
            ann = dataset.annotations[episode_index]
            views = resolve_episode_views(dataset, ann)
            if not views:
                continue

            sample_video_name = views[0]["video_name"]
            _, sample_dir = build_sample_paths(output_dir, sample_video_name)
            sample_name = f"{sample_video_name['episode_id']}/{sample_dir.name}"
            comparison_path = sample_dir / "comparison.mp4"
            if comparison_path.exists() and not cli_args.overwrite:
                print(f"Skip existing sample {sample_name}")
                continue

            view_results = {}
            for view in views:
                video_name = view["video_name"]
                gt_video = view["video"].unsqueeze(0)
                full_actions = view["action"]
                total_frames = gt_video.shape[1]
                current_conditioning_frames = min(requested_conditioning_frames, total_frames)
                chunk_step = int(args.num_frames) - current_conditioning_frames
                if chunk_step <= 0:
                    raise ValueError(
                        "conditioning_frames must be smaller than num_frames for episode inference. "
                        f"Got conditioning_frames={current_conditioning_frames}, num_frames={args.num_frames}."
                    )

                mask_x = encode_video(
                    gt_video[:, :current_conditioning_frames],
                    vae=vae,
                    device=device,
                    chunk_size=cli_args.encode_chunk_size,
                )

                segment_start = 0
                segment_index = 0
                pred_video_segments = []
                pred_latent_segments = []
                segment_ranges = []

                while segment_start < total_frames:
                    available_frames = total_frames - segment_start
                    if segment_index > 0 and available_frames <= current_conditioning_frames:
                        break

                    action_slice = full_actions[segment_start:segment_start + int(args.num_frames) - 1]
                    padded_actions = pad_actions(action_slice, int(args.num_frames) - 1)
                    padded_actions = padded_actions.unsqueeze(0).to(device=device, dtype=torch.float32)

                    pred_videos, pred_latents = pipeline(
                        padded_actions,
                        mask_x=mask_x,
                        video_length=args.num_frames,
                        height=args.video_size[0],
                        width=args.video_size[1],
                        num_inference_steps=args.infer_num_sampling_steps,
                        guidance_scale=args.guidance_scale,
                        device=device,
                        return_dict=False,
                        output_type="both",
                        decode_chunk_size=cli_args.decode_chunk_size,
                    )

                    valid_frames = min(int(args.num_frames), available_frames)
                    if segment_index == 0:
                        keep_start = 0
                    else:
                        keep_start = current_conditioning_frames

                    kept_pred_video = pred_videos[:, keep_start:valid_frames]
                    kept_pred_latents = pred_latents[:, keep_start:valid_frames]
                    if kept_pred_video.shape[1] > 0:
                        pred_video_segments.append(kept_pred_video)
                        pred_latent_segments.append(kept_pred_latents)

                    segment_ranges.append({
                        "segment_index": int(segment_index),
                        "segment_start": int(segment_start),
                        "valid_frames": int(valid_frames),
                        "kept_frames": int(max(valid_frames - keep_start, 0)),
                    })

                    if available_frames <= int(args.num_frames):
                        break

                    mask_x = pred_latents[:, -current_conditioning_frames:].detach()
                    segment_start += chunk_step
                    segment_index += 1

                pred_video = torch.cat(pred_video_segments, dim=1)
                pred_latents = torch.cat(pred_latent_segments, dim=1)
                if pred_video.shape[1] != total_frames:
                    raise RuntimeError(
                        f"Episode stitching failed for {sample_name}/{video_name.get('cam_name', video_name['cam_id'])}: "
                        f"pred_frames={pred_video.shape[1]}, gt_frames={total_frames}."
                    )

                saved_pred_video, saved_gt_video, saved_pred_latents = trim_saved_outputs(
                    pred_video,
                    gt_video,
                    pred_latents,
                    current_conditioning_frames,
                )

                view_name = sanitize_name(video_name.get("cam_name", video_name["cam_id"]))
                metadata = {
                    "inference_mode": "episode",
                    "task_name": split_episode_id(video_name["episode_id"])[0],
                    "episode_num": split_episode_id(video_name["episode_id"])[1],
                    "sample_name": sample_name,
                    "view_name": view_name,
                    "episode_index": int(episode_index),
                    "episode_id": video_name["episode_id"],
                    "start_frame_id": video_name["start_frame_id"],
                    "cam_id": video_name["cam_id"],
                    "cam_name": video_name.get("cam_name", video_name["cam_id"]),
                    "checkpoint": str(checkpoint_path),
                    "config": cli_args.config,
                    "mode": args.mode,
                    "episode_total_frames": int(total_frames),
                    "num_frames": int(args.num_frames),
                    "chunk_step": int(chunk_step),
                    "conditioning_frames": int(current_conditioning_frames),
                    "num_inference_steps": int(args.infer_num_sampling_steps),
                    "encode_chunk_size": int(cli_args.encode_chunk_size),
                    "decode_chunk_size": int(cli_args.decode_chunk_size),
                    "segment_count": int(len(segment_ranges)),
                    "segment_ranges": segment_ranges,
                    "saved_pred_frames": int(saved_pred_video.shape[1]),
                }
                view_results[view_name] = {
                    "gt_array": to_numpy_video(saved_gt_video),
                    "pred_array": to_numpy_video(saved_pred_video),
                    "metadata": metadata,
                }

            if view_results:
                save_sample_prediction(
                    sample_dir=sample_dir,
                    sample_name=sample_name,
                    view_results=view_results,
                    fps=cli_args.fps,
                )


def main():
    cli_args = parse_args()
    args = load_config(cli_args.config)
    if args.dataset != "libero":
        raise ValueError(f"This script only supports dataset=libero, got {args.dataset}.")

    args.mode = cli_args.mode
    if cli_args.sample_interval is not None:
        args.sample_interval = cli_args.sample_interval
    if cli_args.cam_id is not None:
        args.cam_ids = cli_args.cam_id
    if cli_args.num_inference_steps is not None:
        args.infer_num_sampling_steps = cli_args.num_inference_steps

    rank, world_size, local_rank = resolve_runtime_context(cli_args)
    device = resolve_device(cli_args.device, local_rank)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    torch.manual_seed(cli_args.seed + rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(cli_args.seed + rank)

    checkpoint_path = Path(cli_args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = (
        Path(cli_args.output_dir)
        if cli_args.output_dir is not None
        else Path("work_dirs/inference/libero") / checkpoint_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    args.latent_size = [size // 8 for size in args.video_size]
    dataset = Dataset_Libero(args, mode=args.mode)

    model = get_models(args).to(device)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device)
    vae.eval()
    pipeline = Trajectory2VideoGenPipeline(
        vae=vae,
        scheduler=build_scheduler(args),
        transformer=model,
    )

    if cli_args.inference_mode == "episode":
        run_episode_inference(
            dataset=dataset,
            pipeline=pipeline,
            vae=vae,
            device=device,
            args=args,
            cli_args=cli_args,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            rank=rank,
            world_size=world_size,
        )
    else:
        run_slice_inference(
            dataset=dataset,
            pipeline=pipeline,
            vae=vae,
            device=device,
            args=args,
            cli_args=cli_args,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            rank=rank,
            world_size=world_size,
        )


if __name__ == "__main__":
    main()
