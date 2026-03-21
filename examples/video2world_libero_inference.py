import argparse
import json
from pathlib import Path

import imageio
import numpy as np
import torch
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DDPMScheduler, PNDMScheduler
from einops import rearrange
from omegaconf import OmegaConf

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
        default="episode",
        choices=["episode", "slice"],
        help=(
            "`episode` runs autoregressive chunked inference on full episodes. "
            "`slice` keeps the original fixed-window dataset-slice inference."
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
        default=8,
        help="How many episodes/slices to run. Use -1 for all remaining items.",
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


def save_prediction(sample_dir, sample_name, pred_video, gt_video, pred_latents, metadata, fps):
    pred_uint8 = to_uint8_video(pred_video[0])
    gt_uint8 = to_uint8_video(gt_video[0])
    pred_array = pred_uint8.permute(0, 2, 3, 1).contiguous().numpy()
    gt_array = gt_uint8.permute(0, 2, 3, 1).contiguous().numpy()
    comparison_array = torch.cat([gt_uint8, pred_uint8], dim=-1).permute(0, 2, 3, 1).contiguous().numpy()

    write_video(sample_dir / "pred.mp4", pred_array, fps=fps)
    write_video(sample_dir / "gt.mp4", gt_array, fps=fps)
    write_video(sample_dir / "comparison.mp4", comparison_array, fps=fps)
    torch.save(pred_latents[0].cpu(), sample_dir / "pred_latents.pt")

    with open(sample_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved {sample_name} to {sample_dir}")


def run_slice_inference(dataset, pipeline, vae, device, args, cli_args, checkpoint_path, output_dir):
    start_index = max(cli_args.start_index, 0)
    if cli_args.max_samples < 0:
        end_index = len(dataset)
    else:
        end_index = min(len(dataset), start_index + cli_args.max_samples)

    if start_index >= end_index:
        raise ValueError(
            f"Empty inference range: start_index={start_index}, end_index={end_index}, "
            f"dataset_size={len(dataset)}."
        )

    conditioning_frames = max(int(cli_args.conditioning_frames), 1)

    with torch.no_grad():
        for dataset_index in range(start_index, end_index):
            sample = dataset[dataset_index]
            for view in expand_views(sample):
                video_name = view["video_name"]
                sample_name = (
                    f"{dataset_index:06d}_"
                    f"{video_name['episode_id']}_"
                    f"cam{video_name['cam_id']}_"
                    f"start{video_name['start_frame_id']}"
                )
                sample_dir = output_dir / sample_name
                pred_video_path = sample_dir / "pred.mp4"
                if pred_video_path.exists() and not cli_args.overwrite:
                    print(f"Skip existing sample {sample_name}")
                    continue

                sample_dir.mkdir(parents=True, exist_ok=True)
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

                metadata = {
                    "inference_mode": "slice",
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
                }
                save_prediction(
                    sample_dir=sample_dir,
                    sample_name=sample_name,
                    pred_video=pred_videos,
                    gt_video=gt_video,
                    pred_latents=pred_latents,
                    metadata=metadata,
                    fps=cli_args.fps,
                )


def run_episode_inference(dataset, pipeline, vae, device, args, cli_args, checkpoint_path, output_dir):
    if int(args.num_frames) <= 1:
        raise ValueError(f"`num_frames` must be > 1, got {args.num_frames}.")

    requested_conditioning_frames = max(int(cli_args.conditioning_frames), 1)
    requested_conditioning_frames = min(requested_conditioning_frames, int(args.num_frames) - 1)

    start_index = max(cli_args.start_index, 0)
    if cli_args.max_samples < 0:
        end_index = len(dataset.annotations)
    else:
        end_index = min(len(dataset.annotations), start_index + cli_args.max_samples)

    if start_index >= end_index:
        raise ValueError(
            f"Empty inference range: start_index={start_index}, end_index={end_index}, "
            f"episode_count={len(dataset.annotations)}."
        )

    with torch.no_grad():
        for episode_index in range(start_index, end_index):
            ann = dataset.annotations[episode_index]
            for view in resolve_episode_views(dataset, ann):
                video_name = view["video_name"]
                sample_name = (
                    f"{episode_index:06d}_"
                    f"{video_name['episode_id']}_"
                    f"cam{video_name['cam_id']}"
                )
                sample_dir = output_dir / sample_name
                pred_video_path = sample_dir / "pred.mp4"
                if pred_video_path.exists() and not cli_args.overwrite:
                    print(f"Skip existing sample {sample_name}")
                    continue

                sample_dir.mkdir(parents=True, exist_ok=True)
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
                        f"Episode stitching failed for {sample_name}: "
                        f"pred_frames={pred_video.shape[1]}, gt_frames={total_frames}."
                    )

                metadata = {
                    "inference_mode": "episode",
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
                }
                save_prediction(
                    sample_dir=sample_dir,
                    sample_name=sample_name,
                    pred_video=pred_video,
                    gt_video=gt_video,
                    pred_latents=pred_latents,
                    metadata=metadata,
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

    device = torch.device(cli_args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    torch.manual_seed(cli_args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cli_args.seed)

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
        )


if __name__ == "__main__":
    main()
