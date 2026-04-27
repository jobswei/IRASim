#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import imageio
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fix rollout outputs where saved GT and prediction lengths differ. "
            "The script truncates per-view predictions to the available GT length "
            "when GT exists, then rebuilds prediction/comparison videos."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default="work_dirs/inference",
        help="Root directory to scan for rollout output folders.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="FPS used when rebuilding mp4 files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the changes without writing files.",
    )
    return parser.parse_args()


def write_video(video_path, video_array, fps):
    writer = imageio.get_writer(video_path, fps=fps)
    for frame in video_array:
        writer.append_data(frame)
    writer.close()


def find_rollout_dirs(root):
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_path}")

    rollout_dirs = []
    for metadata_path in root_path.rglob("metadata.json"):
        sample_dir = metadata_path.parent
        if "rollout_" not in sample_dir.name:
            continue
        if list(sample_dir.glob("*.pred.npy")):
            rollout_dirs.append(sample_dir)
    return sorted(set(rollout_dirs))


def load_metadata(sample_dir):
    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.exists():
        return {"sample_name": sample_dir.name, "views": {}}

    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_view_names(sample_dir, metadata):
    metadata_views = list(metadata.get("views", {}).keys())
    pred_views = sorted(path.name[:-9] for path in sample_dir.glob("*.pred.npy"))
    if metadata_views:
        return [view_name for view_name in metadata_views if (sample_dir / f"{view_name}.pred.npy").exists()]
    return pred_views


def fix_sample_dir(sample_dir, fps, dry_run):
    metadata = load_metadata(sample_dir)
    view_names = resolve_view_names(sample_dir, metadata)
    if not view_names:
        return {
            "changed": False,
            "sample_dir": str(sample_dir),
            "reason": "no_views",
        }

    pred_columns = []
    comparison_columns = []
    changed = False

    metadata.setdefault("views", {})

    for view_name in view_names:
        pred_path = sample_dir / f"{view_name}.pred.npy"
        gt_path = sample_dir / f"{view_name}.gt.npy"
        pred_mp4_path = sample_dir / f"{view_name}.pred.mp4"

        pred_array = np.load(pred_path)
        saved_pred_array = pred_array
        saved_gt_array = None

        if gt_path.exists():
            gt_array = np.load(gt_path)
            if gt_array.shape[0] > 0:
                saved_frames = min(gt_array.shape[0], pred_array.shape[0])
                saved_gt_array = gt_array[:saved_frames]
                saved_pred_array = pred_array[:saved_frames]
                if saved_frames != gt_array.shape[0] or saved_frames != pred_array.shape[0]:
                    changed = True

        if saved_gt_array is None and pred_array.shape[0] > 0:
            saved_pred_array = pred_array

        pred_columns.append(saved_pred_array)

        if saved_gt_array is not None:
            comparison_columns.append(np.concatenate([saved_gt_array, saved_pred_array], axis=1))

        if not dry_run:
            if saved_gt_array is not None:
                np.save(gt_path, saved_gt_array)
            np.save(pred_path, saved_pred_array)
            write_video(pred_mp4_path, saved_pred_array, fps=fps)

        view_metadata = dict(metadata["views"].get(view_name, {}))
        view_metadata["saved_pred_frames"] = int(saved_pred_array.shape[0])
        view_metadata["saved_gt_frames"] = int(saved_gt_array.shape[0]) if saved_gt_array is not None else 0
        view_metadata["saved_comparison_frames"] = int(saved_pred_array.shape[0]) if saved_gt_array is not None else 0
        metadata["views"][view_name] = view_metadata

    saved_prediction_frames = min(pred_array.shape[0] for pred_array in pred_columns)
    prediction_array = pred_columns[0][:saved_prediction_frames]
    if len(pred_columns) > 1:
        prediction_array = np.concatenate(
            [pred_array[:saved_prediction_frames] for pred_array in pred_columns],
            axis=2,
        )

    if comparison_columns:
        saved_comparison_frames = min(comp_array.shape[0] for comp_array in comparison_columns)
        comparison_array = comparison_columns[0][:saved_comparison_frames]
        if len(comparison_columns) > 1:
            comparison_array = np.concatenate(
                [comp_array[:saved_comparison_frames] for comp_array in comparison_columns],
                axis=2,
            )
    else:
        saved_comparison_frames = 0
        comparison_array = None

    prediction_mp4_path = sample_dir / "prediction.mp4"
    comparison_mp4_path = sample_dir / "comparison.mp4"

    if not dry_run:
        write_video(prediction_mp4_path, prediction_array, fps=fps)
        if comparison_array is not None:
            write_video(comparison_mp4_path, comparison_array, fps=fps)
        elif comparison_mp4_path.exists():
            comparison_mp4_path.unlink()

    metadata["saved_prediction_frames"] = int(saved_prediction_frames)
    metadata["saved_comparison_frames"] = int(saved_comparison_frames)
    if not dry_run:
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

    return {
        "changed": changed,
        "sample_dir": str(sample_dir),
        "saved_prediction_frames": int(saved_prediction_frames),
        "saved_comparison_frames": int(saved_comparison_frames),
        "view_count": len(view_names),
    }


def main():
    args = parse_args()
    rollout_dirs = find_rollout_dirs(args.root)
    print(f"Found {len(rollout_dirs)} rollout directories under {args.root}")

    changed_count = 0
    for sample_dir in rollout_dirs:
        result = fix_sample_dir(sample_dir, fps=args.fps, dry_run=args.dry_run)
        if result.get("changed"):
            changed_count += 1
            print(
                f"Fixed {result['sample_dir']} "
                f"(views={result['view_count']}, "
                f"prediction_frames={result['saved_prediction_frames']}, "
                f"comparison_frames={result['saved_comparison_frames']})"
            )

    mode = "Would fix" if args.dry_run else "Fixed"
    print(f"{mode} {changed_count} rollout directories")


if __name__ == "__main__":
    main()
