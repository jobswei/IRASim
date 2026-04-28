# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import random
import traceback
import warnings
from pathlib import Path

import imageio
import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms as T

from dataset.video_transforms import Resize_Preprocess, ToTensorVideo


class Dataset_Libero(Dataset):
    def __init__(self, args, mode='val'):
        super().__init__()
        self.args = args
        self.mode = mode
        self.dataset_name = str(getattr(args, 'dataset', 'libero'))
        self.sequence_length = int(args.num_frames)
        self.cam_ids = list(getattr(args, 'cam_ids', [0]))
        self.video_reader_backend = str(
            getattr(args, 'video_reader_backend', self._get_default_video_reader_backend())
        ).lower()
        self.use_all_views = bool(
            getattr(
                args,
                'use_all_views',
                getattr(args, f'{self.dataset_name}_use_all_views', False),
            )
        )
        self.sample_interval = int(getattr(args, 'sample_interval', 10))
        self.sample_strategy = str(getattr(args, 'sample_strategy', 'uniform')).lower()
        if self.sample_strategy not in {'uniform', 'random'}:
            raise ValueError(f"Unsupported sample_strategy: {self.sample_strategy}")

        if getattr(args, 'pre_encode', False):
            raise NotImplementedError(
                f"{self.dataset_name} does not provide latent videos. Set `pre_encode: False`."
            )

        self.action_dim = int(getattr(args, 'action_dim', self._get_default_action_dim()))
        self.c_act_scaler = np.array(
            getattr(
                args,
                'action_scaler',
                getattr(args, f'{self.dataset_name}_c_act_scaler', [1.0] * self.action_dim),
            ),
            dtype=float,
        )
        if self.c_act_scaler.shape != (self.action_dim,):
            raise ValueError(
                f"`action_scaler` for {self.dataset_name} must have shape "
                f"({self.action_dim},), got {self.c_act_scaler.shape}."
            )
        self.dataset_log_first_n = int(getattr(args, 'dataset_log_first_n', 2))
        self.training = False
        self.wrong_number = 0
        self._logged_samples = 0
        self._ffmpeg_fallback_warned_paths = set()

        self.data_root = self._resolve_data_root()
        self.annotation_path, self.annotation_source = self._resolve_annotation_path(mode)
        self.annotations = self._load_annotations(self.annotation_path)
        if self.annotation_path is None:
            self.annotation_path = self.annotation_source
        self.ann_dict = {self._get_ann_id(ann): ann for ann in self.annotations}
        self.ann_files = list(self.ann_dict.keys())
        self.samples = self._init_sequences(self.annotations)
        self.samples = sorted(
            self.samples,
            key=lambda x: (x['ann_file'], x.get('frame_interval', [0])[0]),
        )
        if args.debug and not args.do_evaluate:
            self.samples = self.samples[0:10]

        print(f'{len(self.ann_files)} trajectories in total')
        print(f'{len(self.samples)} samples in total')
        self._print_dataset_info()

        self.preprocess = T.Compose([
            ToTensorVideo(),
            Resize_Preprocess(tuple(args.video_size)),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.not_norm_preprocess = T.Compose([
            ToTensorVideo(),
            Resize_Preprocess(tuple(args.video_size)),
        ])

    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.annotation_path}"

    def _log_prefix(self, stage):
        return f"[Dataset_{self.dataset_name.capitalize()}:{stage}]"

    def _get_default_action_dim(self):
        if self.dataset_name == 'agibot':
            return 16
        return 7

    def _get_default_video_reader_backend(self):
        if self.dataset_name == 'agibot':
            return 'ffmpeg'
        return 'auto'

    def _print_dataset_info(self):
        frame_counts = [int(ann['num_frames']) for ann in self.annotations]
        if frame_counts:
            min_frames = min(frame_counts)
            max_frames = max(frame_counts)
            avg_frames = sum(frame_counts) / len(frame_counts)
        else:
            min_frames = max_frames = avg_frames = 0

        print(
            f"{self._log_prefix('init')} mode={self.mode} "
            f"data_root={self.data_root} annotation_path={self.annotation_path}"
        )
        print(
            f"{self._log_prefix('init')} num_frames={self.sequence_length} "
            f"mask_frame_num={getattr(self.args, 'mask_frame_num', 'N/A')} "
            f"sample_interval={self.sample_interval} "
            f"sequence_interval={getattr(self.args, 'sequence_interval', 'N/A')} "
            f"sample_strategy={self.sample_strategy}"
        )
        print(
            f"{self._log_prefix('init')} cam_ids={self.cam_ids} "
            f"use_all_views={self.use_all_views} normalize={self.args.normalize} "
            f"pre_encode={getattr(self.args, 'pre_encode', False)} "
            f"video_reader_backend={self.video_reader_backend} "
            f"action_dim={self.action_dim} "
            f"dataset_log_first_n={self.dataset_log_first_n}"
        )
        print(
            f"{self._log_prefix('init')} trajectories={len(self.ann_files)} "
            f"samples={len(self.samples)} frame_stats(min/avg/max)="
            f"{min_frames}/{avg_frames:.1f}/{max_frames}"
        )
        if self.annotations:
            first_ann = self.annotations[0]
            print(
                f"{self._log_prefix('init')} first_ann={self._get_ann_id(first_ann)} "
                f"cameras={first_ann.get('camera_order', list(first_ann['videos'].keys()))} "
                f"num_frames={first_ann['num_frames']} actions={first_ann['actions']}"
            )

    def _maybe_log_loaded_sample(self, index, ann, frame_start, frame_end, data):
        if self.dataset_log_first_n <= 0 or self._logged_samples >= self.dataset_log_first_n:
            return

        worker_info = get_worker_info()
        if worker_info is not None and worker_info.id != 0:
            return

        video = data['video']
        action = data['action']
        video_name = data['video_name']
        if isinstance(video_name.get('cam_id'), list):
            view_count = len(video_name['cam_id'])
            view_desc = list(zip(video_name.get('cam_name', []), video_name['cam_id']))
        else:
            view_count = 1
            view_desc = [(video_name.get('cam_name', ''), video_name['cam_id'])]

        worker_id = worker_info.id if worker_info is not None else 'main'
        print(
            f"{self._log_prefix('getitem')} worker={worker_id} mode={self.mode} "
            f"index={index} episode={self._get_ann_id(ann)} "
            f"frame_range=[{frame_start}, {frame_end}) total_frames={ann['num_frames']} "
            f"views={view_count} view_desc={view_desc}"
        )
        print(
            f"{self._log_prefix('getitem')} video_shape={tuple(video.shape)} "
            f"action_shape={tuple(action.shape)} video_name={video_name}"
        )
        self._logged_samples += 1

    def _resolve_data_root(self):
        explicit_root = getattr(self.args, f'{self.dataset_name}_data_root', None)
        if explicit_root:
            root = Path(explicit_root)
            if root.exists():
                return root

        base_dir = Path(getattr(self.args, 'base_dir', ''))
        candidates = []
        if self.dataset_name == 'libero':
            split = str(getattr(self.args, 'libero_split', '90'))
            candidates.append(base_dir / 'work_dirs' / 'Datasets' / 'EWM_infer_meta' / 'libero' / split)
        else:
            candidates.append(base_dir / 'work_dirs' / 'Datasets' / 'EWM_infer_meta' / self.dataset_name)

        dataset_dir = getattr(self.args, 'dataset_dir', None)
        if dataset_dir:
            if self.dataset_name == 'libero':
                candidates.append(Path(dataset_dir) / 'libero' / split)
            else:
                candidates.append(Path(dataset_dir) / self.dataset_name)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Failed to locate {self.dataset_name} data root. "
            f"Set `{self.dataset_name}_data_root` to a valid directory."
        )

    def _resolve_annotation_path(self, mode):
        if mode == 'train':
            mode_name = 'train'
        else:
            mode_name = 'eval'

        explicit_ann = getattr(self.args, f'{self.dataset_name}_{mode_name}_annotation', None)
        candidates = []
        if explicit_ann:
            candidates.append(Path(explicit_ann))
        candidates.append(self.data_root / f'annotations_{mode_name}.json')

        for ann_path in candidates:
            if ann_path.exists():
                return ann_path, str(ann_path)

        if self.dataset_name == 'agibot':
            split_percent = int(getattr(self.args, 'agibot_split_percent', 90))
            return None, f'auto-scan:{mode_name}:split={split_percent}'

        raise FileNotFoundError(
            f"Missing {self.dataset_name} annotation file. Checked: "
            + ', '.join(str(path) for path in candidates)
        )

    def _load_annotations(self, annotation_path):
        if annotation_path is None:
            return self._build_annotations_from_episode_files()
        with open(annotation_path, 'r') as file:
            annotations = json.load(file)
        return annotations

    def _episode_sort_key(self, episode_name):
        if str(episode_name).isdigit():
            return (0, int(episode_name))
        return (1, str(episode_name))

    def _task_split_seed(self, task_name):
        base_seed = int(getattr(self.args, 'agibot_split_seed', 3407))
        digest = hashlib.md5(f'{task_name}:{base_seed}'.encode('utf-8')).hexdigest()
        return int(digest[:8], 16)

    def _build_annotations_from_episode_files(self):
        annotation_paths = sorted(self.data_root.glob('*/*/annotation.json'))
        if not annotation_paths:
            raise FileNotFoundError(
                f'No per-episode annotation.json files found under {self.data_root}.'
            )

        annotations_by_task = {}
        for ann_path in annotation_paths:
            with open(ann_path, 'r') as file:
                ann = json.load(file)
            task_name = ann_path.parent.parent.name
            episode_name = ann_path.parent.name
            annotations_by_task.setdefault(task_name, []).append((episode_name, ann))

        split_percent = int(getattr(self.args, 'agibot_split_percent', 90))
        if split_percent <= 0 or split_percent >= 100:
            raise ValueError(f'`agibot_split_percent` must be in (0, 100), got {split_percent}.')

        selected = []
        for task_name, task_items in sorted(annotations_by_task.items()):
            ordered_items = sorted(task_items, key=lambda item: self._episode_sort_key(item[0]))
            task_rng = random.Random(self._task_split_seed(task_name))
            task_rng.shuffle(ordered_items)

            total_items = len(ordered_items)
            eval_count = 1 if total_items > 1 else 0
            eval_count = max(eval_count, int(round(total_items * (100 - split_percent) / 100.0)))
            eval_count = min(eval_count, max(total_items - 1, 0))
            train_count = total_items - eval_count

            if self.mode == 'train':
                chosen_items = ordered_items[:train_count]
            else:
                chosen_items = ordered_items[train_count:]
                if not chosen_items and ordered_items:
                    chosen_items = ordered_items[-1:]

            selected.extend(ann for _, ann in chosen_items)

        return selected

    def _get_ann_id(self, ann):
        action_path = Path(ann['actions'])
        task_name = action_path.parent.parent.name
        episode_id = action_path.parent.name
        return f'{task_name}_{episode_id}'

    def _init_sequences(self, annotations):
        samples = []
        for ann in annotations:
            ann_id = self._get_ann_id(ann)
            total_frames = int(ann['num_frames'])
            if total_frames <= 0:
                continue

            if self.sample_strategy == 'random':
                samples.append({
                    'ann_file': ann_id,
                    'total_frames': total_frames,
                })
                continue

            frame_start = 0
            reach_end = False
            while not reach_end:
                frame_end = frame_start + self.sequence_length
                if frame_end >= total_frames:
                    reach_end = True
                samples.append({
                    'ann_file': ann_id,
                    'total_frames': total_frames,
                    'frame_interval': [frame_start, frame_end],
                })
                frame_start += self.sample_interval
        return samples

    def __len__(self):
        return len(self.samples)

    def _resolve_camera(self, ann, cam_id):
        camera_order = ann.get('camera_order', list(ann['videos'].keys()))
        if cam_id is None:
            selected = random.choice(self.cam_ids) if self.cam_ids else 0
        else:
            selected = cam_id

        if isinstance(selected, str) and selected in ann['videos']:
            cam_name = selected
            cam_index = camera_order.index(cam_name)
        else:
            cam_index = int(selected)
            if cam_index < 0 or cam_index >= len(camera_order):
                raise IndexError(f'Camera index {cam_index} is out of range for {camera_order}')
            cam_name = camera_order[cam_index]
        return cam_name, cam_index

    def _load_video_ffmpeg(self, video_path, frame_ids, max_frames):
        reader = imageio.get_reader(str(video_path), 'ffmpeg')
        try:
            usable_frames = min(reader.count_frames(), max_frames)
            if usable_frames <= 0:
                raise ValueError(f'No usable frames found in {video_path}')

            capped_frame_ids = np.clip(np.asarray(frame_ids), 0, usable_frames - 1)
            frame_data = [reader.get_data(int(frame_id)) for frame_id in capped_frame_ids.tolist()]
            return np.stack(frame_data, axis=0)
        finally:
            reader.close()

    def _load_video(self, video_path, frame_ids, max_frames):
        if self.video_reader_backend == 'ffmpeg':
            return self._load_video_ffmpeg(video_path, frame_ids, max_frames)
        if self.video_reader_backend == 'decord':
            vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
            usable_frames = min(len(vr), max_frames)
            if usable_frames <= 0:
                raise ValueError(f'No usable frames found in {video_path}')

            capped_frame_ids = np.clip(np.asarray(frame_ids), 0, usable_frames - 1)
            frame_data = vr.get_batch(capped_frame_ids.tolist()).asnumpy()
            return frame_data
        if self.video_reader_backend != 'auto':
            raise ValueError(
                f"Unsupported video_reader_backend={self.video_reader_backend}. "
                "Expected one of: auto, decord, ffmpeg."
            )

        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
        usable_frames = min(len(vr), max_frames)
        if usable_frames <= 0:
            raise ValueError(f'No usable frames found in {video_path}')

        capped_frame_ids = np.clip(np.asarray(frame_ids), 0, usable_frames - 1)
        frame_data = vr.get_batch(capped_frame_ids.tolist()).asnumpy()
        return frame_data

    def _get_frames(self, ann, frame_start, frame_end, cam_id):
        cam_name, cam_index = self._resolve_camera(ann, cam_id)
        video_path = self.data_root / ann['videos'][cam_name]
        frame_ids = list(range(frame_start, frame_end))
        try:
            frames = self._load_video(video_path, frame_ids, max_frames=int(ann['num_frames']))
        except Exception as decord_error:
            if self.video_reader_backend != 'auto':
                raise
            if str(video_path) not in self._ffmpeg_fallback_warned_paths:
                warnings.warn(
                    f"Failed to decode {video_path} with decord; "
                    f"falling back to ffmpeg/imageio. Error: {decord_error}"
                )
                self._ffmpeg_fallback_warned_paths.add(str(video_path))
            frames = self._load_video_ffmpeg(video_path, frame_ids, max_frames=int(ann['num_frames']))
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

        if self.args.normalize:
            frames = self.preprocess(frames)
        else:
            frames = self.not_norm_preprocess(frames)
            frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)

        return frames, cam_name, cam_index

    def _get_actions(self, ann, frame_start, frame_end):
        action_path = self.data_root / ann['actions']
        actions = np.load(action_path)
        max_actions = min(actions.shape[0], max(int(ann['num_frames']) - 1, 0))
        clipped_end = min(frame_end - 1, max_actions)

        if clipped_end > frame_start:
            action_clip = actions[frame_start:clipped_end]
        else:
            action_clip = np.zeros((0, self.action_dim), dtype=actions.dtype)

        expected_actions = self.sequence_length - 1
        if action_clip.shape[0] < expected_actions:
            padding = np.zeros(
                (expected_actions - action_clip.shape[0],) + action_clip.shape[1:],
                dtype=actions.dtype,
            )
            action_clip = np.concatenate([action_clip, padding], axis=0)
        elif action_clip.shape[0] > expected_actions:
            action_clip = action_clip[:expected_actions]

        return torch.from_numpy(action_clip)

    def __getitem__(self, index, cam_id=None, return_video=False):
        if self.mode != 'train':
            np.random.seed(index)
            random.seed(index)

        try:
            sample = self.samples[index]
            ann = self.ann_dict[sample['ann_file']]

            if self.sample_strategy == 'random':
                total_frames = sample['total_frames']
                max_start = max(0, total_frames - self.sequence_length // 2)
                frame_start = np.random.randint(0, max_start + 1)
                frame_end = frame_start + self.sequence_length
            else:
                frame_start, frame_end = sample['frame_interval']

            actions = self._get_actions(ann, frame_start, frame_end)
            actions = actions * self.c_act_scaler

            if self.use_all_views and cam_id is None:
                view_ids = self.cam_ids or ann.get('camera_order', list(ann['videos'].keys()))
                videos = []
                cam_names = []
                cam_indices = []
                for view_id in view_ids:
                    view_video, cam_name, cam_index = self._get_frames(ann, frame_start, frame_end, view_id)
                    videos.append(view_video)
                    cam_names.append(cam_name)
                    cam_indices.append(cam_index)

                view_count = len(videos)
                data = {
                    'action': actions.unsqueeze(0).repeat(view_count, 1, 1).float(),
                    'video': torch.stack(videos, dim=0).float(),
                    'video_name': {
                        'episode_id': [self._get_ann_id(ann)] * view_count,
                        'start_frame_id': [str(frame_start)] * view_count,
                        'cam_id': [str(cam_index) for cam_index in cam_indices],
                        'cam_name': cam_names,
                    },
                }
            else:
                video, _, cam_index = self._get_frames(ann, frame_start, frame_end, cam_id)
                data = {
                    'action': actions.float(),
                    'video': video.float(),
                    'video_name': {
                        'episode_id': self._get_ann_id(ann),
                        'start_frame_id': str(frame_start),
                        'cam_id': str(cam_index),
                    },
                }
            self._maybe_log_loaded_sample(index, ann, frame_start, frame_end, data)
            return data
        except Exception:
            warnings.warn(
                f"Invalid data encountered: {self.samples[index]['ann_file']}. "
                "Skipped (by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.samples))]
