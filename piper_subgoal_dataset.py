"""PiperX LeRobot dataset for latent subgoal predictor training.

This dataset reuses :class:`PiperXLeRobotDataset` to decode frames from the
same local LeRobot export, but changes the sampling contract:

    history_pixels = frames t-history_size+1 ... t
    future_pixels = frames t+offset for offset in subgoal_offsets

Windows are built inside a single episode by the underlying Piper dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from piper_lerobot_dataset import (
    DEFAULT_CAMERA_KEY,
    DEFAULT_DATASET_ROOT,
    PiperXLeRobotDataset,
)


DEFAULT_SUBGOAL_OFFSETS = (5, 10, 15, 20, 30)


def parse_subgoal_offsets(offsets: Union[str, Sequence[int]]) -> List[int]:
    """Parse comma-separated or sequence-style future frame offsets."""

    if isinstance(offsets, str):
        values = [part.strip() for part in offsets.split(",") if part.strip()]
        parsed = [int(value) for value in values]
    else:
        parsed = [int(value) for value in offsets]

    if not parsed:
        raise ValueError("subgoal_offsets must contain at least one offset")
    if any(offset <= 0 for offset in parsed):
        raise ValueError(f"subgoal_offsets must be positive, got {parsed}")
    return parsed


class PiperSubgoalDataset(Dataset):
    """Episode-local Piper windows for future latent subgoal supervision.

    Each sample returns:
        ``history_pixels``: ``(history_size, 3, image_size, image_size)``
        ``future_pixels``: ``(num_subgoals, 3, image_size, image_size)``

    The default pixel preprocessing matches Piper LeWM training:
    RGB resize to 224 and ImageNet normalization.
    """

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_DATASET_ROOT,
        *,
        history_size: int = 10,
        subgoal_offsets: Union[str, Sequence[int]] = DEFAULT_SUBGOAL_OFFSETS,
        image_key: str = DEFAULT_CAMERA_KEY,
        image_size: int = 224,
        normalize_pixels: bool = True,
        video_backend: str = "auto",
        validate_videos: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.history_size = int(history_size)
        self.subgoal_offsets = parse_subgoal_offsets(subgoal_offsets)
        self.image_key = image_key
        self.image_size = int(image_size)
        self.normalize_pixels = bool(normalize_pixels)
        self.video_backend = video_backend

        if self.history_size <= 0:
            raise ValueError("history_size must be positive")

        self.future_indices = torch.tensor(
            [self.history_size - 1 + offset for offset in self.subgoal_offsets],
            dtype=torch.long,
        )
        self._history_local_indices = np.arange(self.history_size, dtype=np.int64)
        self._future_local_indices = np.asarray(
            [self.history_size - 1 + offset for offset in self.subgoal_offsets],
            dtype=np.int64,
        )
        self._selected_local_indices = np.concatenate(
            [self._history_local_indices, self._future_local_indices]
        )
        window_steps = self.history_size + max(self.subgoal_offsets)

        self.base_dataset = PiperXLeRobotDataset(
            root=self.root,
            num_steps=window_steps,
            image_size=self.image_size,
            frame_stride=1,
            camera_key=self.image_key,
            include_metadata=True,
            normalize_pixels=self.normalize_pixels,
            video_backend=self.video_backend,
            validate_videos=validate_videos,
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = self.base_dataset
        window = base.windows[idx]
        source = base.episode_sources[window.episode_source_index]
        local_offsets = window.local_start + self._selected_local_indices
        global_indices = source.dataset_from_index + local_offsets
        row_indices = base._rows_for_global_indices(global_indices)
        video_indices = source.video_frame_offset + local_offsets

        frames = base._read_video_frames(source.video_path, video_indices)
        pixels = base._frames_to_tensor(frames)
        action = torch.from_numpy(base.actions[row_indices]).float()
        history_pixels = pixels[: self.history_size]
        future_pixels = pixels[self.history_size :]

        output: Dict[str, Any] = {
            "history_pixels": history_pixels,
            "future_pixels": future_pixels,
            "subgoal_offsets": torch.tensor(self.subgoal_offsets, dtype=torch.long),
            "history_action": action[: self.history_size],
            "future_action": action[self.history_size :],
            "episode_index": torch.tensor(source.episode_index, dtype=torch.long),
            "history_frame_index": torch.from_numpy(
                base.frame_indices[row_indices[: self.history_size]]
            ).long(),
            "current_frame_index": torch.tensor(
                int(base.frame_indices[row_indices[self.history_size - 1]]),
                dtype=torch.long,
            ),
            "future_frame_index": torch.from_numpy(
                base.frame_indices[row_indices[self.history_size :]]
            ).long(),
            "history_index": torch.from_numpy(
                base.global_indices[row_indices[: self.history_size]]
            ).long(),
            "future_index": torch.from_numpy(
                base.global_indices[row_indices[self.history_size :]]
            ).long(),
            "current_index": torch.tensor(
                int(base.global_indices[row_indices[self.history_size - 1]]),
                dtype=torch.long,
            ),
        }

        return output


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test Piper subgoal windows.")
    parser.add_argument("--root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--history-size", type=int, default=10)
    parser.add_argument("--subgoal-offsets", default="5,10,15,20,30")
    parser.add_argument("--image-key", default=DEFAULT_CAMERA_KEY)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--video-backend", default="auto")
    parser.add_argument(
        "--no-normalize-pixels",
        action="store_true",
        help="Return pixels in [0, 1] instead of ImageNet-normalized pixels.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    dataset = PiperSubgoalDataset(
        root=args.root,
        history_size=args.history_size,
        subgoal_offsets=args.subgoal_offsets,
        image_key=args.image_key,
        image_size=args.image_size,
        normalize_pixels=not args.no_normalize_pixels,
        video_backend=args.video_backend,
    )
    sample = dataset[0]
    print(f"windows: {len(dataset)}")
    print("history_pixels", tuple(sample["history_pixels"].shape), sample["history_pixels"].dtype)
    print("future_pixels", tuple(sample["future_pixels"].shape), sample["future_pixels"].dtype)
    print("subgoal_offsets", sample["subgoal_offsets"].tolist())
    print("episode_index", int(sample["episode_index"]))
    print("current_frame_index", int(sample["current_frame_index"]))
    print("future_frame_index", sample["future_frame_index"].tolist())


if __name__ == "__main__":
    main()
