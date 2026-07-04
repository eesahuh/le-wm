"""PyTorch dataset adapter for the local PiperX LeRobot export.

The adapter reads LeRobot parquet metadata/actions and decodes temporal windows
from ``videos/observation.image``. It intentionally ignores
``observation.extra_image`` for this first pass.

Example:
    dataset = PiperXLeRobotDataset(
        "/Users/isahe/le-wm/lerobot_data/static_eraser_pnp_v30_fps30",
        num_steps=4,
    )
    sample = dataset[0]
    assert sample["pixels"].shape == (4, 3, 224, 224)
    assert sample["action"].shape == (4, 7)
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_DATASET_ROOT = (
    Path(__file__).resolve().parent
    / "lerobot_data"
    / "static_eraser_pnp_v30_fps30"
)

DEFAULT_CAMERA_KEY = "observation.image"

ACTION_NAMES = (
    "left.joint1.pos",
    "left.joint2.pos",
    "left.joint3.pos",
    "left.joint4.pos",
    "left.joint5.pos",
    "left.joint6.pos",
    "left.gripper.pos",
)


@dataclass(frozen=True)
class EpisodeWindowSource:
    episode_index: int
    dataset_from_index: int
    dataset_to_index: int
    video_path: Path
    video_frame_offset: int

    @property
    def length(self) -> int:
        return self.dataset_to_index - self.dataset_from_index


@dataclass(frozen=True)
class WindowIndex:
    episode_source_index: int
    local_start: int


class _VideoFrameReader:
    """Small lazy video reader with optional backends.

    Prefer installing one of ``decord``, ``opencv-python``, or
    ``imageio[ffmpeg]`` for training. A system ``ffmpeg`` executable is also
    supported as a slow fallback.
    """

    def __init__(
        self,
        path: Path,
        *,
        fps: float,
        frame_shape: Tuple[int, int, int],
        backend: str = "auto",
    ) -> None:
        self.path = path
        self.fps = fps
        self.height, self.width, self.channels = frame_shape
        self.backend = self._resolve_backend(backend)
        self._reader: Any = None

    def _resolve_backend(self, backend: str) -> str:
        if backend != "auto":
            return backend

        if _can_import("decord"):
            return "decord"
        if _can_import("cv2"):
            return "cv2"
        if _can_import("imageio_ffmpeg"):
            return "imageio"
        if shutil.which("ffmpeg"):
            return "ffmpeg"
        if _can_import("imageio"):
            return "imageio"
        return "missing"

    def get_batch(self, frame_indices: Sequence[int]) -> np.ndarray:
        indices = [int(i) for i in frame_indices]
        if self.backend == "decord":
            return self._read_decord(indices)
        if self.backend == "cv2":
            return self._read_cv2(indices)
        if self.backend == "imageio":
            return self._read_imageio(indices)
        if self.backend == "ffmpeg":
            return self._read_ffmpeg(indices)

        raise ImportError(
            "No video decoder is available. Install one of: decord, "
            "opencv-python, imageio[ffmpeg], or make ffmpeg available on PATH."
        )

    def _read_decord(self, indices: Sequence[int]) -> np.ndarray:
        if self._reader is None:
            import decord

            self._reader = decord.VideoReader(str(self.path), ctx=decord.cpu(0))
        return self._reader.get_batch(list(indices)).asnumpy()

    def _read_cv2(self, indices: Sequence[int]) -> np.ndarray:
        import cv2

        if self._reader is None:
            self._reader = cv2.VideoCapture(str(self.path))
            if not self._reader.isOpened():
                raise RuntimeError(f"Could not open video: {self.path}")

        frames = []
        for frame_idx in indices:
            self._reader.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self._reader.read()
            if not ok:
                raise IndexError(f"Could not read frame {frame_idx} from {self.path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.stack(frames, axis=0)

    def _read_imageio(self, indices: Sequence[int]) -> np.ndarray:
        if self._reader is None:
            import imageio

            try:
                self._reader = imageio.get_reader(str(self.path), format="ffmpeg")
            except Exception as exc:  # pragma: no cover - backend dependent
                raise ImportError(
                    "imageio is installed, but its ffmpeg plugin is unavailable. "
                    "Install imageio[ffmpeg], or install decord/opencv-python."
                ) from exc

        frames = []
        for frame_idx in indices:
            frames.append(np.asarray(self._reader.get_data(frame_idx)))
        return np.stack(frames, axis=0)

    def _read_ffmpeg(self, indices: Sequence[int]) -> np.ndarray:
        frames = [self._read_one_frame_ffmpeg(frame_idx) for frame_idx in indices]
        return np.stack(frames, axis=0)

    def _read_one_frame_ffmpeg(self, frame_idx: int) -> np.ndarray:
        timestamp = frame_idx / self.fps
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.9f}",
            "-i",
            str(self.path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        raw = subprocess.check_output(cmd)
        expected = self.height * self.width * self.channels
        if len(raw) != expected:
            raise RuntimeError(
                f"ffmpeg returned {len(raw)} bytes for frame {frame_idx}; "
                f"expected {expected}"
            )
        return np.frombuffer(raw, dtype=np.uint8).reshape(
            self.height, self.width, self.channels
        )


class PiperXLeRobotDataset(Dataset):
    """Temporal-window dataset for PiperX LeRobot recordings.

    Each item returns:
        ``pixels``: float tensor with shape ``(T, 3, image_size, image_size)``
        ``action``: float tensor with shape ``(T, 7)``

    Windows are sampled within a single episode. By default, ``T=4`` to match
    the current LeWM config's ``history_size=3`` and ``num_preds=1``.
    """

    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_DATASET_ROOT,
        *,
        num_steps: Optional[int] = None,
        window_size: Optional[int] = None,
        image_size: Union[int, Tuple[int, int]] = 224,
        frame_stride: int = 1,
        frameskip: Optional[int] = None,
        camera_key: str = DEFAULT_CAMERA_KEY,
        action_key: str = "action",
        transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, Any]]] = None,
        include_metadata: bool = False,
        normalize_pixels: bool = False,
        video_backend: str = "auto",
        validate_videos: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.num_steps = int(window_size or num_steps or 4)
        self.image_size = _normalize_image_size(image_size)
        self.frame_stride = int(frameskip if frameskip is not None else frame_stride)
        self.camera_key = camera_key
        self.action_key = action_key
        self.transform = transform
        self.include_metadata = include_metadata
        self.normalize_pixels = normalize_pixels
        self.video_backend = video_backend
        self.validate_videos = validate_videos
        self.column_names = [
            "pixels",
            "action",
            "observation.state",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]

        if self.num_steps <= 0:
            raise ValueError("num_steps/window_size must be positive")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride/frameskip must be positive")

        self.info = self._load_info()
        self.fps = float(self.info.get("fps") or self._feature_info(camera_key)["fps"])
        self.video_shape = tuple(self._feature_info(camera_key)["shape"])
        if len(self.video_shape) != 3:
            raise ValueError(f"Expected HWC video shape, got {self.video_shape}")

        self.actions: np.ndarray
        self.observation_state: np.ndarray
        self.timestamps: np.ndarray
        self.frame_indices: np.ndarray
        self.episode_indices: np.ndarray
        self.global_indices: np.ndarray
        self.task_indices: np.ndarray
        self._global_to_row: np.ndarray
        self._load_data_table()

        self.episode_sources = self._load_episode_sources()
        self.windows = self._build_windows()
        if not self.windows:
            raise ValueError(
                "No valid windows found. Reduce num_steps or frame_stride, "
                "or check episode lengths."
            )

        self._video_readers: Dict[Path, _VideoFrameReader] = {}

    def __len__(self) -> int:
        return len(self.windows)

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_video_readers"] = {}
        return state

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        window = self.windows[idx]
        source = self.episode_sources[window.episode_source_index]
        local_offsets = window.local_start + np.arange(self.num_steps) * self.frame_stride
        global_indices = source.dataset_from_index + local_offsets
        row_indices = self._rows_for_global_indices(global_indices)
        video_indices = source.video_frame_offset + local_offsets

        frames = self._read_video_frames(source.video_path, video_indices)
        pixels = self._frames_to_tensor(frames)
        action = torch.from_numpy(self.actions[row_indices]).float()

        sample: Dict[str, Any] = {
            "pixels": pixels,
            "action": action,
        }

        if self.include_metadata:
            sample.update(
                {
                    "episode_index": torch.tensor(source.episode_index, dtype=torch.long),
                    "frame_index": torch.from_numpy(self.frame_indices[row_indices]).long(),
                    "index": torch.from_numpy(self.global_indices[row_indices]).long(),
                    "timestamp": torch.from_numpy(self.timestamps[row_indices]).float(),
                }
            )

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_dim(self, column: str) -> int:
        if column == "action":
            return int(self.actions.shape[-1])
        if column == "observation.state":
            return int(self.observation_state.shape[-1])
        raise KeyError(f"Unsupported column for get_dim: {column}")

    def get_col_data(self, column: str) -> np.ndarray:
        if column == "action":
            return self.actions
        if column == "observation.state":
            return self.observation_state
        if column == "timestamp":
            return self.timestamps
        if column == "frame_index":
            return self.frame_indices
        if column == "episode_index":
            return self.episode_indices
        if column == "index":
            return self.global_indices
        if column == "task_index":
            return self.task_indices
        raise KeyError(f"Unsupported column for get_col_data: {column}")

    def _load_info(self) -> Dict[str, Any]:
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing LeRobot info file: {info_path}")
        return json.loads(info_path.read_text())

    def _feature_info(self, key: str) -> Dict[str, Any]:
        features = self.info.get("features", {})
        if key not in features:
            raise KeyError(f"Feature {key!r} not found in {self.root / 'meta/info.json'}")
        return features[key]

    def _load_data_table(self) -> None:
        data_paths = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        if not data_paths:
            raise FileNotFoundError(f"No data parquet files found under {self.root / 'data'}")

        tables = [
            pq.read_table(
                path,
                columns=[
                    "observation.state",
                    self.action_key,
                    "timestamp",
                    "frame_index",
                    "episode_index",
                    "index",
                    "task_index",
                ],
            )
            for path in data_paths
        ]
        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        data = table.to_pydict()

        self.actions = np.asarray(data[self.action_key], dtype=np.float32)
        if self.actions.ndim != 2 or self.actions.shape[1] != len(ACTION_NAMES):
            raise ValueError(
                f"Expected action shape (N, {len(ACTION_NAMES)}), got {self.actions.shape}"
            )

        self.observation_state = np.asarray(data["observation.state"], dtype=np.float32)
        self.timestamps = np.asarray(data["timestamp"], dtype=np.float32)
        self.frame_indices = np.asarray(data["frame_index"], dtype=np.int64)
        self.episode_indices = np.asarray(data["episode_index"], dtype=np.int64)
        self.global_indices = np.asarray(data["index"], dtype=np.int64)
        self.task_indices = np.asarray(data["task_index"], dtype=np.int64)

        if (self.global_indices < 0).any():
            raise ValueError("Dataset indices must be non-negative")
        max_index = int(self.global_indices.max())
        self._global_to_row = np.full(max_index + 1, -1, dtype=np.int64)
        self._global_to_row[self.global_indices] = np.arange(len(self.global_indices))

    def _load_episode_sources(self) -> List[EpisodeWindowSource]:
        episode_paths = sorted(
            (self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet")
        )
        if not episode_paths:
            raise FileNotFoundError(
                f"No episode parquet files found under {self.root / 'meta' / 'episodes'}"
            )

        tables = [pq.read_table(path) for path in episode_paths]
        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        rows = sorted(table.to_pylist(), key=lambda row: row["dataset_from_index"])

        sources = []
        chunk_col = f"videos/{self.camera_key}/chunk_index"
        file_col = f"videos/{self.camera_key}/file_index"
        from_ts_col = f"videos/{self.camera_key}/from_timestamp"

        for row in rows:
            missing = [col for col in (chunk_col, file_col, from_ts_col) if col not in row]
            if missing:
                raise KeyError(
                    f"Episode metadata is missing camera columns for {self.camera_key}: "
                    f"{missing}"
                )

            chunk_index = int(row[chunk_col])
            file_index = int(row[file_col])
            video_path = (
                self.root
                / "videos"
                / self.camera_key
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            if self.validate_videos and not video_path.exists():
                raise FileNotFoundError(
                    f"Missing video for {self.camera_key}: {video_path}. "
                    "This adapter intentionally ignores observation.extra_image."
                )

            sources.append(
                EpisodeWindowSource(
                    episode_index=int(row["episode_index"]),
                    dataset_from_index=int(row["dataset_from_index"]),
                    dataset_to_index=int(row["dataset_to_index"]),
                    video_path=video_path,
                    video_frame_offset=int(round(float(row[from_ts_col]) * self.fps)),
                )
            )
        return sources

    def _build_windows(self) -> List[WindowIndex]:
        windows: List[WindowIndex] = []
        required_span = (self.num_steps - 1) * self.frame_stride + 1
        for source_idx, source in enumerate(self.episode_sources):
            num_starts = source.length - required_span + 1
            if num_starts <= 0:
                continue
            windows.extend(
                WindowIndex(source_idx, local_start)
                for local_start in range(num_starts)
            )
        return windows

    def _rows_for_global_indices(self, global_indices: np.ndarray) -> np.ndarray:
        if int(global_indices.max()) >= len(self._global_to_row):
            raise IndexError(f"Global index out of range: {int(global_indices.max())}")
        rows = self._global_to_row[global_indices]
        if (rows < 0).any():
            missing = global_indices[rows < 0]
            raise IndexError(f"Missing data rows for global indices: {missing.tolist()}")
        return rows

    def _read_video_frames(
        self, video_path: Path, video_indices: Sequence[int]
    ) -> np.ndarray:
        reader = self._video_readers.get(video_path)
        if reader is None:
            reader = _VideoFrameReader(
                video_path,
                fps=self.fps,
                frame_shape=self.video_shape,
                backend=self.video_backend,
            )
            self._video_readers[video_path] = reader
        return reader.get_batch(video_indices)

    def _frames_to_tensor(self, frames: np.ndarray) -> torch.Tensor:
        resized = []
        for frame in frames:
            image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
            image = image.resize(self.image_size, Image.BILINEAR)
            resized.append(np.asarray(image, dtype=np.uint8))

        array = np.stack(resized, axis=0)
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float().div(255.0)

        if self.normalize_pixels:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
            tensor = (tensor - mean) / std
        return tensor


def _normalize_image_size(image_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(image_size, int):
        return (image_size, image_size)
    if len(image_size) != 2:
        raise ValueError("image_size must be an int or a (height, width) tuple")
    height, width = int(image_size[0]), int(image_size[1])
    return (width, height)


def _can_import(module_name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module_name) is not None


if __name__ == "__main__":
    dataset = PiperXLeRobotDataset()
    print(f"windows: {len(dataset)}")
    sample = dataset[0]
    print("pixels", tuple(sample["pixels"].shape), sample["pixels"].dtype)
    print("action", tuple(sample["action"].shape), sample["action"].dtype)
