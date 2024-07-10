import decord

decord.bridge.set_bridge("torch")
import numpy as np
from torch.utils.data import Dataset
from einops import rearrange


class TuneAVideoDataset(Dataset):
    def __init__(
        self,
        video_path: str,
        prompt: str,
        width: int = 512,
        height: int = 512,
        n_sample_frames: int = 8,
        sample_start_idx: int = 0,
        sample_frame_rate: int = 1,
        mask_l: float = 0,
        mask_r: float = 0,
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        self.mask_l = int(mask_l * width)
        self.mask_r = int(mask_r * width)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # load and sample video frames

        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        sample_index = list(
            range(self.sample_start_idx, len(vr), self.sample_frame_rate)
        )[: self.n_sample_frames]
        assert len(sample_index) == self.n_sample_frames
        if len(sample_index) < self.n_sample_frames:
            # Calculate the number of frames to duplicate
            missing_frames = self.n_sample_frames - len(sample_index)

            # Duplicate some frames to reach 16 frames
            # For simplicity, duplicate the last frame
            last_frame_index = sample_index[-1] if sample_index else 0
            sample_index.extend([last_frame_index] * missing_frames)
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        video = (
            video[..., self.mask_l :]
            if self.mask_r == 0
            else video[..., self.mask_l : -self.mask_r]
        )

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids,
            "video": video,
        }

        return example


MAX_LEN = 64


class TuneAVideoDatasetSplit(Dataset):
    def __init__(
        self,
        video_path: str,
        prompt: str,
        width: int = 512,
        height: int = 512,
        stride: int = 1,
        n_sample_frames: int = 8,
        sample_start_idx: int = 0,
        sample_frame_rate: int = 1,
    ):
        self.video_path = video_path
        if isinstance(prompt, str):
            self.prompt = prompt
        self.prompt_ids = None
        self.null_prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        self.stride = stride
        assert self.stride == 1
        self.vr = decord.VideoReader(
            self.video_path, width=self.width, height=self.height
        )
        self.sample_index = list(
            range(self.sample_start_idx, len(self.vr), self.sample_frame_rate)
        )
        self.full_video = rearrange(
            self.vr.get_batch(self.sample_index), "f h w c -> f c h w"
        )
        self.len_video = len(self.sample_index)
        self.len_video = min(self.len_video, MAX_LEN)
        self.len_video = self.len_video - self.len_video % self.n_sample_frames
        self.sample_index = self.sample_index[: self.len_video]
        self.full_video = self.full_video[: self.len_video]
        self.len_dataset = self.len_video - self.n_sample_frames + 1

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        prompt_ids = (
            self.prompt_ids if isinstance(self.prompt, str) else self.prompt_ids[index]
        )
        # index = index * self.stride
        index = np.random.randint(self.len_dataset)
        sample_index = self.sample_index[index : index + self.n_sample_frames]
        video = self.vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids,
            "video_length": self.len_video,
            "full_video": (self.full_video / 127.5 - 1.0),
            "clip_id": index,
        }

        return example
