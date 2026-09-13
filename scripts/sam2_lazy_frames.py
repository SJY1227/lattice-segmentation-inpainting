"""Bounded JPEG loading with the same preprocessing as SAM2's eager loader."""

from collections import OrderedDict
from unittest.mock import patch

import torch
from sam2.utils.misc import _load_img_as_tensor


class LazyJpegFrames:
    def __init__(self, paths, image_size, offload_video_to_cpu=True,
                 img_mean=(0.485, 0.456, 0.406), img_std=(0.229, 0.224, 0.225),
                 compute_device=torch.device("cuda"), cache_size=2):
        if not paths or cache_size < 1:
            raise ValueError("At least one frame and one cache slot are required")
        self.paths = list(paths)
        self.image_size = image_size
        self.device = torch.device("cpu") if offload_video_to_cpu else compute_device
        self.mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.height = self.width = None
        self[0]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if index < 0:
            index += len(self.paths)
        if not 0 <= index < len(self.paths):
            raise IndexError(index)
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        image, height, width = _load_img_as_tensor(str(self.paths[index]), self.image_size)
        if self.height is not None and (height, width) != (self.height, self.width):
            raise ValueError("Frame dimensions changed within the sequence")
        self.height, self.width = height, width
        # The eager loader casts into its float32 allocation before normalizing.
        image = image.float()
        image.sub_(self.mean).div_(self.std)
        image = image.to(self.device)
        self.cache[index] = image
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return image


def init_lazy_state(predictor, frame_paths, video_path, offload_state_to_cpu=False):
    def load_frames(**kwargs):
        images = LazyJpegFrames(
            frame_paths, kwargs["image_size"],
            offload_video_to_cpu=kwargs["offload_video_to_cpu"],
            compute_device=kwargs["compute_device"],
        )
        return images, images.height, images.width

    # Substitute only the frame I/O for this initialization, restoring it immediately.
    # All predictor state, memory, prompts and propagation remain SAM2's implementation.
    with patch("sam2.sam2_video_predictor.load_video_frames", load_frames):
        return predictor.init_state(
            video_path=video_path, offload_video_to_cpu=True,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=False,
        )
