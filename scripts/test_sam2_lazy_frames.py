import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam2.utils.misc import load_video_frames

from sam2_lazy_frames import LazyJpegFrames


class LazyFrameTests(unittest.TestCase):
    def test_matches_eager_loader_and_evicts(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            rng = np.random.default_rng(17)
            for index in range(4):
                path = Path(folder) / f"{index:05d}.jpg"
                Image.fromarray(rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)).save(path)
                paths.append(path)
            eager, height, width = load_video_frames(str(folder), 128, True)
            lazy = LazyJpegFrames(paths, 128)
            self.assertEqual((lazy.height, lazy.width), (height, width))
            self.assertEqual(len(lazy), len(eager))
            for index in [0, 1, 2, 3, 1, 0, -1]:
                self.assertTrue(torch.equal(lazy[index], eager[index]))
                self.assertLessEqual(len(lazy.cache), 2)
            with self.assertRaises(IndexError):
                lazy[4]


if __name__ == "__main__":
    unittest.main()
