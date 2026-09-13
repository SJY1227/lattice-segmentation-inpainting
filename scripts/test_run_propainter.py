import tempfile
import unittest
from pathlib import Path

from run_propainter import build_command, parse_args


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder, ext in (("frames", "jpg"), ("masks", "png")):
            (self.root / folder).mkdir()
            for index in range(2):
                (self.root / folder / f"{index:05d}.{ext}").touch()
        self.args = parse_args(["--frames-dir", str(self.root / "frames"), "--masks-dir", str(self.root / "masks"),
                                "--output-dir", str(self.root / "output"), "--fps", "56", "--cpu-offload"])

    def test_absolute_paths_and_temporal_settings(self):
        command, output, count = build_command(self.args)
        self.assertEqual(count, 2)
        self.assertTrue(output.is_absolute())
        self.assertIn("--cpu_offload", command)
        self.assertEqual(command[command.index("--neighbor_length") + 1], "10")
        self.assertEqual(command[command.index("--mask_dilation") + 1], "2")

    def test_mismatched_counts(self):
        (self.root / "masks/00001.png").unlink()
        with self.assertRaises(ValueError):
            build_command(self.args)

    def test_rejects_non_image_files(self):
        (self.root / "frames/metadata.json").touch()
        with self.assertRaises(ValueError):
            build_command(self.args)

    def test_refuses_existing_results(self):
        (self.root / "output").mkdir()
        (self.root / "output/result.mp4").touch()
        with self.assertRaises(ValueError):
            build_command(self.args)

    def test_invalid_parameters(self):
        for name, value in (("neighbor_length", 0), ("neighbor_length", 3), ("ref_stride", 0), ("width", 511),
                            ("subvideo_length", 1), ("raft_iter", -1), ("mask_dilation", -1), ("fps", 0)):
            with self.subTest(name=name, value=value):
                old = getattr(self.args, name)
                setattr(self.args, name, value)
                with self.assertRaises(ValueError):
                    build_command(self.args)
                setattr(self.args, name, old)


if __name__ == "__main__":
    unittest.main()
