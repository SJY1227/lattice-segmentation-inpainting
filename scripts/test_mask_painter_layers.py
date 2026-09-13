import base64
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from mask_painter_layers import LAYER_IDS, LayerStore, data_url, png_bytes, read_image


class LayerStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frame = self.root / "frame.png"
        self.frame.write_bytes(png_bytes(np.full((64, 64, 3), 100, np.uint8)))
        self.seed = self.root / "seed.png"
        seed = np.zeros((64, 64), np.uint8)
        seed[10:20, 10:20] = 255
        self.seed.write_bytes(png_bytes(seed))
        self.original = self.seed.read_bytes()
        self.store = LayerStore(self.root / "layers", self.frame, self.seed)

    def payload(self):
        masks = {}
        for index, key in enumerate(LAYER_IDS):
            rgba = np.zeros((64, 64, 4), np.uint8)
            rgba[20:30, 20:30] = [70, 255, 85, 255]
            rgba[40 + index, 40 + index] = [70, 255, 85, 1]
            masks[key] = data_url(png_bytes(rgba))
        return {"base_revision": self.store.manifest()["revision"], "layers": masks,
                "included": ["near", "far"]}

    def test_seed_preserved_and_restart_does_not_reseed(self):
        self.assertEqual(self.seed.read_bytes(), self.original)
        self.assertEqual(np.count_nonzero(read_image(self.store.directory / "near.png")), 100)
        self.assertEqual(np.count_nonzero(read_image(self.store.directory / "far.png")), 0)
        result = self.store.save(self.payload())
        reopened = LayerStore(self.store.directory, self.frame, self.seed)
        self.assertEqual(reopened.manifest()["revision"], result["revision"])
        self.assertEqual(self.seed.read_bytes(), self.original)

    def test_overlap_and_selected_union_and_history(self):
        initial = self.store.manifest()["revision"]
        result = self.store.save(self.payload())
        masks = {key: read_image(path) for key, path in result["layers"].items()}
        for mask in masks.values():
            self.assertEqual(mask.shape, (64, 64))
            self.assertEqual(set(np.unique(mask)), {0, 255})
            self.assertEqual(mask[25, 25], 255)
        np.testing.assert_array_equal(read_image(result["mask"]), masks["near"] | masks["far"])
        self.assertEqual(read_image(result["mask"])[42, 42], 0)
        self.assertTrue((self.store.directory / "history" / initial / "near.png").exists())
        self.assertEqual(np.count_nonzero(read_image(self.store.directory / "history" / initial / "far.png")), 0)

    def test_invalid_payloads_write_nothing(self):
        original = self.store.manifest_path.read_bytes()
        invalid = [
            {"layers": {"near": "garbage"}},
            {"included": ["unknown"]},
            {"included": ["near", "near"]},
            {"included": [1]},
            {"included": None},
            {"base_revision": "stale"},
        ]
        for changes in invalid:
            payload = self.payload()
            payload.update(changes)
            with self.assertRaises(ValueError):
                self.store.save(payload)
            self.assertEqual(self.store.manifest_path.read_bytes(), original)
        for image in [np.zeros((32, 32, 4), np.uint8), np.zeros((64, 64, 3), np.uint8)]:
            payload = self.payload()
            payload["layers"]["reflection"] = data_url(png_bytes(image))
            with self.assertRaises(ValueError):
                self.store.save(payload)
            self.assertEqual(self.store.manifest_path.read_bytes(), original)

    def test_stale_editor_cannot_overwrite(self):
        payload = self.payload()
        result = self.store.save(payload)
        with self.assertRaisesRegex(ValueError, "newer save"):
            self.store.save(payload)
        self.assertEqual(self.store.manifest()["revision"], result["revision"])

    def test_empty_union_and_load(self):
        payload = self.payload()
        payload["included"] = []
        result = self.store.save(payload)
        self.assertEqual(np.count_nonzero(read_image(result["mask"])), 0)
        loaded = self.store.load()
        self.assertEqual(loaded["included"], [])
        for key, value in loaded["layers"].items():
            raw = base64.b64decode(value.split(",", 1)[1])
            self.assertEqual(raw, Path(result["layers"][key]).read_bytes())

    def test_different_frame_rejected(self):
        self.frame.write_bytes(png_bytes(np.full((64, 64, 3), 200, np.uint8)))
        with self.assertRaisesRegex(ValueError, "different reference"):
            LayerStore(self.store.directory, self.frame, self.seed)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LayerStoreTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
