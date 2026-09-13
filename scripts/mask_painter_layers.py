"""Independent, overlapping annotation layers for the local mask painter."""

import base64
import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


LAYERS = [
    {"id": "near", "label": "Near lattice", "color": [70, 255, 85]},
    {"id": "far", "label": "Far lattice", "color": [61, 192, 255]},
    {"id": "reflection", "label": "Reflection", "color": [255, 111, 186]},
]
LAYER_IDS = [layer["id"] for layer in LAYERS]


def read_image(path, flags=cv2.IMREAD_UNCHANGED):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), flags)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def png_bytes(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Cannot encode mask PNG")
    return encoded.tobytes()


def data_url(raw):
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


class LayerStore:
    def __init__(self, directory, frame_path, seed_path=None):
        self.directory = Path(directory).resolve()
        self.frame_path = Path(frame_path).resolve()
        self.frame = read_image(self.frame_path, cv2.IMREAD_COLOR)
        self.frame_sha256 = hashlib.sha256(self.frame_path.read_bytes()).hexdigest()
        self.lock = threading.Lock()
        self.manifest_path = self.directory / "layers.json"
        if self.manifest_path.exists():
            manifest = self.manifest()
            if manifest["frame_sha256"] != self.frame_sha256:
                raise ValueError("Layer directory belongs to a different reference frame")
            self.current_files(manifest)
            return
        if self.directory.exists() and any(self.directory.iterdir()):
            raise ValueError("Nonempty layer directory has no layers.json; refusing to overwrite")
        self.directory.mkdir(parents=True, exist_ok=True)
        masks = {key: np.zeros(self.frame.shape[:2], np.uint8) for key in LAYER_IDS}
        seed = None
        if seed_path and Path(seed_path).exists():
            path = Path(seed_path).resolve()
            original = path.read_bytes()
            mask = read_image(path, cv2.IMREAD_GRAYSCALE)
            if mask.shape != self.frame.shape[:2]:
                raise ValueError("Seed mask dimensions must match the reference frame")
            masks["near"] = np.where(mask > 0, 255, 0).astype(np.uint8)
            seed = {"path": str(path), "sha256": hashlib.sha256(original).hexdigest(),
                    "note": "Unclassified seed copied into near; not an automatic depth classification."}
            (self.directory / "seed_original.png").write_bytes(png_bytes(mask))
        self._commit(masks, LAYER_IDS, seed)

    def manifest(self):
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def current_files(self, manifest):
        # The manifest points to a complete immutable save, never a partial update.
        revision = manifest["revision"]
        if not isinstance(revision, str) or not revision.isalnum():
            raise ValueError("Invalid saved revision")
        directory = self.directory / "history" / revision
        return {key: (directory / f"{key}.png").read_bytes() for key in LAYER_IDS}

    def load(self):
        with self.lock:
            manifest = self.manifest()
            files = self.current_files(manifest)
            return {"ok": True, "revision": manifest["revision"],
                    "included": manifest["included"],
                    "layers": {key: data_url(raw) for key, raw in files.items()}}

    def save(self, payload):
        with self.lock:
            previous = self.manifest()
            if payload.get("base_revision") != previous["revision"]:
                raise ValueError("A newer save exists. Reload layers before saving from this window.")
            layers = payload.get("layers")
            if not isinstance(layers, dict) or set(layers) != set(LAYER_IDS):
                raise ValueError("Expected exactly near, far and reflection layers")
            included = payload.get("included")
            if (not isinstance(included, list) or
                    any(not isinstance(key, str) or key not in LAYER_IDS for key in included) or
                    len(included) != len(set(included))):
                raise ValueError("Invalid combined-mask layer selection")
            masks = {}
            for key, value in layers.items():
                if not isinstance(value, str) or not value.startswith("data:image/png;base64,"):
                    raise ValueError(f"Expected a PNG data URL for {key}")
                raw = base64.b64decode(value.split(",", 1)[1], validate=True)
                rgba = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
                if (rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4 or
                        rgba.dtype != np.uint8 or rgba.shape[:2] != self.frame.shape[:2]):
                    raise ValueError(f"{key}: expected an RGBA canvas matching the reference frame")
                masks[key] = np.where(rgba[:, :, 3] > 0, 255, 0).astype(np.uint8)
            return self._commit(masks, included, previous.get("seed"))

    def _commit(self, masks, included, seed):
        combined = np.zeros(self.frame.shape[:2], np.uint8)
        overlay = self.frame.copy()
        for layer in LAYERS:
            key = layer["id"]
            if key in included:
                combined |= masks[key]
            color = np.full_like(self.frame, layer["color"][::-1])
            selected = masks[key] > 0
            overlay[selected] = cv2.addWeighted(overlay, 0.55, color, 0.45, 0)[selected]
        files = {f"{key}.png": png_bytes(mask) for key, mask in masks.items()}
        files["combined_mask.png"] = png_bytes(combined)
        files["overlay.png"] = png_bytes(overlay)
        revision = uuid.uuid4().hex
        snapshot = self.directory / "history" / revision
        snapshot.mkdir(parents=True)
        manifest = {
            "version": 1, "revision": revision,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "frame": str(self.frame_path), "frame_sha256": self.frame_sha256,
            "width": self.frame.shape[1], "height": self.frame.shape[0],
            "layers": LAYERS, "included": included, "seed": seed,
            "semantics": "Independent binary masks; overlaps allowed. White = selected.",
            "combined_mask": "combined_mask.png",
            "snapshot": str(snapshot),
        }
        manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
        for name, raw in files.items():
            (snapshot / name).write_bytes(raw)
        (snapshot / "layers.json").write_bytes(manifest_bytes)
        # Stable export paths are convenient for downstream scripts; history is authoritative.
        for name, raw in {**files, "layers.json": manifest_bytes}.items():
            temp = self.directory / (name + ".tmp")
            temp.write_bytes(raw)
            temp.replace(self.directory / name)
        return {"ok": True, "revision": revision,
                "mask": str(self.directory / "combined_mask.png"),
                "overlay": str(self.directory / "overlay.png"),
                "layers": {key: str(self.directory / f"{key}.png") for key in LAYER_IDS},
                "snapshot": str(snapshot)}
