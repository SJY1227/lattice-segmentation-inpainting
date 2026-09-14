"""Reject capture media, weights, private paths and common secret formats."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
MASK_ARCHIVES = ("5cell_mask_v3", "5cell_mask_v4_reflection")
MASKS = {f"masks/{archive}/{name}" for archive in MASK_ARCHIVES
         for name in (f"{archive}.png", "near.png", "far.png", "reflection.png")}
BANNED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif", ".jpg", ".jpeg", ".png",
                   ".bmp", ".tif", ".tiff", ".webp", ".pdf", ".pt", ".pth", ".ckpt", ".onnx",
                   ".safetensors", ".npy", ".npz", ".zip", ".tar", ".gz", ".pyd", ".so", ".dll"}
PATTERNS = [r"[A-Z]:[\\/]+(?:Users|SynologyDrive)[\\/]", r"gh[pousr]_[A-Za-z0-9]{30,}",
            r"github_pat_[A-Za-z0-9_]{30,}", r"AKIA[0-9A-Z]{16}",
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"]


def main():
    if (ROOT / ".git").exists():
        paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        paths = [p for p in paths if p]
    else:
        paths = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()
                 and "__pycache__" not in p.parts]
    if not paths:
        raise SystemExit("Nothing staged/tracked; stage the intended release before auditing.")
    failures = []
    for relative in paths:
        path = ROOT / relative
        if relative in MASKS:
            with Image.open(path) as image:
                if image.mode != "L" or image.size != (640, 640) or not set(np.unique(np.array(image))) <= {0, 255}:
                    failures.append(relative + ": expected binary 640x640 annotation")
            continue
        if path.suffix.lower() in BANNED_SUFFIXES or ".private." in path.name or path.name.startswith(".env"):
            failures.append(relative + ": forbidden artifact")
            continue
        if path.stat().st_size > 2_000_000:
            failures.append(relative + ": unexpectedly large source file")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            failures.append(relative + ": unexpected binary")
            continue
        if any(re.search(pattern, content, re.IGNORECASE) for pattern in PATTERNS):
            failures.append(relative + ": possible secret or private local path")
    missing = MASKS - set(paths)
    failures.extend(sorted(missing))
    for archive in MASK_ARCHIVES:
        directory = ROOT / "masks" / archive
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        mask = directory / f"{archive}.png"
        if manifest["mask"] != mask.name or hashlib.sha256(mask.read_bytes()).hexdigest() != manifest["mask_sha256"]:
            failures.append(archive + ": archived mask digest or filename mismatch")
        with Image.open(mask) as image:
            combined = np.array(image) > 0
        union = np.zeros_like(combined)
        for layer in manifest["layers"]:
            if layer["id"] not in ("near", "far", "reflection") or layer["file"] != layer["id"] + ".png":
                failures.append(archive + ": invalid layer identifier or filename")
                continue
            with Image.open(directory / layer["file"]) as image:
                selected = np.array(image) > 0
            if int(selected.sum()) != layer["selected_pixels"]:
                failures.append(archive + ": layer area differs from manifest")
            if layer["id"] in manifest["included_layers"]:
                union |= selected
        if not np.array_equal(combined, union) or int(combined.sum()) != manifest["selected_pixels"]:
            failures.append(archive + ": combined mask differs from layer union or area")
    if failures:
        raise SystemExit("\n".join(failures))
    print(json.dumps({"checked_files": len(paths), "total_bytes": sum((ROOT / p).stat().st_size for p in paths),
                      "allowed_binary_files": sorted(MASKS), "status": "passed"}, indent=2))


if __name__ == "__main__":
    main()
