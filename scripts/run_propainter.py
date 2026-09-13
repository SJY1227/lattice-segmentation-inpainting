"""Validate a frame/mask handoff and run the preserved ProPainter entrypoint."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_files(folder):
    if not folder.is_dir():
        raise ValueError(f"Not a frame/mask directory: {folder}")
    files = sorted(folder.iterdir())
    if not files or any(not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES for p in files):
        raise ValueError(f"Directory must contain only ordered image files: {folder}")
    return files


def build_command(args):
    frames = args.frames_dir.resolve()
    masks = args.masks_dir.resolve()
    output = args.output_dir.resolve()
    frame_files, mask_files = image_files(frames), image_files(masks)
    if len(frame_files) < 2 or len(frame_files) != len(mask_files):
        raise ValueError("At least two frames and exactly one mask per frame are required.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty.")
    if min(args.width, args.height) < 8 or args.width % 8 or args.height % 8:
        raise ValueError("Width and height must be positive multiples of eight.")
    if args.fps <= 0 or args.mask_dilation < 0 or args.raft_iter < 0:
        raise ValueError("FPS must be positive; dilation and RAFT iterations cannot be negative.")
    if args.neighbor_length < 2 or args.neighbor_length % 2:
        raise ValueError("Neighbor length must be an even number of at least two.")
    if args.ref_stride < 1 or args.subvideo_length < 2:
        raise ValueError("Reference stride must be positive; subvideo length must be at least two.")
    command = [args.python, str(ROOT / "vendor/propainter/inference_propainter_offload.py"),
               "--video", str(frames), "--mask", str(masks), "--output", str(output)]
    for name in ("width", "height", "mask_dilation", "raft_iter", "neighbor_length", "ref_stride", "subvideo_length"):
        command.extend(["--" + name, str(getattr(args, name))])
    command.extend(["--save_fps", str(args.fps)])
    for name in ("fp16", "cpu_offload", "save_frames"):
        if getattr(args, name):
            command.append("--" + name)
    return command, output, len(frame_files)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--fps", type=float, required=True)
    for name, default in (("width", 512), ("height", 512), ("mask-dilation", 2), ("raft-iter", 20),
                          ("neighbor-length", 10), ("ref-stride", 10), ("subvideo-length", 50)):
        parser.add_argument("--" + name, type=int, default=default)
    for name in ("fp16", "cpu-offload", "save-frames", "dry-run"):
        parser.add_argument("--" + name, action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    command, output, count = build_command(args)
    record = {"frame_count": count, "command": command, "cwd": str(ROOT / "vendor/propainter")}
    print(json.dumps(record, indent=2))
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    subprocess.run(command, cwd=ROOT / "vendor/propainter", check=True)


if __name__ == "__main__":
    main()
