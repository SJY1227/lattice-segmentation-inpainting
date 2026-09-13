import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a video into numerically named JPEG frames for SAM2."
    )
    parser.add_argument("--video", default="video.mp4", help="Input video path")
    parser.add_argument(
        "--frames-dir", default="work/frames", help="Output frame directory"
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum frames to save")
    parser.add_argument("--stride", type=int, default=1, help="Save every Nth frame")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite an existing frame folder"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)
    out_dir = Path(args.frames_dir)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    existing = sorted(out_dir.glob("*.jpg"))
    if existing and not args.overwrite:
        print(f"Using existing frames in {out_dir} ({len(existing)} jpg files).")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in out_dir.glob("*.jpg"):
            path.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video: {total} frames, {fps:.3f} fps, {width}x{height}")

    saved = 0
    idx = 0
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride == 0:
            out_path = out_dir / f"{saved:05d}.jpg"
            cv2.imwrite(str(out_path), frame, encode_params)
            saved += 1
            if args.limit is not None and saved >= args.limit:
                break
        idx += 1

    cap.release()
    print(f"saved {saved} frames to {out_dir}")


if __name__ == "__main__":
    main()
