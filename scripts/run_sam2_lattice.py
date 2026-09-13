import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor


def parse_thresholds(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def threshold_tag(value):
    sign = "p" if value >= 0 else "m"
    return f"{sign}{abs(value):.2f}".replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAM2 video segmentation for the lattice with threshold sweeps."
    )
    parser.add_argument("--frames-dir", default="work/frames")
    parser.add_argument(
        "--prompt-json", default="work/prompts/lattice_points_frame0.json"
    )
    parser.add_argument(
        "--init-mask",
        default=None,
        help="Binary mask prompt PNG. If provided, this is used instead of point prompts.",
    )
    parser.add_argument("--init-frame-idx", type=int, default=0)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument(
        "--checkpoint", default="checkpoints/sam2.1_hiera_large.pt"
    )
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--output-dir", default="work/results/lattice_sweep")
    parser.add_argument(
        "--thresholds", default="-1.0,-0.5,0.0,0.5,1.0,1.5"
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=56.0)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-masks", action="store_true")
    parser.add_argument("--offload-video-to-cpu", action="store_true", default=True)
    parser.add_argument("--offload-state-to-cpu", action="store_true", default=False)
    parser.add_argument("--lazy-frames", action="store_true", help="Read JPEGs on demand with a two-frame CPU cache")
    parser.add_argument("--disable-multimask-tracking", action="store_true", help="Use the single-mask decoder output during propagation")
    parser.add_argument("--save-raw-masks", action="store_true", help="Also save thresholded masks before morphological cleanup")
    parser.add_argument("--memory-stride", type=int, default=None, help="Override temporal memory sampling stride (must be positive)")
    parser.add_argument(
        "--roi-mask",
        default=None,
        help="Optional mask that defines the allowed lattice region after dilation.",
    )
    parser.add_argument(
        "--roi-dilate",
        type=int,
        default=36,
        help="Dilation radius in pixels for roi-mask. Use 0 to disable dilation.",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=80,
        help="Remove connected mask components smaller than this area in pixels.",
    )
    parser.add_argument(
        "--morph-open",
        type=int,
        default=0,
        help="Opening kernel size. Use 0 to disable.",
    )
    parser.add_argument(
        "--morph-close",
        type=int,
        default=3,
        help="Closing kernel size. Use 0 to disable.",
    )
    args = parser.parse_args()
    if args.memory_stride is not None and args.memory_stride < 1:
        parser.error("--memory-stride must be positive")
    return args


def sorted_frame_paths(frames_dir):
    paths = sorted(
        Path(frames_dir).glob("*.jpg"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    if not paths:
        raise FileNotFoundError(f"No .jpg frames found in {frames_dir}")
    return paths


def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        prompt = json.load(f)
    points = np.array([[p["x"], p["y"]] for p in prompt["points"]], dtype=np.float32)
    labels = np.array([p["label"] for p in prompt["points"]], dtype=np.int32)
    frame_idx = int(prompt.get("frame_idx", 0))
    object_id = int(prompt.get("object_id", 1))
    return frame_idx, object_id, points, labels


def load_init_mask(path, expected_size):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    width, height = expected_size
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def build_roi(mask, radius):
    if mask is None:
        return None
    roi = mask.astype(np.uint8)
    if radius > 0:
        size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        roi = cv2.dilate(roi, kernel, iterations=1)
    return roi > 0


def remove_small_components(mask, min_area):
    if min_area <= 0:
        return mask
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if labels_count <= 1:
        return mask
    cleaned = np.zeros_like(mask, dtype=bool)
    for label in range(1, labels_count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = True
    return cleaned


def postprocess_mask(mask, roi, min_area, open_size, close_size):
    proc = mask.astype(np.uint8)
    if roi is not None:
        proc = np.logical_and(proc > 0, roi).astype(np.uint8)
    if open_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_size, open_size)
        )
        proc = cv2.morphologyEx(proc, cv2.MORPH_OPEN, kernel)
    if close_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size)
        )
        proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel)
    return remove_small_components(proc > 0, min_area)


def overlay_mask(frame, mask, alpha):
    color = np.zeros_like(frame)
    color[:, :, 1] = 255
    color[:, :, 2] = 60
    blended = cv2.addWeighted(frame, 1.0, color, alpha, 0.0)
    out = frame.copy()
    out[mask] = blended[mask]
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, (0, 255, 255), 1, lineType=cv2.LINE_AA)
    return out


def make_writer(path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def main():
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    frame_paths = sorted_frame_paths(args.frames_dir)
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dirs = {}
    raw_mask_dirs = {}
    overlay_dirs = {}
    writers = {}
    for th in thresholds:
        tag = threshold_tag(th)
        if not args.no_masks:
            mask_dirs[th] = out_dir / f"masks_t{tag}"
            mask_dirs[th].mkdir(parents=True, exist_ok=True)
        if args.save_raw_masks:
            raw_mask_dirs[th] = out_dir / f"raw_masks_t{tag}"
            raw_mask_dirs[th].mkdir(parents=True, exist_ok=True)
        overlay_dirs[th] = out_dir / f"overlays_t{tag}"
        overlay_dirs[th].mkdir(parents=True, exist_ok=True)
        if not args.no_video:
            writers[th] = make_writer(
                out_dir / f"overlay_t{tag}.mp4", args.fps, (width, height)
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"device: {device}")

    overrides = []
    if args.disable_multimask_tracking:
        overrides.append("++model.multimask_output_for_tracking=false")
    if args.memory_stride is not None:
        overrides.append(f"++model.memory_temporal_stride_for_eval={args.memory_stride}")
    predictor = build_sam2_video_predictor(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device=device,
        apply_postprocessing=False,
        hydra_overrides_extra=overrides,
    )
    effective = {
        "multimask_output_for_tracking": predictor.multimask_output_for_tracking,
        "memory_temporal_stride_for_eval": predictor.memory_temporal_stride_for_eval,
        "num_maskmem": predictor.num_maskmem,
        "thresholds": thresholds,
        "min_component_area": args.min_component_area,
        "morph_open": args.morph_open,
        "morph_close": args.morph_close,
        "roi_mask": args.roi_mask,
        "apply_postprocessing": False,
    }
    (out_dir / "effective_settings.json").write_text(json.dumps(effective, indent=2), encoding="utf-8")
    print(f"effective settings: {effective}", flush=True)

    roi_source = args.roi_mask
    roi = None
    if roi_source is not None:
        roi = build_roi(load_init_mask(roi_source, (width, height)), args.roi_dilate)
        roi_preview = (roi.astype(np.uint8) * 255)
        cv2.imwrite(str(out_dir / "postprocess_roi.png"), roi_preview)

    if args.init_mask is not None:
        frame_idx = args.init_frame_idx
        object_id = args.object_id
        init_mask = load_init_mask(args.init_mask, (width, height))
        points = labels = None
        print(
            f"mask prompt frame: {frame_idx}, object id: {object_id}, "
            f"area: {int(init_mask.sum())} px"
        )
    else:
        frame_idx, object_id, points, labels = load_prompt(args.prompt_json)
        init_mask = None
        print(f"point prompt frame: {frame_idx}, object id: {object_id}, points: {len(points)}")

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )

    metrics_path = out_dir / "mask_area_metrics.csv"
    with torch.inference_mode(), autocast_ctx:
        if args.lazy_frames:
            from sam2_lazy_frames import init_lazy_state
            state = init_lazy_state(predictor, frame_paths, args.frames_dir, args.offload_state_to_cpu)
        else:
            state = predictor.init_state(
                video_path=args.frames_dir,
                offload_video_to_cpu=args.offload_video_to_cpu,
                offload_state_to_cpu=args.offload_state_to_cpu,
                async_loading_frames=False,
            )
        if init_mask is not None:
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=object_id,
                mask=init_mask,
            )
        else:
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
                clear_old_points=True,
            )

        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "threshold", "area_px", "area_ratio"])
            iterator = predictor.propagate_in_video(
                state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=len(frame_paths) - frame_idx,
                reverse=False,
            )
            for out_frame_idx, out_obj_ids, out_mask_logits in tqdm(
                iterator, total=len(frame_paths) - frame_idx, desc="propagating"
            ):
                if out_frame_idx >= len(frame_paths):
                    break
                frame = cv2.imread(str(frame_paths[out_frame_idx]))
                if frame is None:
                    raise RuntimeError(f"Could not read frame {frame_paths[out_frame_idx]}")

                obj_pos = list(out_obj_ids).index(object_id)
                logits = out_mask_logits[obj_pos].float().cpu().numpy().squeeze()

                for th in thresholds:
                    raw_mask = logits > th
                    mask = postprocess_mask(
                        raw_mask,
                        roi=roi,
                        min_area=args.min_component_area,
                        open_size=args.morph_open,
                        close_size=args.morph_close,
                    )
                    area_px = int(mask.sum())
                    writer.writerow([out_frame_idx, th, area_px, area_px / mask.size])

                    tag = threshold_tag(th)
                    if th in raw_mask_dirs:
                        if not cv2.imwrite(str(raw_mask_dirs[th] / f"{out_frame_idx:05d}.png"), raw_mask.astype(np.uint8) * 255):
                            raise OSError("Could not save raw mask")
                    if not args.no_masks:
                        cv2.imwrite(
                            str(mask_dirs[th] / f"{out_frame_idx:05d}.png"),
                            (mask.astype(np.uint8) * 255),
                        )

                    overlay = overlay_mask(frame, mask, args.alpha)
                    cv2.imwrite(str(overlay_dirs[th] / f"{out_frame_idx:05d}.jpg"), overlay)
                    if th in writers:
                        writers[th].write(overlay)

    for writer in writers.values():
        writer.release()

    print(f"saved results to {out_dir}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
