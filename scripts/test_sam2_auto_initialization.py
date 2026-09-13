import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {
    "default32": {"points_per_side": 32},
    "dense64": {"points_per_side": 64},
    "crop64": {"points_per_side": 64, "crop_n_layers": 1, "crop_n_points_downscale_factor": 2},
    "relaxed_crop64": {
        "points_per_side": 64, "crop_n_layers": 1, "crop_n_points_downscale_factor": 2,
        "pred_iou_thresh": 0.65, "stability_score_thresh": 0.85, "mask_threshold": -0.5,
    },
    "unfiltered32": {
        "points_per_side": 32, "pred_iou_thresh": 0.0, "stability_score_thresh": 0.0,
        "mask_threshold": -0.5, "box_nms_thresh": 0.95,
    },
    "m2m_crop64": {
        "points_per_side": 64, "crop_n_layers": 1, "crop_n_points_downscale_factor": 2,
        "pred_iou_thresh": 0.65, "stability_score_thresh": 0.85,
        "mask_threshold": -0.5, "use_m2m": True,
    },
}


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(path)


def overlay(frame, mask, alpha=0.35):
    result = frame.copy()
    result[mask] = np.rint((1 - alpha) * frame[mask] + alpha * np.array([40, 255, 50])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 255), 1)
    return result


def panel(frame, title, note="", width=480):
    height = round(frame.shape[0] * width / frame.shape[1])
    result = np.full((height + 64, width, 3), 22, np.uint8)
    result[64:] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    cv2.putText(result, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(result, note, (10, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
    return result


def image_cues(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    cutoff, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    b, g, r = [channel.astype(np.float32) for channel in cv2.split(frame)]
    red = (r > 1.25 * g) & (r > 1.25 * b) & (r > 70)
    bright = (binary > 0) & ~red
    return bright, red, float(cutoff)


def score_candidate(mask, bright, red):
    area = int(mask.sum())
    overlap = int((mask & bright).sum())
    precision = overlap / max(area, 1)
    recall = overlap / max(int(bright.sum()), 1)
    cue_f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    red_fraction = float((mask & red).sum() / max(area, 1))
    return {
        "area_ratio": area / mask.size,
        "bright_precision_proxy": precision,
        "bright_recall_proxy": recall,
        "bright_f1_proxy": cue_f1,
        "red_fraction": red_fraction,
        "selection_score_proxy": cue_f1 - 2 * red_fraction,
    }


def summarize_config(name, annotations, frame, bright, red, output, elapsed, parameters):
    directory = output / name
    masks_dir = directory / "candidates"
    masks_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, item in enumerate(annotations):
        mask = item["segmentation"]
        filename = f"{index:04d}.png"
        write_image(masks_dir / filename, mask.astype(np.uint8) * 255)
        records.append({
            "id": index, "mask_file": f"candidates/{filename}",
            "area": int(item["area"]), "bbox": item["bbox"],
            "predicted_iou": float(item["predicted_iou"]),
            "stability_score": float(item["stability_score"]),
            "point_coords": item["point_coords"], "crop_box": item["crop_box"],
            **score_candidate(mask, bright, red),
        })
    ranked = sorted(records, key=lambda record: record["selection_score_proxy"], reverse=True)
    best = annotations[ranked[0]["id"]]["segmentation"] if ranked else np.zeros(frame.shape[:2], bool)

    # This explicit image-specific heuristic is not a semantic lattice classifier or ground truth.
    selected_ids = [record["id"] for record in records
                    if 0.0005 <= record["area_ratio"] <= 0.85
                    and record["bright_precision_proxy"] >= 0.60
                    and record["red_fraction"] <= 0.02]
    union = np.zeros(frame.shape[:2], bool)
    for index in selected_ids:
        union |= annotations[index]["segmentation"]
    for filename, mask in [("best_candidate_mask.png", best), ("auto_union_mask.png", union)]:
        write_image(directory / filename, mask.astype(np.uint8) * 255)
        write_image(directory / filename.replace("_mask.png", "_overlay.png"), overlay(frame, mask))

    tiles = []
    for record in ranked[:24]:
        picture = overlay(frame, annotations[record["id"]]["segmentation"])
        tiles.append(panel(picture, f"Candidate {record['id']:04d}",
                           f"IoU(pred) {record['predicted_iou']:.2f} | cue {record['selection_score_proxy']:.2f}", 320))
    while tiles and len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 22))
    if tiles:
        write_image(directory / "candidate_contact_sheet.jpg",
                    np.vstack([np.hstack(tiles[start:start + 4]) for start in range(0, len(tiles), 4)]))

    result = {
        "configuration": name, "parameters": parameters, "elapsed_seconds": elapsed,
        "candidate_count": len(records), "best_candidate_id": ranked[0]["id"] if ranked else None,
        "auto_union_selected_ids": selected_ids,
        "best_candidate": score_candidate(best, bright, red),
        "auto_union": score_candidate(union, bright, red),
        "candidates": records,
        "note": "Scores measure agreement with brightness/color cues, NOT true segmentation accuracy.",
    }
    (directory / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    row = np.hstack([
        panel(frame, name, f"{len(records)} candidates | frame 0"),
        panel(overlay(frame, best), "Automatic best candidate", f"area {best.mean():.1%}"),
        panel(overlay(frame, union), "Automatic candidate union", f"{len(selected_ids)} selected | area {union.mean():.1%}"),
    ])
    write_image(directory / "comparison.jpg", row)
    compact = {key: value for key, value in result.items() if key != "candidates"}
    print(json.dumps(compact, indent=2), flush=True)
    return compact, row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True)
    parser.add_argument("--output", default="work/auto_init")
    parser.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--points-per-batch", type=int, default=32)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in args.configs:
        if (output / name).exists():
            raise FileExistsError(output / name)
    frame_path = (ROOT / args.frame).resolve()
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError(frame_path)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bright, red, cutoff = image_cues(frame)
    write_image(output / "brightness_cue_mask.png", bright.astype(np.uint8) * 255)
    write_image(output / "brightness_cue_overlay.png", overlay(frame, bright))
    write_image(output / "red_exclusion_cue.png", red.astype(np.uint8) * 255)
    checkpoint = ROOT / args.checkpoint
    config_file = "configs/sam2.1/sam2.1_hiera_l.yaml"
    metadata = {
        "frame": str(frame_path), "frame_sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
        "frame_size": list(frame.shape[1::-1]), "checkpoint": str(checkpoint), "model_config": config_file,
        "manual_masks_read_or_used": False, "video_propagation_run": False,
        "brightness_otsu_cutoff": cutoff,
        "selection_rule": {
            "best": "maximize brightness-cue F1 minus twice red-pixel fraction",
            "union": "union candidates with bright precision >= 0.60, red fraction <= 0.02, area ratio in [0.0005, 0.85]",
            "red_cue": "R > 1.25*G and R > 1.25*B and R > 70",
            "limitation": "Appearance heuristic for this lighting/object; acrylic/glue may also pass. Not validated for deployment.",
        },
        "experiments": [],
    }
    metadata_path = output / "summary.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = build_sam2(config_file, str(checkpoint), device=device, apply_postprocessing=False)
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for name in args.configs:
            parameters = {
                "points_per_side": 32, "points_per_batch": args.points_per_batch,
                "pred_iou_thresh": 0.8, "stability_score_thresh": 0.95,
                "stability_score_offset": 1.0, "mask_threshold": 0.0,
                "box_nms_thresh": 0.7, "crop_n_layers": 0, "crop_nms_thresh": 0.7,
                "crop_overlap_ratio": 512 / 1500, "crop_n_points_downscale_factor": 1,
                "min_mask_region_area": 0, "output_mode": "binary_mask", "use_m2m": False,
                **CONFIGS[name],
            }
            print(f"Starting {name}: {json.dumps(parameters)}", flush=True)
            start = time.monotonic()
            generator = SAM2AutomaticMaskGenerator(model, **parameters)
            annotations = generator.generate(rgb)
            elapsed = time.monotonic() - start
            result, row = summarize_config(name, annotations, frame, bright, red, output, elapsed, parameters)
            rows.append(row)
            metadata["experiments"].append(result)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            write_image(output / "comparison_all_settings.jpg", np.vstack(rows))
            del annotations, generator
            if device == "cuda":
                torch.cuda.empty_cache()
    fields = ["configuration", "candidate_count", "best_candidate_id", "elapsed_seconds"]
    with (output / "experiments.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: result[key] for key in fields} for result in metadata["experiments"])
    print(f"All experiments completed: {output}", flush=True)


if __name__ == "__main__":
    main()
