# Lattice Segmentation and Inpainting

SAM2 mask-prompt propagation and ProPainter inpainting for a lattice-based
visuotactile sensor. This is an inference-source snapshot of a local research
workflow, not a new segmentation model or a depth-conditioned ProPainter model.

## Included

- Brush-based initial-mask editor with zoom, pan, Shift axis locking, undo/redo,
  and independent near/far/reflection annotation layers.
- SAM2.1 propagation from an initial mask, output-logit threshold sweeps,
  morphological cleanup, and optional bounded-memory JPEG loading.
- ProPainter inference with optional CPU tensor offloading and temporal controls.
- The explicitly archived `masks/5cell_mask_v3/5cell_mask_v3.png` initial mask.

Capture videos, frame images, reference photographs, overlays, predicted
per-frame masks, inpainting outputs, checkpoints, local paths, and old Git
history are intentionally NOT included. Only four binary annotation PNGs are
included, including the two currently empty annotation layers.

## Layout

```text
scripts/                  Local workflow tools and synthetic-data unit tests
masks/5cell_mask_v3/       Initial mask, independent layers, sanitized provenance
vendor/sam2/              SAM2 inference package and configurations
vendor/propainter/        ProPainter inference source, including CPU offloading
work/                     Your local inputs and outputs (ignored by Git)
UPSTREAM.json             Source revisions and local modification summary
```

## Environment

Python 3.12 and an NVIDIA CUDA GPU were used locally. SAM2 was used with
torch 2.11.0 / CUDA 12.8; ProPainter with torch 2.6.0 / CUDA 12.4 in a separate
environment. These are provenance notes, not a claim that every version pair
or a fresh unified environment has been tested.

Create a virtual environment and install a matching CUDA-enabled
`torch`/`torchvision` pair following [PyTorch](https://pytorch.org/get-started/locally/).
SAM2 requires torch >= 2.5.1 and torchvision >= 0.20.1.
Then, from the repository root:

```powershell
python -m pip install -r requirements.txt
$env:SAM2_BUILD_CUDA = "0"
python -m pip install --no-build-isolation -e vendor/sam2
```

Skipping the optional SAM2 CUDA extension is sufficient for this runner's
`apply_postprocessing=False` path. The Python/OpenCV cleanup still runs.
The vendored package omits upstream demo media and training tools.
Keep SAM2 and ProPainter in separate environments when reproducing the local
setup; `scripts/run_propainter.py --python <executable>` selects the latter.

## Model Weights

Weights are downloaded separately and ignored by Git. Download SAM2.1 Hiera
Large from the official Meta endpoint:

```powershell
New-Item -ItemType Directory -Force checkpoints | Out-Null
Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -OutFile checkpoints/sam2.1_hiera_large.pt
```

ProPainter automatically downloads `raft-things.pth`,
`recurrent_flow_completion.pth` and `ProPainter.pth` from its
[official release](https://github.com/sczhou/ProPainter/releases/tag/v0.1.0)
into `vendor/propainter/weights/` on the first inference run.

## 1. Prepare Frames

```powershell
python scripts/prepare_frames.py --video work/input.mp4 --frames-dir work/frames
```

Existing frame folders can be used without a video. SAM2 expects numeric JPEG
stems such as `00000.jpg`, `00001.jpg`, ... in chronological order. Make a
renamed copy when camera filenames are not numeric. Keep original captures
unchanged. Record the actual capture FPS for later video export.

## 2. Initial Mask

The archived mask is 640 x 640, binary grayscale: white selects the lattice,
black preserves the scene. Its SHA-256 and annotation revision are in
`masks/5cell_mask_v3/manifest.json`. Far and reflection layers are currently
empty, not automatically classified. The combined annotation is a single
SAM2 object in the command below.

Reuse it only when the lattice, camera pose, crop, and initial undeformed state
match. Resizing cannot fix an alignment mismatch. It is not a universal mask
for all five-cell or full-cell sensors.

To edit a working copy against your own first frame:

```powershell
python scripts/mask_painter_server.py --frame work/frames/00000.jpg --mask masks/5cell_mask_v3/5cell_mask_v3.png --layers-dir work/initial_mask --port 8770
```

Open `http://127.0.0.1:8770/`. Layered saves go to `work/initial_mask/`, including
`combined_mask.png` and revision history; the archived seed is not overwritten.
For automatic candidate generation before manual correction:

```powershell
python scripts/test_sam2_auto_initialization.py --frame work/frames/00000.jpg --output work/auto_init --configs unfiltered32 --checkpoint checkpoints/sam2.1_hiera_large.pt
```

Candidate ranking uses brightness/color heuristics. It is not a trained lattice
classifier, a depth classifier, or ground-truth evaluation.

## 3. SAM2 Propagation

```powershell
python scripts/run_sam2_lattice.py --frames-dir work/frames --init-mask masks/5cell_mask_v3/5cell_mask_v3.png --checkpoint checkpoints/sam2.1_hiera_large.pt --thresholds=-0.50,2.00 --output-dir work/sam2 --fps 56 --lazy-frames --offload-state-to-cpu
```

Replace `--fps 56` with the actual capture FPS, and `--init-mask` with your
edited `work/initial_mask/combined_mask.png` when needed.

- `masks_tm0p50/`: per-frame binary masks for `logit > -0.50`.
- `masks_tp2p00/`: per-frame binary masks for `logit > 2.00`.
- `overlay_tm0p50.mp4`, `overlay_tp2p00.mp4`: inspection previews.
- `effective_settings.json`: effective predictor and cleanup settings.

Thresholds are absolute output logits, not offsets or probabilities. Lower
values generally select more pixels; they do not change SAM2's memory update.
Defaults are `--min-component-area 80 --morph-open 0 --morph-close 3`.
`--save-raw-masks` retains masks before cleanup. Multimask tracking remains
enabled unless explicitly disabled; memory stride is not overridden by default.

## 4. ProPainter

```powershell
python scripts/run_propainter.py --frames-dir work/frames --masks-dir work/sam2/masks_tm0p50 --output-dir work/inpainting --fps 56 --width 512 --height 512 --mask-dilation 2 --raft-iter 20 --neighbor-length 10 --ref-stride 10 --subvideo-length 50 --fp16 --cpu-offload --save-frames
```

The wrapper checks input counts, requires a new/empty output directory, and
runs the vendored entrypoint from the correct working directory. It records
the command locally. `--dry-run` validates without loading models.
The native output is under `work/inpainting/<frame-folder-name>/`, including
`inpaint_out.mp4` and, with `--save-frames`, `frames/` PNGs.

SAM2 output and ProPainter input have the same white = remove convention.
Match every frame with one mask in sorted order. A single initial mask is a
SAM2 prompt, not a substitute for moving per-frame inpainting masks.
ProPainter resizes frames/masks to the processing size, then applies dilation.

| Option | Example | Meaning |
| --- | --- | --- |
| `--mask-dilation` | 2 | Spatial mask expansion at processing resolution |
| `--raft-iter` | 20 | Optical-flow refinement iterations, not a frame count |
| `--neighbor-length` | 10 | Local window: up to 5 before + center + 5 after |
| `--ref-stride` | 10 | Spacing of additional reference frames |
| `--subvideo-length` | 50 | Propagation chunk length; also affects reference budget |
| `--fp16` | flag | Half-precision inference |
| `--cpu-offload` | flag | Store full-sequence tensors on CPU between GPU operations |

Processing uses both past and future frames, so this is an offline workflow.
Subvideos overlap; `subvideo_length` is not a hard bound on all temporal
information. Image-propagation chunks are capped at 100 before padding.
Increasing temporal context can increase memory use and is not guaranteed to
improve deforming, translucent lattice scenes. No displacement/depth input is
used. `raft-iter=0` is retained for experimental zero-flow tests, not recommended
as a normal replacement for motion estimation.

## Verification

Synthetic-data checks do not need any capture media or model weights:

```powershell
python -m unittest discover -s scripts -p test_mask_painter_layers.py
python -m unittest discover -s scripts -p test_sam2_lazy_frames.py
python -m unittest discover -s scripts -p test_run_propainter.py
python scripts/audit_public_tree.py
```

The source snapshot preserves the previously used model logic. Packaging smoke
tests are not a new end-to-end quality benchmark. Generated inpainting can
invent or blur hidden surfaces; it is not a measurement of true object shape.

## Attribution and License

See [LICENSE.md](LICENSE.md) and [UPSTREAM.json](UPSTREAM.json).
[SAM2](https://github.com/facebookresearch/sam2) is from Meta;
[ProPainter](https://github.com/sczhou/ProPainter) is from S-Lab.
**ProPainter is subject to a non-commercial license.** Publishing this collection
does not make the entire workflow permissively licensed for commercial use.
