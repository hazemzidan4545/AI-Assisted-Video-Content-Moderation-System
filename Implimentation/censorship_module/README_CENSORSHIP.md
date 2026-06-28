# AI-Assisted Video Content Moderation System - Censorship Module

This module contains the standalone video censorship pipeline used by the Streamlit demo and CLI entrypoint. It loads the trained visual models, predicts unsafe temporal segments, renders censorship, finalizes the output MP4, and writes JSON audit logs.

## Models

The default pipeline uses local project checkpoints:

- `checkpoints/binary_temporal_roi_best.pt`
- `checkpoints/convnext_sexynude_hardneg_best.pt`
- `yolov8n.pt`

The final demo path is temporal-only, ROI-aware, and full-frame during detected unsafe segments.

## How It Works

1. Read video metadata with OpenCV.
2. Split the video into overlapping 4-second windows with 2-second stride.
3. Sample 16 frames per window.
4. Use YOLOv8n to crop the main person ROI when available.
5. Run the binary ROI temporal model to estimate unsafe probability.
6. Smooth window predictions and merge unsafe windows into continuous segments.
7. Apply pixelation or blur during unsafe segments.
8. Use FFmpeg to finalize a browser-compatible H.264 MP4 and preserve/remux audio when available.
9. Save window, segment, frame-box, and summary JSON files.

## Run From The Implementation Directory

```bash
cd Implimentation

python -m censorship_module.run_censorship \
  --input path/to/input.mp4 \
  --output outputs/censored/output_censored.mp4 \
  --device cuda \
  --mode pixelate \
  --fused-policy temporal_only \
  --censor-region-mode full_frame
```

For CPU inference:

```bash
python -m censorship_module.run_censorship \
  --input path/to/input.mp4 \
  --output outputs/censored/output_censored.mp4 \
  --device cpu \
  --mode pixelate \
  --fused-policy temporal_only \
  --censor-region-mode full_frame
```

## Run The Web Demo

```bash
cd Implimentation
streamlit run app.py
```

The web demo supports upload, pixelation or blur, threshold control, local inference, video preview, MP4 download, and JSON summary download.

## Output Files

For an input named `sample.mp4`, the module writes:

- `outputs/censored/sample_censored.mp4`
- `outputs/censored/sample_windows.json`
- `outputs/censored/sample_segments.json`
- `outputs/censored/sample_frame_boxes.json`
- `outputs/censored/sample_summary.json`

If a custom `--output` path is supplied, the JSON files are written beside that output video.

## Core Defaults

```python
SEQ_LEN = 16
WINDOW_SECONDS = 4.0
WINDOW_STRIDE_SECONDS = 2.0
SMOOTHING_WINDOW = 3
MERGE_GAP_SECONDS = 0.5
TEMPORAL_UNSAFE_THRESHOLD = 0.50
FUSED_UNSAFE_POLICY = "temporal_only"
CENSOR_REGION_MODE = "full_frame"
ROI_MODE = "auto"
ROI_CONF = 0.25
PRESERVE_AUDIO = True
```

## Censorship Modes

- `pixelate`: applies block pixelation to frames inside unsafe segments.
- `blur`: applies Gaussian blur to frames inside unsafe segments.

## ROI Modes

- `auto`: use YOLOv8n person ROI extraction when available and fall back gracefully.
- `yolo`: require local YOLO ROI extraction.
- `none`: use full-frame temporal sampling.

## Audit Logs

The summary JSON records:

- temporal threshold
- unsafe window count
- merged unsafe segments
- ROI availability and fallback statistics
- censorship mode
- audio-remux status
- browser-compatible MP4 finalization status
- model checkpoint paths

## Smoke Test

Check model loading:

```bash
cd Implimentation
python -m censorship_module.smoke_test --device cpu
```

Run a short video end-to-end:

```bash
python -m censorship_module.smoke_test \
  --input path/to/test_video.mp4 \
  --output outputs/censored/test_censored.mp4 \
  --device cuda
```
