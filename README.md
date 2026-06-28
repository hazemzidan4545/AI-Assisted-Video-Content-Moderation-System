# AI-Assisted Video Content Moderation System

An end-to-end computer vision prototype for detecting unsafe visual content in videos and generating censored MP4 outputs with pixelation or blur. The system combines a 3-class ConvNeXt-Small image classifier with a binary ROI-aware temporal video classifier, then converts unsafe window predictions into continuous censorship segments.

## Demo

<video src="media/system-demo.mp4" controls width="100%" title="System demo"></video>

[Watch the system demo video](media/system-demo.mp4)

## Project Highlights

- Built an end-to-end computer vision prototype for 3-class unsafe visual content detection, applying blur or pixelation to generate censored MP4 outputs.
- Fine-tuned ConvNeXt-Small with hard-negative tuning, improving test macro-F1 from `0.8745` to `0.8915`.
- Developed a YOLOv8n ROI-aware temporal classifier using person-region cropping and 16-frame overlapping windows, achieving `0.7506` macro-F1 and `0.8586` unsafe recall.
- Outperformed the full-frame temporal baseline by `+0.0650` macro-F1 and converted window predictions into continuous censorship segments.

## System Overview

The deployed prototype follows a modular pipeline:

1. Upload a video through the Streamlit interface.
2. Sample overlapping temporal windows from the video.
3. Use YOLOv8n person-region cropping for ROI-aware temporal inference.
4. Classify each 16-frame window as safe or unsafe using the binary temporal model.
5. Smooth and merge unsafe windows into continuous censorship intervals.
6. Apply full-frame pixelation or blur only during detected unsafe intervals.
7. Export a browser-compatible MP4 and JSON audit logs.

## Repository Layout

```text
Implimentation/
  app.py                         Streamlit demo interface
  censorship_module/             Inference, rendering, model loading, and smoke tests
  pipelines/roi_detector.py      YOLOv8n person ROI extraction
  Models/temporal_model.py       ConvNeXt temporal model definitions
  checkpoints/                   Final model files tracked with Git LFS
  eval_logs/                     Experiment summaries and metrics
media/
  system-demo.mp4                README demo video
reports/
  thesis_evidence_bundle/        Thesis evidence files and metric summaries
```

## Run The Streamlit Demo

```bash
cd Implimentation
streamlit run app.py
```

The web app supports video upload, censorship style selection, unsafe-threshold adjustment, local inference, output preview, MP4 download, and JSON summary download.

## Run CLI Inference

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

Use `--device cpu` when CUDA is unavailable.

## Model Files

The final model files are stored with Git LFS:

- `Implimentation/checkpoints/binary_temporal_roi_best.pt`
- `Implimentation/checkpoints/convnext_sexynude_hardneg_best.pt`
- `Implimentation/yolov8n.pt`

After cloning:

```bash
git lfs install
git lfs pull
```

## Outputs

For each processed video, the system writes:

- Censored MP4 output
- Window-level prediction JSON
- Merged unsafe segment JSON
- Frame-level censorship log JSON
- Summary JSON with model, threshold, rendering, and audio-remux metadata

## Requirements

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

FFmpeg is required for browser-compatible MP4 finalization and audio remuxing.
