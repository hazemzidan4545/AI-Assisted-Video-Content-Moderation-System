# CensorAI: Binary ROI Temporal Video Censorship

Local Streamlit demo and CLI pipeline for AI-assisted video censorship. The final supported system uses a visual binary ROI temporal model to detect unsafe video time segments, then applies full-frame pixelation or blur during detected unsafe intervals.

## Final Claim Boundary

This repository supports the final visual temporal censorship pipeline only:

- Final temporal model: `Implimentation/checkpoints/binary_temporal_roi_best.pt`
- Final image backbone: `Implimentation/checkpoints/convnext_sexynude_hardneg_best.pt`
- Local ROI helper weights: `Implimentation/yolov8n.pt`
- Default inference policy: temporal-only full-frame censorship

It does **not** claim final profanity detection, violence detection, accepted audio fusion, segment-level audio muting, or precise nude-region segmentation. Audio is preserved/remuxed into the output video when possible; it is not used by the final accepted classifier.

## Repository Layout

```text
Implimentation/
  app.py                         Streamlit local demo
  censorship_module/             Inference, rendering, model loading, smoke test
  pipelines/roi_detector.py      Local YOLO person ROI helper
  Models/temporal_model.py       ConvNeXt temporal model definitions
  checkpoints/                   Final model files tracked with Git LFS
  eval_logs/                     Experiment summaries and metrics
reports/
  thesis_evidence_bundle/        Evidence files used for thesis claims
  thesis_chapters_4_7_generated.md
Hazem Zidan.jpeg                 Preview image used by the Streamlit UI
```

## Run the Streamlit Demo

```bash
cd Implimentation
streamlit run app.py
```

The app lets the user upload a video, choose pixelation or blur, adjust the unsafe threshold, run the binary ROI temporal detector, preview the censored output, and download the output MP4 plus summary JSON.

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

Use `--device cpu` if CUDA is unavailable.

## Model Files

The final model files are larger than GitHub's normal 100 MB blob limit, so they are tracked with Git LFS:

- `Implimentation/checkpoints/binary_temporal_roi_best.pt`
- `Implimentation/checkpoints/convnext_sexynude_hardneg_best.pt`
- `Implimentation/yolov8n.pt`

After cloning, install Git LFS and pull model content:

```bash
git lfs install
git lfs pull
```

## Key Results

- Binary ROI temporal model macro-F1: `0.7506`
- Binary ROI temporal unsafe recall: `0.8586`
- ROI temporal improvement over full-frame temporal baseline: `+0.0650` macro-F1
- Hard-negative tuning reduced unsafe predictions on 813 normal hard negatives from `66.91%` to `3.81%`

Supporting metrics are in `Implimentation/eval_logs/` and `reports/thesis_evidence_bundle/`.

## Notes

The original datasets, generated videos, uploads, virtual environments, and duplicate exploratory checkpoints are intentionally excluded from git. This keeps the repository cloneable while preserving the final runnable code, selected final models, and thesis evidence.
