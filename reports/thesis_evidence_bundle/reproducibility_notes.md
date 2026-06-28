# Reproducibility Notes

- Python version: `3.10.12 (main, Mar  3 2026, 11:56:32) [GCC 11.4.0]`
- PyTorch version: `2.11.0+cu130`
- CUDA version if available: `13.0`
- GPU name: `NVIDIA GeForce RTX 4070 Laptop GPU`
- CPU: `x86_64`
- RAM: `NOT_FOUND`
- OS: `Linux-6.8.0-117-generic-x86_64-with-glibc2.35`
- Random seeds used: NOT_FOUND as a consistently logged final setting. Some exploratory sections mention seed 42 for Dataset 3 sample capping, but final binary ROI temporal summary does not log a random seed.
- Image size: `224x224`, source `Implimentation/censorship_module/convnext_frame_risk.py` and notebook transform summaries.
- Batch size: final binary ROI summary logs `1` with gradient accumulation `2` and effective batch size `2`.
- Optimizer: final binary ROI summary does not name optimizer; NOT_FOUND. Image backbone run used ConvNeXt training logs but optimizer not in run_summary.
- Learning rate: NOT_FOUND for final binary ROI summary.
- Weight decay: NOT_FOUND for final binary ROI summary.
- Epochs: final binary ROI best epoch `18`; image classifier baseline epochs `15`.
- Clip length: `16` frames.
- ROI confidence threshold: `0.25`.
- Selected classification thresholds: binary ROI threshold `0.5`; recall-oriented threshold `0.4`.
- Final checkpoint paths: image baseline `/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation/checkpoints/convnext_sexynude_best.pt`; hard-negative image `checkpoints/convnext_sexynude_hardneg_best.pt`; binary ROI temporal `/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation/checkpoints/binary_temporal_roi_best.pt`.
- Exact command to run inference:

```bash
cd "/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation"
python -m censorship_module.run_censorship --input path/to/input.mp4 --output outputs/censored/output_censored.mp4 --device cuda --mode pixelate --fused-policy temporal_only --censor-region-mode full_frame
```

- Exact command to run training if available: NOT_FOUND. Training was primarily notebook-based in `Implimentation/PORNMODULE.ipynb`; no single final CLI training command was found.
