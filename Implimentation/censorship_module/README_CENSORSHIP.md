# Standalone Hybrid Visual Fusion Censorship Module

This module censors high-risk regions in videos using only models developed in this project:

- `checkpoints/binary_temporal_roi_best.pt`
- `checkpoints/convnext_sexynude_hardneg_best.pt`

The module supports hybrid visual fusion for debugging, but the recommended demo path is temporal-only full-frame censorship. The binary ROI temporal model decides unsafe segments, then the renderer pixelates or blurs the whole frame only during those unsafe segments.

This is not a segmentation model and does not guarantee perfect nude-region localization.

For the final demo, the recommended default is safety-first full-frame censorship inside unsafe temporal segments. This is the clearest and safest demo behavior: safe frames remain unchanged, and frames inside unsafe segments are fully pixelated or blurred.

## How It Works

1. The binary temporal ROI model scans sliding video windows and predicts whether each segment is unsafe.
2. Optional debug modes can use the ConvNeXt image classifier to score sampled frames with:

   ```python
   risk = prob_nude + 0.5 * prob_sexy
   ```

3. By default, unsafe windows are decided by the binary temporal ROI model only. ConvNeXt frame risk is disabled in default full-frame mode and does not affect unsafe decisions unless `FUSED_UNSAFE_POLICY` is changed.
4. Unsafe windows are converted into tight unsafe segments using scene-cut-aware smoothing and merging.
5. In the default `full_frame` mode, every frame inside an unsafe segment is censored fully.
6. In `person_body` mode, the module finds a valid person/body ROI and censors the configured central body region inside that ROI.
7. In `patch_debug` mode, experimental patch heatmaps can be used for debugging only.
8. OpenCV applies pixelation or blur to the selected boxes.

The module does not use NudeNet, external nudity detectors, internet, or downloaded models.

## Run From The Project Root

Use module execution from the implementation directory to avoid import-path issues:

```bash
cd "/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation"

python -m censorship_module.run_censorship \
  --input path/to/input.mp4 \
  --output outputs/censored/output_censored.mp4 \
  --device cuda \
  --mode pixelate \
  --fused-policy temporal_only \
  --censor-region-mode full_frame
```

## Run The Web Demo

Start the local Streamlit dashboard from the implementation directory:

```bash
cd "/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation"
streamlit run app.py
```

The webapp is a simple local demo: upload a video, choose pixelate or blur, set the temporal unsafe threshold, keep original audio if `ffmpeg` is available, then download the censored video and summary JSON.

If `streamlit` is not installed in your active environment, install it first:

```bash
python -m pip install streamlit
```

For CPU:

```bash
python -m censorship_module.run_censorship \
  --input path/to/input.mp4 \
  --output outputs/censored/output_censored.mp4 \
  --device cpu \
  --mode pixelate \
  --fused-policy temporal_only \
  --censor-region-mode full_frame
```

## Output Files

For an input named `sample.mp4`, the module writes:

- `outputs/censored/sample_censored.mp4`
- `outputs/censored/sample_windows.json`
- `outputs/censored/sample_segments.json`
- `outputs/censored/sample_frame_boxes.json`
- `outputs/censored/sample_summary.json`

If you pass a custom `--output`, the video uses that path and the JSON logs are written beside it.

## ROI Modes

`ROI_MODE` has three options:

- `auto`: try the same local project ROI/person crop logic used during ROI temporal training. If unavailable, fall back to full-frame sampling and log a warning.
- `yolo`: require the local ROI detector and local weights; fail clearly if unavailable.
- `none`: use full-frame sampling.

No YOLO weights are downloaded. Only local project ROI code/weights are used if already available.

The summary JSON reports:

- `roi_mode`
- `roi_available`
- `roi_fallback_used`
- `roi_frames_found`
- `roi_frames_total`
- `roi_frames_found_percent`

## Censor Region Modes

Final demo mode defaults to safety-first full-frame censorship:

```python
CENSOR_REGION_MODE = "full_frame"
```

Available modes:

- `full_frame`: recommended safety-first demo mode. It does not run person ROI localization or patch heatmaps. Frames inside unsafe temporal segments are fully censored; frames outside unsafe segments are unchanged.
- `person_body`: selective but approximate. It finds a valid person ROI and censors the configured central body region inside that ROI.
- `patch_debug`: experimental. Weak classifier heatmaps can respond to background regions, so this mode is for debugging only.

The person-body mode uses:

```python
BODY_FALLBACK_VERTICAL_RANGE = [0.25, 0.90]
BODY_FALLBACK_HORIZONTAL_RANGE = [0.15, 0.85]
MIN_PERSON_ROI_AREA_RATIO = 0.03
MAX_PERSON_ROI_AREA_RATIO = 0.80
```

Patch heatmaps are retained only for debugging:

```python
CENSOR_REGION_MODE = "patch_debug"
PATCH_RISK_THRESHOLD = 0.85
TOP_PERCENTILE = 95
FRAME_RISK_THRESHOLD = 0.90
REQUIRE_PERSON_ROI_FOR_PATCHES = True
PERSON_ROI_BODY_FALLBACK = True
```

Legacy `LOCALIZATION_SCOPE` values remain supported internally for compatibility, but new demos should use `CENSOR_REGION_MODE`.

The frame logs include `censor_region_mode`, `full_frame_censored`, `person_roi_used`, `roi_found`, `roi_valid`, `roi_area_ratio`, `person_roi_box`, `body_region_box`, `patch_heatmap_used`, and `fallback_type`.

## Temporal-Only Default

The default fusion policy is temporal-only:

```python
FUSED_UNSAFE_POLICY = "temporal_only"
TEMPORAL_UNSAFE_THRESHOLD = 0.50
```

In this mode, a window is unsafe only when `temporal_unsafe_prob >= TEMPORAL_UNSAFE_THRESHOLD`. Frame risk does not mark windows unsafe unless you explicitly switch to `frame_only`, `or`, or `weighted`. In default full-frame mode, frame-risk scoring is disabled for speed unless `COMPUTE_FRAME_RISK_IN_FULL_FRAME_MODE=True`.

## Stable Timing Defaults

The recommended demo restores the stable temporal settings:

```python
WINDOW_SECONDS = 4.0
WINDOW_STRIDE_SECONDS = 2.0
PRE_PAD_SECONDS = 0.0
POST_PAD_SECONDS = 0.0
MERGE_GAP_SECONDS = 0.5
SMOOTHING_WINDOW = 3
ENABLE_SCENE_CUT_BOUNDARY = False
SCENE_CUT_THRESHOLD = 35.0
FULL_FRAME_END_TRIM_SECONDS = 0.35
```

`PRE_PAD_SECONDS = 0.0` and `POST_PAD_SECONDS = 0.0` mean detected unsafe segments are not expanded. `FULL_FRAME_END_TRIM_SECONDS = 0.35` is a render-only correction that ends full-frame blur slightly earlier without changing the saved segment detections. Scene-cut logic remains available in code but is disabled by default.

The summary JSON includes the active window, smoothing, merge-gap, and full-frame trim settings.

## Speed Warning

Patch heatmaps can be slow in `patch_debug` mode. For faster debug inference, increase `LOCALIZE_EVERY_N_FRAMES` or reduce `PATCH_SIZES`.

Example:

```python
LOCALIZE_EVERY_N_FRAMES = 8
PATCH_SIZES = [224]
```

## Fallback Tracking

The summary JSON logs:

- `censor_region_mode`
- `frames_full_frame_censored`
- `fallback_used_frame_count`
- `fallback_used_percent`

In `full_frame` mode, fallback use should usually be low because no person ROI or patch localization is required.

## Audio Preservation

OpenCV writes `mp4v` video-only files, which may not preview reliably in browsers. The current default renders a temporary OpenCV video first, then uses local `ffmpeg` to finalize a browser-compatible H.264 MP4 and remux the original audio stream when available:

```python
PRESERVE_AUDIO = True
MUTE_UNSAFE_AUDIO = False
AUDIO_CODEC = "aac"
```

The final video is encoded as `h264/avc1` with `yuv420p` pixel format for Streamlit/browser playback. If `ffmpeg` is unavailable or finalization fails, the module still writes the censored OpenCV output, but browser preview may fail. The code does not intentionally mute audio unless `MUTE_UNSAFE_AUDIO=True`, and segment-level muting is not implemented in this release.

Summary fields include:

- `preserve_audio`
- `mute_unsafe_audio`
- `has_input_audio`
- `audio_remux_attempted`
- `audio_remux_success`
- `audio_remux_error`
- `audio_codec`
- `video_finalize_attempted`
- `video_finalize_success`
- `video_codec`
- `browser_compatible_video`
- `video_finalize_error`

`MUTE_UNSAFE_AUDIO=True` is accepted but not implemented; it prints a warning and preserves original audio when audio preservation is enabled.

## Smoke Test

Check model loading only:

```bash
cd "/media/hazem/6A47B344CD80864E/GP Implementation/Implimentation"
python -m censorship_module.smoke_test --device cpu
```

Run one short video end-to-end:

```bash
python -m censorship_module.smoke_test \
  --input path/to/test_video.mp4 \
  --output outputs/censored/test_censored.mp4 \
  --device cuda
```

The smoke test checks that both checkpoints load, the output video is created, JSON logs are created, frame boxes have valid coordinates, fallback fields exist, and safe no-censor runs still produce all expected files.
