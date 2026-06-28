"""Command-line entrypoint for standalone hybrid visual censorship."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from . import config
from .censor_renderer import process_video_with_segments
from .model_loader import load_binary_temporal_model, load_convnext_image_classifier
from .video_window_detector import (
    get_video_metadata,
    merge_unsafe_segments,
    predict_video_windows,
    smooth_window_predictions,
)


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _notify_progress(callback: Callable[[float], None] | None, value: float) -> None:
    if callback is not None:
        callback(float(max(0.0, min(1.0, value))))


def _notify_status(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(str(message))


@contextmanager
def _temporary_config(overrides: dict[str, Any]):
    previous = {name: getattr(config, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(config, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(config, name, value)


def _output_paths(input_video: Path, output_video: Path | None) -> dict:
    if output_video is None:
        out_dir = Path(config.OUTPUT_DIR)
        output_video = out_dir / f"{input_video.stem}_censored.mp4"
    else:
        output_video = Path(output_video)
        out_dir = output_video.parent
    return {
        "video": output_video,
        "windows": out_dir / f"{input_video.stem}_windows.json",
        "segments": out_dir / f"{input_video.stem}_segments.json",
        "frame_boxes": out_dir / f"{input_video.stem}_frame_boxes.json",
        "summary": out_dir / f"{input_video.stem}_summary.json",
    }


def _roi_summary(windows: list[dict]) -> dict:
    roi_total = int(sum(int(row.get("roi_frames_total", 0)) for row in windows))
    roi_found = int(sum(int(row.get("roi_frames_found", 0)) for row in windows))
    roi_available = bool(any(row.get("roi_available", False) for row in windows))
    roi_fallback_used = bool(any(row.get("roi_fallback_used", False) for row in windows))
    return {
        "roi_mode": str(config.ROI_MODE),
        "roi_available": roi_available,
        "roi_fallback_used": roi_fallback_used,
        "roi_frames_found": roi_found,
        "roi_frames_total": roi_total,
        "roi_frames_found_percent": float(100.0 * roi_found / roi_total) if roi_total else 0.0,
    }


def _has_input_audio(input_video: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(input_video),
            ],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and "audio" in result.stdout.lower()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    result = subprocess.run([ffmpeg, "-i", str(input_video)], capture_output=True, text=True)
    probe_text = f"{result.stdout}\n{result.stderr}".lower()
    return "audio:" in probe_text


def _finalize_video_to_browser_mp4(
    *,
    video_only_path: Path,
    original_input: Path,
    final_output: Path,
) -> dict:
    has_audio = _has_input_audio(original_input)
    preserve_audio = bool(config.PRESERVE_AUDIO)
    mute_unsafe_audio = bool(config.MUTE_UNSAFE_AUDIO)
    result = {
        "preserve_audio": preserve_audio,
        "mute_unsafe_audio": mute_unsafe_audio,
        "has_input_audio": bool(has_audio),
        "audio_remux_attempted": False,
        "audio_remux_success": False,
        "audio_remux_error": None,
        "audio_codec": str(getattr(config, "AUDIO_CODEC", "aac")),
        "video_only_output": str(video_only_path),
        "temp_video_only_path": None,
        "video_finalize_attempted": False,
        "video_finalize_success": False,
        "video_codec": "libx264",
        "browser_compatible_video": False,
        "video_finalize_error": None,
    }

    if mute_unsafe_audio:
        warning = "MUTE_UNSAFE_AUDIO=True is not implemented in this patch; preserving original audio if enabled."
        print(f"WARNING: {warning}")
        result["audio_remux_error"] = warning

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        error = "ffmpeg not found; keeping OpenCV video-only output. Browser preview may fail."
        print(f"WARNING: {error}")
        result["video_finalize_error"] = error
        result["audio_remux_error"] = error
        if video_only_path.resolve() != final_output.resolve():
            final_output.parent.mkdir(parents=True, exist_ok=True)
            video_only_path.replace(final_output)
            result["temp_video_only_path"] = str(final_output)
        return result

    result["video_finalize_attempted"] = True
    result["audio_remux_attempted"] = bool(preserve_audio and has_audio)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_only_path),
    ]
    if preserve_audio and has_audio:
        cmd += [
            "-i",
            str(original_input),
        ]
    cmd += [
        "-map",
        "0:v:0",
    ]
    if preserve_audio and has_audio:
        cmd += [
            "-map",
            "1:a:0?",
        ]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
    ]
    if preserve_audio and has_audio:
        cmd += [
            "-c:a",
            str(getattr(config, "AUDIO_CODEC", "aac")),
            "-shortest",
        ]
    else:
        cmd += ["-an"]
    cmd += [
        "-movflags",
        "+faststart",
        str(final_output),
    ]
    finalize = subprocess.run(cmd, capture_output=True, text=True)
    if finalize.returncode == 0 and final_output.exists():
        result["video_finalize_success"] = True
        result["browser_compatible_video"] = True
        result["audio_remux_success"] = bool(preserve_audio and has_audio)
        try:
            video_only_path.unlink()
        except FileNotFoundError:
            pass
        return result

    error = (finalize.stderr or finalize.stdout or "Unknown ffmpeg finalization failure").strip()
    print(f"WARNING: H.264 video finalization failed; keeping OpenCV output. Browser preview may fail. {error}")
    result["video_finalize_error"] = error
    if preserve_audio and has_audio:
        result["audio_remux_error"] = error
    if video_only_path.resolve() != final_output.resolve():
        final_output.parent.mkdir(parents=True, exist_ok=True)
        video_only_path.replace(final_output)
        result["temp_video_only_path"] = str(final_output)
    else:
        result["temp_video_only_path"] = str(video_only_path)
    return result


def _remux_audio_to_final(
    *,
    video_only_path: Path,
    original_input: Path,
    final_output: Path,
) -> dict:
    """Backward-compatible alias for the browser-compatible finalization path."""

    return _finalize_video_to_browser_mp4(
        video_only_path=video_only_path,
        original_input=original_input,
        final_output=final_output,
    )


def _run_pipeline_with_active_config(
    input_video: str | Path,
    output_video: str | Path | None = None,
    *,
    device: str = "cuda",
    progress_callback: Callable[[float], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> dict:
    input_video = Path(input_video)
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"
    torch_device = torch.device(device)

    paths = _output_paths(input_video, Path(output_video) if output_video else None)
    final_video_path = Path(paths["video"])
    video_only_path = final_video_path.with_name(f"{final_video_path.stem}_video_only_tmp{final_video_path.suffix}")

    _notify_status(status_callback, "Loading censorship models...")
    _notify_progress(progress_callback, 0.05)
    temporal_model, temporal_path = load_binary_temporal_model(torch_device, return_path=True)
    image_model = None
    image_path = None
    needs_image_model = not (
        str(config.CENSOR_REGION_MODE).lower() == "full_frame"
        and str(config.FUSED_UNSAFE_POLICY).lower() == "temporal_only"
        and not bool(getattr(config, "COMPUTE_FRAME_RISK_IN_FULL_FRAME_MODE", False))
    )
    if needs_image_model:
        image_model, image_path = load_convnext_image_classifier(torch_device, return_path=True)

    _notify_status(status_callback, "Reading video metadata...")
    video_metadata = get_video_metadata(input_video)
    _notify_status(status_callback, "Analyzing video windows...")
    _notify_progress(progress_callback, 0.20)
    raw_windows = predict_video_windows(input_video, temporal_model, image_model, torch_device)
    _notify_status(status_callback, "Smoothing temporal predictions...")
    windows = smooth_window_predictions(raw_windows)
    _notify_status(status_callback, "Merging unsafe segments...")
    _notify_progress(progress_callback, 0.45)
    segments = merge_unsafe_segments(windows, video_duration=float(video_metadata.get("duration", 0.0)))
    frame_logs, render_stats = process_video_with_segments(
        input_video,
        video_only_path,
        segments,
        image_model,
        torch_device,
        progress_callback=progress_callback,
        status_callback=status_callback,
        progress_start=0.50,
        progress_end=0.90,
    )
    _notify_status(status_callback, "Finalizing browser-compatible video...")
    _notify_progress(progress_callback, 0.95)
    audio_stats = _remux_audio_to_final(
        video_only_path=video_only_path,
        original_input=input_video,
        final_output=final_video_path,
    )
    _notify_progress(progress_callback, 1.00)
    _notify_status(status_callback, "Censorship complete.")

    _write_json(paths["windows"], windows)
    _write_json(paths["segments"], segments)
    _write_json(paths["frame_boxes"], frame_logs)

    total_unsafe_duration = float(sum(max(0.0, float(seg["end"]) - float(seg["start"])) for seg in segments))
    num_windows = len(windows)
    num_windows_marked_unsafe = int(sum(1 for row in windows if row.get("unsafe")))
    percent_windows_marked_unsafe = float(100.0 * num_windows_marked_unsafe / num_windows) if num_windows else 0.0
    if percent_windows_marked_unsafe > 80.0:
        print("WARNING: More than 80% of windows marked unsafe. Check thresholds/fusion policy.")
    summary = {
        "input_video": str(input_video),
        "output_video": str(final_video_path),
        "temporal_threshold": float(config.TEMPORAL_UNSAFE_THRESHOLD),
        "frame_risk_threshold": float(config.FRAME_RISK_THRESHOLD),
        "patch_risk_threshold": float(config.PATCH_RISK_THRESHOLD),
        "fused_policy": str(config.FUSED_UNSAFE_POLICY),
        "temporal_only_mode": str(config.FUSED_UNSAFE_POLICY).lower() == "temporal_only",
        "num_windows_marked_unsafe": int(num_windows_marked_unsafe),
        "percent_windows_marked_unsafe": float(percent_windows_marked_unsafe),
        "window_seconds": float(config.WINDOW_SECONDS),
        "window_stride_seconds": float(config.WINDOW_STRIDE_SECONDS),
        "pre_pad_seconds": float(config.PRE_PAD_SECONDS),
        "post_pad_seconds": float(config.POST_PAD_SECONDS),
        "merge_gap_seconds": float(config.MERGE_GAP_SECONDS),
        "full_frame_end_trim_seconds": float(config.FULL_FRAME_END_TRIM_SECONDS),
        "smoothing_window": int(config.SMOOTHING_WINDOW),
        "enable_scene_cut_boundary": bool(config.ENABLE_SCENE_CUT_BOUNDARY),
        "scene_cut_threshold": float(config.SCENE_CUT_THRESHOLD),
        "scene_cut_count": int(sum(1 for row in windows if row.get("scene_cut_from_previous"))),
        "num_unsafe_segments": int(len(segments)),
        "total_unsafe_duration": total_unsafe_duration,
        "max_temporal_unsafe_prob": float(max([row.get("temporal_unsafe_prob", 0.0) for row in windows] or [0.0])),
        "max_frame_clip_risk": float(max([row.get("frame_clip_risk", 0.0) for row in windows] or [0.0])),
        "max_fused_score": float(max([row.get("fused_score", 0.0) for row in windows] or [0.0])),
        "censor_mode": str(config.CENSOR_MODE),
        "pixelation_factor": int(config.PIXELATION_FACTOR),
        "blur_kernel": int(config.BLUR_KERNEL),
        "blur_passes": int(getattr(config, "BLUR_PASSES", 1)),
        "fallback_mode": str(config.FALLBACK_MODE),
        "fallback_used_frame_count": int(render_stats["fallback_used_frame_count"]),
        "fallback_used_percent": float(render_stats["fallback_used_percent"]),
        "max_area_filter_frame_count": int(render_stats["max_area_filter_frame_count"]),
        "frames_with_person_roi": int(render_stats["frames_with_person_roi"]),
        "frames_with_valid_person_roi": int(render_stats["frames_with_valid_person_roi"]),
        "frames_with_invalid_person_roi": int(render_stats["frames_with_invalid_person_roi"]),
        "frames_using_body_fallback": int(render_stats["frames_using_body_fallback"]),
        "frames_using_center_fallback": int(render_stats["frames_using_center_fallback"]),
        "frames_using_full_frame_debug": int(render_stats["frames_using_full_frame_debug"]),
        "frames_using_patch_heatmap": int(render_stats["frames_using_patch_heatmap"]),
        "frames_full_frame_censored": int(render_stats["frames_full_frame_censored"]),
        "censor_region_mode": str(config.CENSOR_REGION_MODE),
        "localization_scope": str(config.LOCALIZATION_SCOPE),
        "preserve_audio": bool(config.PRESERVE_AUDIO),
        "mute_unsafe_audio": bool(config.MUTE_UNSAFE_AUDIO),
        "audio_remux_attempted": bool(audio_stats["audio_remux_attempted"]),
        "audio_remux_success": bool(audio_stats["audio_remux_success"]),
        "has_input_audio": bool(audio_stats["has_input_audio"]),
        "audio_remux_error": audio_stats.get("audio_remux_error"),
        "audio_codec": audio_stats.get("audio_codec"),
        "temp_video_only_path": audio_stats.get("temp_video_only_path"),
        "video_finalize_attempted": bool(audio_stats["video_finalize_attempted"]),
        "video_finalize_success": bool(audio_stats["video_finalize_success"]),
        "video_codec": audio_stats.get("video_codec"),
        "browser_compatible_video": bool(audio_stats["browser_compatible_video"]),
        "video_finalize_error": audio_stats.get("video_finalize_error"),
        "models": {
            "binary_temporal": str(temporal_path),
            "image_classifier": str(image_path) if image_path is not None else None,
        },
        **_roi_summary(windows),
        "render_stats": render_stats,
        "output_files": {
            "windows": str(paths["windows"]),
            "segments": str(paths["segments"]),
            "frame_boxes": str(paths["frame_boxes"]),
            "summary": str(paths["summary"]),
        },
        "notes": (
            "Stable demo mode: the binary temporal ROI model detects unsafe segments; "
            "default full-frame rendering censors only frames inside those temporal segments. "
            "ConvNeXt frame risk and patch heatmaps are optional debug paths, not default decisions."
        ),
    }
    _write_json(paths["summary"], summary)
    print(json.dumps(_json_safe(summary), indent=2))
    return {
        "windows": windows,
        "segments": segments,
        "frame_logs": frame_logs,
        "summary": summary,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def run_censorship_pipeline(
    input_video: str | Path,
    output_video: str | Path | None = None,
    *,
    device: str = "cuda",
    mode: str = "pixelate",
    pixelation_factor: int | None = None,
    blur_kernel: int | None = None,
    unsafe_threshold: float | None = None,
    preserve_audio: bool = True,
    progress_callback: Callable[[float], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> dict:
    """Run the stable full-frame temporal censorship pipeline.

    Temporary config overrides are restored after the run so repeated calls from
    Streamlit or notebooks do not leak UI settings into later executions.
    """

    selected_mode = str(mode).lower()
    if selected_mode not in {"pixelate", "blur"}:
        raise ValueError(f"Unsupported censor mode: {mode}")

    overrides: dict[str, Any] = {
        "CENSOR_MODE": selected_mode,
        "PRESERVE_AUDIO": bool(preserve_audio),
        "MUTE_UNSAFE_AUDIO": False,
    }
    if pixelation_factor is not None:
        overrides["PIXELATION_FACTOR"] = int(pixelation_factor)
    if blur_kernel is not None:
        kernel = int(blur_kernel)
        if kernel % 2 == 0:
            kernel += 1
        overrides["BLUR_KERNEL"] = kernel
    if unsafe_threshold is not None:
        overrides["TEMPORAL_UNSAFE_THRESHOLD"] = float(unsafe_threshold)

    with _temporary_config(overrides):
        return _run_pipeline_with_active_config(
            input_video,
            output_video,
            device=device,
            progress_callback=progress_callback,
            status_callback=status_callback,
        )


def run_pipeline(
    input_video: str | Path,
    output_video: str | Path | None = None,
    *,
    device: str = "cuda",
) -> dict:
    return _run_pipeline_with_active_config(input_video, output_video, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid visual fusion censorship module.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output", default=None, help="Output censored video path.")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu.")
    parser.add_argument("--mode", choices=["pixelate", "blur"], default=config.CENSOR_MODE)
    parser.add_argument("--temporal-threshold", type=float, default=config.TEMPORAL_UNSAFE_THRESHOLD)
    parser.add_argument("--frame-risk-threshold", type=float, default=config.FRAME_RISK_THRESHOLD)
    parser.add_argument("--patch-risk-threshold", type=float, default=config.PATCH_RISK_THRESHOLD)
    parser.add_argument("--fused-policy", choices=["temporal_only", "frame_only", "or", "weighted"], default=config.FUSED_UNSAFE_POLICY)
    parser.add_argument("--fallback", choices=["center_region", "whole_frame_last_resort", "none"], default=config.FALLBACK_MODE)
    parser.add_argument("--roi-mode", choices=["auto", "yolo", "none"], default=config.ROI_MODE)
    parser.add_argument("--censor-region-mode", choices=["full_frame", "person_body", "patch_debug"], default=config.CENSOR_REGION_MODE)
    parser.add_argument("--preserve-audio", dest="preserve_audio", action="store_true", default=config.PRESERVE_AUDIO)
    parser.add_argument("--no-preserve-audio", dest="preserve_audio", action="store_false")
    parser.add_argument("--mute-unsafe-audio", action="store_true", default=config.MUTE_UNSAFE_AUDIO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.TEMPORAL_UNSAFE_THRESHOLD = float(args.temporal_threshold)
    config.FRAME_RISK_THRESHOLD = float(args.frame_risk_threshold)
    config.PATCH_RISK_THRESHOLD = float(args.patch_risk_threshold)
    config.FUSED_UNSAFE_POLICY = args.fused_policy
    config.FALLBACK_MODE = args.fallback
    config.ROI_MODE = args.roi_mode
    config.CENSOR_REGION_MODE = args.censor_region_mode
    config.LOCALIZATION_SCOPE = args.censor_region_mode
    if bool(args.mute_unsafe_audio):
        print("WARNING: --mute-unsafe-audio is accepted but segment-level audio muting is not implemented.")
    run_censorship_pipeline(
        args.input,
        args.output,
        device=args.device,
        mode=args.mode,
        unsafe_threshold=float(args.temporal_threshold),
        preserve_audio=bool(args.preserve_audio),
    )


if __name__ == "__main__":
    main()
