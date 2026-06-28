"""Lightweight smoke checks for the censorship module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_loader import load_binary_temporal_model, load_convnext_image_classifier
from .run_censorship import run_pipeline


def _load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _assert_valid_boxes(frame_logs: list[dict]) -> None:
    required_fields = {
        "roi_found",
        "roi_valid",
        "roi_area_ratio",
        "censor_region_mode",
        "localization_scope",
        "person_roi_box",
        "body_region_box",
        "patch_boxes_before_roi_mapping",
        "fallback_type",
        "full_frame_censored",
        "person_roi_used",
        "patch_heatmap_used",
    }
    for row in frame_logs:
        missing = required_fields - set(row)
        assert not missing, f"Frame log missing fields {sorted(missing)}"
        for box in row.get("boxes", []):
            assert int(box["x2"]) > int(box["x1"]), f"Invalid box width: {box}"
            assert int(box["y2"]) > int(box["y1"]), f"Invalid box height: {box}"
            assert int(box["x1"]) >= 0 and int(box["y1"]) >= 0, f"Negative box coordinate: {box}"


def run_smoke_test(input_video: str | Path | None, output_video: str | Path | None, device: str) -> None:
    print("Checking model loads...")
    load_binary_temporal_model(device)
    load_convnext_image_classifier(device)
    print("Model load checks passed.")

    if input_video is None:
        print("No --input provided; model-load smoke test complete.")
        return

    result = run_pipeline(input_video, output_video, device=device)
    paths = result["paths"]
    for key in ("video", "windows", "segments", "frame_boxes", "summary"):
        path = Path(paths[key])
        assert path.exists(), f"Expected output missing: {path}"

    frame_logs = _load_json(paths["frame_boxes"])
    summary = _load_json(paths["summary"])
    _assert_valid_boxes(frame_logs)
    assert "fallback_used_frame_count" in summary
    assert "fallback_used_percent" in summary
    for field in (
        "window_seconds",
        "window_stride_seconds",
        "pre_pad_seconds",
        "post_pad_seconds",
        "merge_gap_seconds",
        "full_frame_end_trim_seconds",
        "smoothing_window",
        "enable_scene_cut_boundary",
        "scene_cut_threshold",
        "scene_cut_count",
        "preserve_audio",
        "mute_unsafe_audio",
        "audio_remux_attempted",
        "audio_remux_success",
        "has_input_audio",
        "audio_codec",
        "video_finalize_attempted",
        "video_finalize_success",
        "video_codec",
        "browser_compatible_video",
        "video_finalize_error",
        "fused_policy",
        "temporal_only_mode",
        "num_windows_marked_unsafe",
        "percent_windows_marked_unsafe",
        "frames_with_person_roi",
        "frames_with_valid_person_roi",
        "frames_with_invalid_person_roi",
        "frames_using_body_fallback",
        "frames_using_center_fallback",
        "frames_using_full_frame_debug",
        "frames_using_patch_heatmap",
        "frames_full_frame_censored",
    ):
        assert field in summary, f"Summary missing field: {field}"
    assert "censor_region_mode" in summary

    segments = _load_json(paths["segments"])
    if int(summary.get("num_unsafe_segments", -1)) == 0:
        assert Path(paths["video"]).exists(), "Safe no-censor test failed: output video missing."
        assert Path(paths["windows"]).exists(), "Safe no-censor test failed: windows JSON missing."
        assert segments == [], "Safe no-censor test failed: segments JSON should be an empty list."
        assert Path(paths["frame_boxes"]).exists(), "Safe no-censor test failed: frame boxes JSON missing."
        assert int(summary["num_unsafe_segments"]) == 0

    windows = _load_json(paths["windows"])
    for row in windows:
        assert "scene_cut_from_previous" in row
        assert "scene_cut_score_from_previous" in row

    print("Smoke test passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the censorship module.")
    parser.add_argument("--input", default=None, help="Optional video path for end-to-end smoke test.")
    parser.add_argument("--output", default=None, help="Optional output video path.")
    parser.add_argument("--device", default="cpu", help="Device for smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_smoke_test(args.input, args.output, args.device)


if __name__ == "__main__":
    main()
