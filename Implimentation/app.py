from __future__ import annotations

import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch

from censorship_module import config as censorship_config
from censorship_module.run_censorship import run_censorship_pipeline


APP_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = APP_ROOT / "uploads"
OUTPUT_DIR = APP_ROOT / "outputs" / "censored"
PREVIEW_IMAGE_PATH = APP_ROOT.parent / "Hazem Zidan.jpeg"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return stem or "uploaded_video"


def file_size_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "Calculating..."
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def estimate_eta(progress: float, started_at: float | None) -> float | None:
    if started_at is None:
        return None
    progress = float(max(0.0, min(1.0, progress)))
    elapsed = time.perf_counter() - float(started_at)
    if progress < 0.03 or progress >= 1.0 or elapsed < 1.0:
        return None
    return max(0.0, (elapsed / progress) * (1.0 - progress))


def render_timing_box(elapsed_box, progress: float, started_at: float | None, status: str) -> None:
    if started_at is None:
        elapsed_box.caption("ETA will appear once processing starts.")
        return
    elapsed = time.perf_counter() - float(started_at)
    eta = estimate_eta(progress, started_at)
    if float(progress) >= 1.0:
        elapsed_box.success(f"Completed in {format_duration(elapsed)}")
        return
    elapsed_box.info(
        f"Elapsed: {format_duration(elapsed)} · ETA: {format_duration(eta)} · Current action: {status}"
    )


def scroll_to_process() -> None:
    components.html(
        """
        <script>
        const target = window.parent.document.getElementById("process");
        if (target) {
          target.scrollIntoView({behavior: "smooth", block: "start"});
        }
        </script>
        """,
        height=0,
    )


def render_css() -> None:
    st.markdown(
        """
        <style>
        #MainMenu, footer, header { visibility: hidden; }
        html { scroll-behavior: smooth; }
        .stApp {
            background: #f7f4ef;
            color: #111111;
        }
        .block-container {
            max-width: 1280px;
            padding: 1.25rem 2rem 4rem;
        }
        p, label, span, div {
            font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .top-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 18px 0 28px;
            gap: 28px;
        }
        .brand {
            color: #111111;
            font-size: 1.1rem;
            font-weight: 800;
            letter-spacing: -0.04em;
        }
        .nav-links {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 28px;
            color: #6b6b6b;
            font-size: 0.92rem;
            font-weight: 650;
        }
        .nav-links a {
            color: #6b6b6b;
            text-decoration: none;
        }
        .nav-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            background: #050505;
            color: #ffffff !important;
            padding: 0.78rem 1.25rem;
            font-weight: 750;
            text-decoration: none;
            box-shadow: 0 16px 34px rgba(0,0,0,0.16);
        }
        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.35fr) minmax(360px, 0.65fr);
            gap: 34px;
            align-items: stretch;
            margin: 34px 0 72px;
        }
        .eyebrow {
            display: inline-flex;
            border: 1px solid rgba(17,17,17,0.12);
            border-radius: 999px;
            padding: 0.52rem 0.85rem;
            color: #555555;
            font-size: 0.8rem;
            font-weight: 750;
            margin-bottom: 24px;
            background: rgba(255,255,255,0.42);
        }
        .hero-title {
            color: #111111;
            font-size: clamp(48px, 7vw, 96px);
            line-height: 0.95;
            letter-spacing: -0.065em;
            font-weight: 760;
            margin: 0;
            max-width: 900px;
        }
        .hero-subtitle {
            color: #666666;
            font-size: 18px;
            line-height: 1.65;
            margin: 28px 0 30px;
            max-width: 650px;
        }
        .cta-row {
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }
        .black-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #050505;
            color: #ffffff !important;
            border-radius: 999px;
            padding: 0.9rem 1.35rem;
            font-weight: 750;
            text-decoration: none;
            box-shadow: 0 20px 38px rgba(0,0,0,0.16);
        }
        .text-link {
            color: #111111 !important;
            font-weight: 750;
            text-decoration: none;
            border-bottom: 1px solid rgba(17,17,17,0.35);
        }
        .agency-card {
            background: #ffffff;
            border-radius: 28px;
            border: 1px solid rgba(17, 17, 17, 0.08);
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.08);
            padding: 28px;
            margin-bottom: 22px;
        }
        .status-card {
            min-height: 430px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .status-orb {
            width: 138px;
            height: 138px;
            border-radius: 999px;
            display: grid;
            place-items: center;
            background:
                radial-gradient(circle at center, #ffffff 56%, transparent 57%),
                conic-gradient(#050505 var(--pct), rgba(17,17,17,0.10) 0);
            border: 1px solid rgba(17,17,17,0.08);
        }
        .status-number {
            color: #111111;
            font-size: 1.5rem;
            font-weight: 820;
            line-height: 1;
            text-align: center;
        }
        .status-label {
            color: #6b6b6b;
            font-size: 0.78rem;
            margin-top: 6px;
            display: block;
        }
        .meta-list {
            display: grid;
            gap: 12px;
            margin-top: 26px;
        }
        .meta-row {
            display: flex;
            justify-content: space-between;
            gap: 18px;
            border-bottom: 1px solid rgba(17,17,17,0.08);
            padding-bottom: 12px;
            color: #111111;
            font-size: 0.95rem;
        }
        .meta-row span:first-child {
            color: #777777;
        }
        .section {
            margin: 64px 0;
        }
        .section-heading {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 28px;
            margin-bottom: 24px;
        }
        .section-kicker {
            color: #6b6b6b;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .section-title {
            color: #111111;
            font-size: clamp(30px, 4vw, 56px);
            line-height: 1;
            letter-spacing: -0.052em;
            font-weight: 760;
            margin: 0;
        }
        .section-copy {
            color: #666666;
            font-size: 1rem;
            line-height: 1.65;
            max-width: 430px;
            margin: 0;
        }
        .card-title {
            color: #111111;
            font-size: 1.35rem;
            font-weight: 780;
            letter-spacing: -0.035em;
            margin: 0 0 8px;
        }
        .card-copy {
            color: #6b6b6b;
            font-size: 0.95rem;
            line-height: 1.55;
            margin: 0 0 20px;
        }
        .preview-caption {
            color: #6b6b6b;
            font-size: 0.88rem;
            line-height: 1.45;
            margin-top: -4px;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid rgba(17,17,17,0.08);
            border-radius: 24px;
            padding: 22px;
            box-shadow: 0 18px 45px rgba(0,0,0,0.06);
        }
        .metric-value {
            color: #111111;
            font-size: 1.72rem;
            font-weight: 820;
            letter-spacing: -0.045em;
            margin: 0 0 4px;
        }
        .metric-label {
            color: #6b6b6b;
            font-size: 0.84rem;
            font-weight: 700;
            margin: 0;
        }
        .system-card {
            min-height: 184px;
        }
        .footer-note {
            color: #6b6b6b;
            border-top: 1px solid rgba(17,17,17,0.10);
            padding-top: 28px;
            margin-top: 72px;
            font-size: 0.92rem;
            line-height: 1.6;
        }
        div.stButton > button:first-child,
        div.stDownloadButton > button:first-child {
            width: 100%;
            border-radius: 999px;
            background: #050505;
            color: #ffffff;
            border: 0;
            padding: 0.88rem 1.4rem;
            font-weight: 750;
            box-shadow: 0 16px 34px rgba(0,0,0,0.16);
        }
        div.stButton > button:first-child:hover,
        div.stDownloadButton > button:first-child:hover {
            background: #252525;
            color: #ffffff;
            border: 0;
        }
        div.stButton > button:first-child *,
        div.stDownloadButton > button:first-child * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        .stProgress > div > div > div > div {
            background-color: #050505;
        }
        [data-testid="stFileUploader"] section {
            border-radius: 24px;
            border-color: rgba(17,17,17,0.12);
            background: #fbfaf7;
        }
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p {
            color: #111111 !important;
        }
        [data-baseweb="select"] > div,
        [data-baseweb="popover"] ul,
        [data-baseweb="menu"] {
            background: #ffffff !important;
            color: #111111 !important;
            border-color: rgba(17,17,17,0.14) !important;
        }
        [data-baseweb="select"] span,
        [data-baseweb="select"] svg,
        [data-baseweb="menu"] li,
        [data-baseweb="menu"] div {
            color: #111111 !important;
            fill: #111111 !important;
        }
        div[role="radiogroup"] label,
        div[role="radiogroup"] p,
        [data-testid="stCheckbox"] label,
        [data-testid="stCheckbox"] p,
        [data-testid="stCheckbox"] span,
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] p,
        [data-testid="stRadio"] span {
            color: #111111 !important;
        }
        [data-testid="stSlider"] label,
        [data-testid="stSlider"] p,
        [data-testid="stSlider"] span,
        [data-testid="stSlider"] div {
            color: #111111 !important;
        }
        [data-testid="stSlider"] [role="slider"] {
            background: #050505 !important;
            border-color: #050505 !important;
        }
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] button {
            color: #111111 !important;
        }
        [data-testid="stFileUploader"] button {
            background: #ffffff !important;
            border: 1px solid rgba(17,17,17,0.16) !important;
            border-radius: 999px !important;
        }
        [data-testid="stAlert"] {
            color: #111111 !important;
            border-radius: 18px !important;
        }
        [data-testid="stVideo"] {
            border-radius: 22px;
            overflow: hidden;
            border: 1px solid rgba(17,17,17,0.08);
        }
        @media (max-width: 900px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }
            .nav-links {
                display: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_nav() -> None:
    st.markdown(
        """
        <nav class="top-nav">
          <div class="brand">CensorAI</div>
          <div class="nav-links">
            <a href="#upload">Upload</a>
            <a href="#process">Process</a>
            <a href="#results">Results</a>
            <a href="#system">System</a>
          </div>
          <a class="nav-pill" href="#upload">Start Demo</a>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def render_hero(progress: float, status: str, keep_audio: bool = True, threshold: float | None = None) -> None:
    pct = int(round(max(0.0, min(1.0, progress)) * 100))
    threshold_text = f"{threshold:.2f}" if threshold is not None else f"{float(censorship_config.TEMPORAL_UNSAFE_THRESHOLD):.2f}"
    audio_text = "Preserved" if keep_audio else "Video only"
    st.markdown(
        f"""
        <section class="hero-grid">
          <div>
            <div class="eyebrow">AI media safety demo</div>
            <h1 class="hero-title">AI-powered video censorship, built for safer media review.</h1>
            <div class="cta-row">
              <a class="black-pill" href="#upload">Start censorship</a>
              <a class="text-link" href="#system">How it works</a>
            </div>
          </div>
          <div class="agency-card status-card">
            <div>
              <div class="section-kicker">System status</div>
              <div class="status-orb" style="--pct: {pct}%;">
                <div class="status-number">{pct}%<span class="status-label">{status}</span></div>
              </div>
            </div>
            <div class="meta-list">
              <div class="meta-row"><span>Final Model</span><strong>Binary ROI Temporal</strong></div>
              <div class="meta-row"><span>Censorship Mode</span><strong>Full-frame safety</strong></div>
              <div class="meta-row"><span>Audio</span><strong>{audio_text}</strong></div>
              <div class="meta-row"><span>Threshold</span><strong>{threshold_text}</strong></div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section_heading(kicker: str, title: str, copy: str = "") -> None:
    copy_html = f'<p class="section-copy">{copy}</p>' if copy else ""
    st.markdown(
        f"""
        <div class="section-heading">
          <div>
            <div class="section-kicker">{kicker}</div>
            <h2 class="section-title">{title}</h2>
          </div>
          {copy_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_preview_base() -> np.ndarray:
    image = cv2.imread(str(PREVIEW_IMAGE_PATH), cv2.IMREAD_COLOR)
    if image is not None:
        max_preview_width = 900
        height, width = image.shape[:2]
        if width > max_preview_width:
            scale = max_preview_width / float(width)
            image = cv2.resize(image, (max_preview_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        return image

    height, width = 260, 360
    yy, xx = np.indices((height, width))
    canvas = np.dstack(
        [
            236 + 13 * np.sin(xx / 17.0) + 6 * np.cos(yy / 15.0),
            226 + 10 * np.cos(xx / 23.0) + 8 * np.sin(yy / 18.0),
            210 + 12 * np.sin((xx + yy) / 31.0),
        ]
    ).clip(0, 255).astype(np.uint8)
    rng = np.random.default_rng(7)
    texture = rng.normal(0, 8, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + texture, 0, 255).astype(np.uint8)

    cv2.rectangle(canvas, (18, 18), (342, 242), (255, 255, 255), 2)
    for x in range(32, width, 34):
        cv2.line(canvas, (x, 24), (x - 74, height - 18), (217, 204, 185), 1, cv2.LINE_AA)
    for y in range(38, height, 34):
        cv2.line(canvas, (22, y), (width - 24, y + 18), (229, 217, 198), 1, cv2.LINE_AA)

    cv2.circle(canvas, (105, 88), 38, (42, 42, 42), -1)
    cv2.rectangle(canvas, (72, 126), (138, 220), (32, 32, 32), -1)
    cv2.rectangle(canvas, (176, 65), (314, 92), (184, 174, 154), -1)
    cv2.rectangle(canvas, (176, 112), (314, 139), (204, 188, 160), -1)
    cv2.rectangle(canvas, (176, 159), (284, 186), (166, 154, 136), -1)
    cv2.putText(canvas, "FULL FRAME", (174, 217), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (17, 17, 17), 2, cv2.LINE_AA)
    cv2.putText(canvas, "PREVIEW", (174, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (17, 17, 17), 2, cv2.LINE_AA)
    return canvas


def make_preview_tile(mode: str, strength: int) -> np.ndarray:
    tile = make_preview_base()
    if mode == "pixelate":
        factor = max(2, int(strength))
        small = cv2.resize(tile, (max(1, tile.shape[1] // factor), max(1, tile.shape[0] // factor)))
        tile = cv2.resize(small, (tile.shape[1], tile.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        kernel = int(strength)
        if kernel % 2 == 0:
            kernel += 1
        for _ in range(max(1, int(getattr(censorship_config, "BLUR_PASSES", 1)))):
            tile = cv2.GaussianBlur(tile, (kernel, kernel), 0)
    return cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)


def _odd_kernel(value: int) -> int:
    value = int(value)
    return value if value % 2 == 1 else value + 1


def active_censor_strength(mode: str, pixelation_factor: int | None, blur_kernel: int | None) -> int:
    if mode == "pixelate":
        return int(pixelation_factor or censorship_config.PIXELATION_FACTOR)
    return _odd_kernel(int(blur_kernel or censorship_config.BLUR_KERNEL))


def selected_style_value(mode: str, pixelation_factor: int | None, blur_kernel: int | None) -> str:
    if mode == "pixelate":
        return f"Current Pixelation level: {active_censor_strength(mode, pixelation_factor, blur_kernel)}"
    return f"Current Blur level: {active_censor_strength(mode, pixelation_factor, blur_kernel)}"


def selected_settings_rows(
    mode: str,
    pixelation_factor: int | None,
    blur_kernel: int | None,
    unsafe_threshold: float,
    keep_audio: bool,
) -> str:
    strength_label = "Pixelation level" if mode == "pixelate" else "Blur level"
    strength = active_censor_strength(mode, pixelation_factor, blur_kernel)
    audio_text = "Preserve original audio" if keep_audio else "Video only"
    blur_pass_row = ""
    if mode == "blur":
        blur_pass_row = f'<div class="meta-row"><span>Blur passes</span><strong>{int(getattr(censorship_config, "BLUR_PASSES", 1))}</strong></div>'
    return f"""
      <div class="meta-list">
        <div class="meta-row"><span>Censorship style</span><strong>{mode.title()}</strong></div>
        <div class="meta-row"><span>{strength_label}</span><strong>{strength}</strong></div>
        {blur_pass_row}
        <div class="meta-row"><span>Unsafe threshold</span><strong>{unsafe_threshold:.2f}</strong></div>
        <div class="meta-row"><span>Audio output</span><strong>{audio_text}</strong></div>
      </div>
    """


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <p class="metric-value">{value}</p>
          <p class="metric-label">{label}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def audio_status(summary: dict) -> str:
    if not summary.get("preserve_audio"):
        return "Disabled"
    if not summary.get("has_input_audio"):
        return "No input audio"
    if summary.get("audio_remux_success"):
        return "Preserved"
    if not summary.get("video_finalize_success", True):
        return "Finalize failed"
    return "Remux failed"


def system_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="agency-card system-card">
          <h3 class="card-title">{title}</h3>
          <p class="card-copy">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_video(path: Path, label: str, browser_warning: bool = False) -> None:
    st.markdown(f'<div class="agency-card"><h3 class="card-title">{label}</h3></div>', unsafe_allow_html=True)
    if not path.exists():
        st.error(f"{label} video file was not found: {path}")
    else:
        if browser_warning:
            st.warning("This file may not preview in the browser because H.264 finalization failed. The download is still available.")
        st.video(str(path))


def main() -> None:
    st.set_page_config(page_title="CensorAI | Video Censorship System", layout="wide")
    render_css()
    render_nav()

    st.session_state.setdefault("progress", 0.0)
    st.session_state.setdefault("status", "Ready")
    st.session_state.setdefault("processing_started_at", None)

    selected_threshold = st.session_state.get("selected_threshold", float(censorship_config.TEMPORAL_UNSAFE_THRESHOLD))
    selected_audio = st.session_state.get("selected_audio", True)
    hero = st.empty()
    with hero.container():
        render_hero(float(st.session_state["progress"]), str(st.session_state["status"]), selected_audio, selected_threshold)

    st.markdown('<section class="section" id="system">', unsafe_allow_html=True)
    section_heading(
        "System",
        "How the system works",
    )
    sys_cols = st.columns(3, gap="large")
    with sys_cols[0]:
        system_card("Temporal detection", "The binary ROI temporal model identifies unsafe time windows from sampled video clips.")
    with sys_cols[1]:
        system_card("Safety-first censorship", "Detected unsafe segments are censored using full-frame pixelation or blur for clear demo behavior.")
    with sys_cols[2]:
        system_card("Audio preservation", "The original audio stream is remuxed back into the browser-ready censored output when available.")
    st.markdown("</section>", unsafe_allow_html=True)

    st.markdown('<section class="section" id="upload">', unsafe_allow_html=True)
    section_heading(
        "Upload",
        "Set up the censorship run.",
    )
    upload_col, controls_col = st.columns(2, gap="large")

    with upload_col:
        st.markdown(
            '<div class="agency-card"><h3 class="card-title">Upload video</h3><p class="card-copy">MP4, MOV, AVI, and MKV are supported for the local demo.</p></div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"], label_visibility="collapsed")
        if uploaded is not None:
            st.caption(f"{uploaded.name} · {file_size_mb(uploaded.size)}")
        start = st.button("Start Censorship", disabled=uploaded is None)

    with controls_col:
        st.markdown(
            '<div class="agency-card"><h3 class="card-title">Controls</h3><p class="card-copy">These controls affect rendering and the temporal threshold only; model logic stays unchanged.</p></div>',
            unsafe_allow_html=True,
        )
        device = st.selectbox("Device", ["cuda", "cpu"], index=0)
        if device == "cuda" and not torch.cuda.is_available():
            st.warning("CUDA is not available. The pipeline will fall back to CPU.")
        mode = st.radio("Censorship mode", ["pixelate", "blur"], horizontal=True)
        pixelation_factor = (
            st.slider("Pixelation level", 8, 80, int(censorship_config.PIXELATION_FACTOR))
            if mode == "pixelate"
            else None
        )
        blur_kernel = (
            st.slider("Blur level", 31, 151, int(censorship_config.BLUR_KERNEL), step=2)
            if mode == "blur"
            else None
        )
        unsafe_threshold = st.slider("Unsafe threshold", 0.30, 0.80, float(selected_threshold), step=0.01)
        st.caption(
            "Unsafe threshold is the minimum temporal unsafe probability required to censor a video window. "
            "Lower values censor more aggressively; higher values censor more conservatively."
        )
        keep_audio = st.checkbox("Keep original audio", value=bool(selected_audio))
        st.session_state["selected_threshold"] = float(unsafe_threshold)
        st.session_state["selected_audio"] = bool(keep_audio)
        if keep_audio and shutil.which("ffmpeg") is None:
            st.warning("ffmpeg was not found. The app will still export video, but preview/audio finalization may fail.")
    st.markdown("</section>", unsafe_allow_html=True)

    st.markdown('<section class="section" id="preview">', unsafe_allow_html=True)
    section_heading(
        "Preview",
        "Preview censorship styles",
    )
    active_strength = active_censor_strength(mode, pixelation_factor, blur_kernel)
    preview_col, applied_col = st.columns([1.25, 0.75], gap="large")
    with preview_col:
        st.image(make_preview_tile(mode, active_strength), use_container_width=True)
        active_label = "Pixelation level" if mode == "pixelate" else "Blur level"
        st.markdown(
            f"""
            <div class="agency-card">
            <h3 class="card-title">Current full-frame {mode} preview</h3>
            <p class="preview-caption">
              {active_label}: {active_strength}. The whole preview image is censored with this exact value, matching the full-frame demo mode.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with applied_col:
        st.markdown(
            f"""
            <div class="agency-card">
              <h3 class="card-title">Run settings</h3>
              <p class="card-copy">These are the live controls that will be applied to the next censorship run.</p>
              {selected_settings_rows(mode, pixelation_factor, blur_kernel, unsafe_threshold, keep_audio)}
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</section>", unsafe_allow_html=True)

    st.markdown('<section class="section" id="process">', unsafe_allow_html=True)
    section_heading(
        "Process",
        "Track the run in real time.",
    )
    status_box = st.empty()
    status_box.info(str(st.session_state["status"]))
    progress_box = st.progress(float(st.session_state["progress"]))
    elapsed_box = st.empty()
    render_timing_box(
        elapsed_box,
        float(st.session_state["progress"]),
        st.session_state.get("processing_started_at"),
        str(st.session_state["status"]),
    )
    stage_cols = st.columns(6)
    for col, label in zip(
        stage_cols,
        ["Loading models", "Analyzing windows", "Detecting segments", "Rendering video", "Preserving audio", "Complete"],
    ):
        col.caption(label)
    st.markdown("</section>", unsafe_allow_html=True)

    if start and uploaded is not None:
        scroll_to_process()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{safe_stem(uploaded.name)}_{timestamp}"
        input_path = UPLOAD_DIR / f"{stem}{Path(uploaded.name).suffix.lower() or '.mp4'}"
        output_path = OUTPUT_DIR / f"{stem}_censored.mp4"
        input_path.write_bytes(uploaded.getbuffer())

        start_time = time.perf_counter()
        st.session_state["processing_started_at"] = float(start_time)
        st.session_state["progress"] = 0.0
        st.session_state["status"] = "Starting"
        render_timing_box(elapsed_box, 0.0, start_time, "Starting")

        def progress_callback(value: float) -> None:
            st.session_state["progress"] = float(value)
            progress_box.progress(float(value))
            render_timing_box(
                elapsed_box,
                float(value),
                st.session_state.get("processing_started_at"),
                str(st.session_state["status"]),
            )
            with hero.container():
                render_hero(float(value), str(st.session_state["status"]), keep_audio, unsafe_threshold)

        def status_callback(message: str) -> None:
            st.session_state["status"] = message
            status_box.info(message)
            render_timing_box(
                elapsed_box,
                float(st.session_state["progress"]),
                st.session_state.get("processing_started_at"),
                message,
            )
            with hero.container():
                render_hero(float(st.session_state["progress"]), message, keep_audio, unsafe_threshold)

        try:
            result = run_censorship_pipeline(
                input_path,
                output_path,
                device=device,
                mode=mode,
                pixelation_factor=pixelation_factor,
                blur_kernel=blur_kernel,
                unsafe_threshold=unsafe_threshold,
                preserve_audio=keep_audio,
                progress_callback=progress_callback,
                status_callback=status_callback,
            )
            elapsed = time.perf_counter() - start_time
            result["summary"]["processing_time_seconds"] = float(elapsed)
            summary_path = Path(result["paths"]["summary"])
            summary_path.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
            st.session_state["last_result"] = result
            st.session_state["last_input"] = str(input_path)
            st.session_state["last_output"] = str(result["paths"]["video"])
            st.session_state["status"] = "Complete"
            st.session_state["progress"] = 1.0
            progress_box.progress(1.0)
            status_box.success("Complete")
            render_timing_box(elapsed_box, 1.0, start_time, "Complete")
            with hero.container():
                render_hero(1.0, "Complete", keep_audio, unsafe_threshold)
        except FileNotFoundError as exc:
            st.error(f"Missing file or checkpoint: {exc}")
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            st.exception(exc)

    result = st.session_state.get("last_result")
    st.markdown('<section class="section" id="results">', unsafe_allow_html=True)
    section_heading(
        "Results",
        "Review the censored output.",
    )
    if result:
        summary = result["summary"]
        segments = result.get("segments", [])
        metric_cols = st.columns(6, gap="medium")
        applied_style = str(summary.get("censor_mode", "pixelate")).title()
        if str(summary.get("censor_mode", "pixelate")).lower() == "pixelate":
            applied_strength = f"Pixelate {int(summary.get('pixelation_factor', censorship_config.PIXELATION_FACTOR))}"
        else:
            applied_strength = (
                f"Blur {int(summary.get('blur_kernel', censorship_config.BLUR_KERNEL))} "
                f"x{int(summary.get('blur_passes', getattr(censorship_config, 'BLUR_PASSES', 1)))}"
            )
        metrics = [
            ("Unsafe segments", str(summary.get("num_unsafe_segments", 0))),
            ("Censored duration", f"{float(summary.get('total_unsafe_duration', 0.0)):.1f}s"),
            ("Max unsafe probability", f"{float(summary.get('max_temporal_unsafe_prob', 0.0)):.2f}"),
            ("Applied style", f"{applied_style} · {applied_strength}"),
            ("Processing time", f"{float(summary.get('processing_time_seconds', 0.0)):.1f}s"),
            ("Audio preserved", audio_status(summary)),
        ]
        for col, (label, value) in zip(metric_cols, metrics):
            with col:
                metric_card(label, value)

        video_cols = st.columns(2, gap="large")
        input_video_path = Path(st.session_state["last_input"])
        output_video_path = Path(st.session_state["last_output"])
        with video_cols[0]:
            render_video(input_video_path, "Original video")
        with video_cols[1]:
            render_video(
                output_video_path,
                "Censored video",
                browser_warning=not bool(summary.get("browser_compatible_video", True)),
            )

        st.markdown('<div class="agency-card"><h3 class="card-title">Unsafe segments</h3></div>', unsafe_allow_html=True)
        if segments:
            display_rows = [
                {
                    "start": row.get("start"),
                    "end": row.get("end"),
                    "duration": float(row.get("end", 0.0)) - float(row.get("start", 0.0)),
                    "max_temporal_unsafe_prob": row.get("max_temporal_unsafe_prob"),
                    "max_fused_score": row.get("max_fused_score"),
                    "mean_fused_score": row.get("mean_fused_score"),
                }
                for row in segments
            ]
            st.dataframe(display_rows, use_container_width=True)
        else:
            st.success("No unsafe segments were detected.")

        summary_path = Path(result["paths"]["summary"])
        download_cols = st.columns(2, gap="large")
        with download_cols[0]:
            if output_video_path.exists():
                st.download_button(
                    "Download censored video",
                    data=output_video_path.read_bytes(),
                    file_name=output_video_path.name,
                    mime="video/mp4",
                    use_container_width=True,
                )
            else:
                st.error("Censored video download is unavailable because the output file is missing.")
        with download_cols[1]:
            if summary_path.exists():
                st.download_button(
                    "Download summary JSON",
                    data=summary_path.read_text(encoding="utf-8"),
                    file_name=summary_path.name,
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.error("Summary download is unavailable because the JSON file is missing.")
    else:
        st.markdown(
            '<div class="agency-card"><h3 class="card-title">No run yet</h3><p class="card-copy">Upload a video and start censorship to populate this case-study style results area.</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</section>", unsafe_allow_html=True)

    st.markdown(
        """
        <footer class="footer-note">
          CensorAI is a local project demo for safer media review. The default mode is temporal-only
          full-frame censorship during detected unsafe segments. It does not claim precise nude-region
          segmentation, and experimental localization paths are intentionally hidden from this interface.
        </footer>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
