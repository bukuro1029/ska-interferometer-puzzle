"""Browser-friendly exhibit edition of the SKA Interferometer Puzzle.

Run locally with:
    streamlit run online_app.py

This app deliberately uses manual layouts rather than UDP. Public Streamlit
hosting cannot receive packets directly from a local Raspberry Pi, while the
local Pygame application remains the fast UDP display path.
"""

from __future__ import annotations

import math
from io import BytesIO
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Ellipse
from PIL import Image, ImageOps

from realtime_display import (
    EXHIBIT_THEMES,
    SAMPLE_MODELS,
    blur_array,
    clean_style_products,
    dirty_image_from_uv,
    exhibit_scalar_rgb,
    make_baseline_envelope,
    make_sample_sky,
    make_uv_weight,
    render_exhibit_reconstruction_rgb,
)


st.set_page_config(
    page_title="SKA干渉計パズル オンライン展示版",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        display: none;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] div::before {
        content: "画像をここにドラッグ＆ドロップ";
        font-size: 0.9rem;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        display: none;
    }
    [data-testid="stFileUploaderDropzoneInstructions"]::after {
        content: "PNG・JPG（1ファイル200MBまで）";
        display: block;
        color: #9ca3af;
        font-size: 0.75rem;
        margin-top: 0.2rem;
    }
    [data-testid="stFileUploaderDropzone"] button {
        font-size: 0;
    }
    [data-testid="stFileUploaderDropzone"] button > * {
        display: none !important;
    }
    [data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"],
    [data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"] {
        display: none !important;
    }
    [data-testid="stFileUploaderDropzone"] button::after {
        content: "画像を選択";
        font-size: 0.875rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


SAMPLE_LABELS: Dict[str, str] = {
    "gas": "星雲",
    "points": "点状天体",
    "bubbles": "泡状構造",
    "ska": "SKAの文字",
    "double": "二重天体",
    "jet": "ジェット",
    "ring": "リング",
    "spiral": "渦巻銀河",
    "cluster": "銀河団",
    "cross": "十字状電波源",
    "crescent": "三日月状構造",
    "resolution": "解像度テスト",
}

LAYOUT_LABELS: Dict[str, str] = {
    "core": "中心集中型",
    "line": "直線型",
    "random": "ランダム型",
    "arms": "三本腕型",
    "ska": "SKA型スパイラル",
}

STYLE_LABELS: Dict[str, str] = {
    "exhibit": "展示表示",
    "clean": "CLEAN表示",
    "eht": "EHT風表示",
    "residual": "残差表示",
}

THEME_LABELS: Dict[str, str] = {
    "aurora": "オーロラ",
    "ember": "炎",
    "tide": "海",
    "violet": "紫",
    "mint": "ミント",
    "mono": "モノクロ",
    "coral": "コーラル",
}


def clip_to_radius(pos: np.ndarray, radius: float) -> np.ndarray:
    radius_from_center = np.hypot(pos[:, 0], pos[:, 1])
    outside = radius_from_center > radius
    if np.any(outside):
        pos[outside] *= (radius / radius_from_center[outside])[:, None]
    return pos


def make_layout(kind: str, count: int, radius: float, seed: int) -> np.ndarray:
    """Create clear, deterministic layouts for the public online explorer."""
    rng = np.random.default_rng(seed)

    if kind == "core":
        return clip_to_radius(rng.normal(0.0, 0.19 * radius, size=(count, 2)), radius)

    if kind == "line":
        x = np.linspace(-radius, radius, count)
        y = rng.normal(0.0, 0.025 * radius, count)
        return np.column_stack((x, y))

    if kind == "random":
        theta = rng.uniform(0.0, 2.0 * math.pi, count)
        distance = radius * np.sqrt(rng.uniform(0.0, 1.0, count))
        return np.column_stack((distance * np.cos(theta), distance * np.sin(theta)))

    if kind == "arms":
        arms = np.arange(count) % 3
        distance = radius * (0.08 + 0.92 * np.linspace(0.0, 1.0, count))
        rng.shuffle(distance)
        theta = arms * 2.0 * math.pi / 3.0 + 0.62 * distance / radius + rng.normal(0.0, 0.08, count)
        return clip_to_radius(np.column_stack((distance * np.cos(theta), distance * np.sin(theta))), radius)

    core_count = max(5, int(0.48 * count))
    core = rng.normal(0.0, 0.11 * radius, size=(core_count, 2))
    arm_count = count - core_count
    arms = np.arange(arm_count) % 3
    distance = radius * (0.18 + 0.82 * rng.random(arm_count))
    theta = arms * 2.0 * math.pi / 3.0 + 0.75 * distance / radius + rng.normal(0.0, 0.10, arm_count)
    arm_positions = np.column_stack((distance * np.cos(theta), distance * np.sin(theta)))
    return clip_to_radius(np.vstack((core, arm_positions)), radius)


def uploaded_image_to_sky(upload: BytesIO, grid: int) -> np.ndarray:
    image = Image.open(upload).convert("L")
    image = ImageOps.contain(image, (grid, grid), method=Image.Resampling.LANCZOS)
    canvas = Image.new("L", (grid, grid), color=0)
    canvas.paste(image, ((grid - image.width) // 2, (grid - image.height) // 2))
    sky = np.asarray(canvas, dtype=float) / 255.0
    return sky / max(float(sky.max()), 1e-12)


@st.cache_data(show_spinner=False)
def calculate_reconstruction(
    sample: str,
    layout: str,
    antenna_count: int,
    max_baseline: float,
    layout_seed: int,
    grid: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sky = make_sample_sky(grid, model=sample)
    return calculate_from_sky(sky, layout, antenna_count, max_baseline, layout_seed)


def calculate_from_sky(
    sky: np.ndarray,
    layout: str,
    antenna_count: int,
    max_baseline: float,
    layout_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = sky.shape[0]
    reference_baseline = 150.0
    positions = make_layout(layout, antenna_count, max_baseline / 2.0, layout_seed)
    sparse_uv = make_uv_weight(positions, grid, reference_baseline, 1.0, 2)
    envelope = make_baseline_envelope(grid, max_baseline, reference_baseline, 1.0)
    display_uv = envelope * np.clip(0.55 * sparse_uv + 0.45 * blur_array(sparse_uv, passes=6), 0.0, 1.0)
    dirty = dirty_image_from_uv(sky, display_uv)
    reconstruction, residual = clean_style_products(dirty, display_uv)
    return positions, sparse_uv, display_uv, dirty, reconstruction, residual


def layout_figure(positions: np.ndarray, radius: float) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
    figure.patch.set_facecolor("#07111f")
    axis.set_facecolor("#07111f")
    axis.scatter(positions[:, 0], positions[:, 1], s=32, c="#f6c545", edgecolors="#fff1bd", linewidths=0.5)
    axis.set_xlim(-radius * 1.08, radius * 1.08)
    axis.set_ylim(-radius * 1.08, radius * 1.08)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#3b4b5b")
    return figure


def reconstruction_figure(
    rgb: np.ndarray,
    reconstruction: np.ndarray,
    style: str,
    uv: np.ndarray,
) -> plt.Figure:
    """Render the large image panel, adding conventional CLEAN annotations."""
    grid = rgb.shape[0]
    figure, axis = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    figure.patch.set_facecolor("#07111f")
    axis.set_facecolor("#07111f")
    axis.imshow(rgb, origin="lower", interpolation="lanczos")

    if style == "clean":
        scale = float(np.percentile(np.maximum(reconstruction, 0.0), 99.5))
        if scale > 1e-12:
            levels = [scale * value for value in (0.20, 0.38, 0.58, 0.80)]
            axis.contour(reconstruction, levels=levels, colors="#d5e3e5", linewidths=0.7, origin="lower")
        weight = np.maximum(uv, 0.0)
        y, x = np.mgrid[0 : weight.shape[0], 0 : weight.shape[1]]
        total = float(weight.sum())
        if total > 1e-12:
            x = x - (weight.shape[1] - 1) / 2.0
            y = y - (weight.shape[0] - 1) / 2.0
            covariance = np.array(
                [
                    [float((weight * x * x).sum() / total), float((weight * x * y).sum() / total)],
                    [float((weight * x * y).sum() / total), float((weight * y * y).sum() / total)],
                ]
            )
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-9)
            aspect = float(np.clip(math.sqrt(eigenvalues[1] / eigenvalues[0]), 1.0, 2.5))
            direction = eigenvectors[:, 0]
            angle = math.degrees(math.atan2(direction[1], direction[0]))
            beam = Ellipse(
                (0.13 * grid, 0.13 * grid),
                width=18.0 * math.sqrt(aspect),
                height=18.0 / math.sqrt(aspect),
                angle=angle,
                facecolor="#d2ded9",
                edgecolor="#142025",
                linewidth=0.8,
            )
            axis.add_patch(beam)

    axis.set_axis_off()
    return figure


st.title("SKA干渉計パズル")
st.caption("アンテナのならびを変えて、元の天体画像がどのように再構成されるかを体験します。")

with st.sidebar:
    st.header("展示の設定")
    sample = st.selectbox("元の天体画像", SAMPLE_MODELS, format_func=lambda value: SAMPLE_LABELS[value])
    uploaded_image = st.file_uploader("任意の画像を使う", type=["png", "jpg", "jpeg"])
    layout = st.selectbox("アンテナのならび", list(LAYOUT_LABELS), format_func=lambda value: LAYOUT_LABELS[value], index=4)
    antenna_count = st.slider("アンテナ数", min_value=6, max_value=64, value=16, step=1)
    max_baseline = st.slider("最大基線", min_value=20, max_value=150, value=74, step=1)
    layout_seed = st.slider("ならびの変化", min_value=0, max_value=20, value=0, step=1)
    reconstruction_style = st.selectbox("再構成画像の表示", list(STYLE_LABELS), format_func=lambda value: STYLE_LABELS[value], index=0)
    theme = st.selectbox("展示カラー", EXHIBIT_THEMES, format_func=lambda value: THEME_LABELS[value], index=0)
    st.divider()
    st.caption("このオンライン版はブラウザ操作用です。Raspberry PiからのUDP受信と高速表示にはローカルPygame版を使用します。")

grid = 192
if uploaded_image is None:
    sky = make_sample_sky(grid, model=sample)
    image_label = SAMPLE_LABELS[sample]
    positions, sparse_uv, display_uv, dirty, reconstruction, residual = calculate_reconstruction(
        sample, layout, antenna_count, float(max_baseline), layout_seed, grid
    )
else:
    sky = uploaded_image_to_sky(uploaded_image, grid)
    image_label = uploaded_image.name
    positions, sparse_uv, display_uv, dirty, reconstruction, residual = calculate_from_sky(
        sky, layout, antenna_count, float(max_baseline), layout_seed
    )

reconstruction_rgb = render_exhibit_reconstruction_rgb(
    reconstruction,
    residual,
    reconstruction_style,
    theme,
    contrast_percentile=99.0,
    stretch=4.0,
)
source_rgb = exhibit_scalar_rgb(sky, "sky", theme)
baseline_count = antenna_count * (antenna_count - 1) // 2

metrics = st.columns(3)
metrics[0].metric("アンテナ", antenna_count)
metrics[1].metric("基線", baseline_count)
metrics[2].metric("再構成表示", STYLE_LABELS[reconstruction_style])

left, right = st.columns((1.45, 1.0), gap="large")
with left:
    st.subheader(f"再構成した画像: {STYLE_LABELS[reconstruction_style]}")
    st.pyplot(
        reconstruction_figure(reconstruction_rgb, reconstruction, reconstruction_style, sparse_uv),
        width="stretch",
    )
    if reconstruction_style == "eht":
        st.caption("EHT風表示は見せ方のみを変える機能で、観測情報や解像度を増やすものではありません。")
    elif reconstruction_style == "residual":
        st.caption("赤と青は、CLEAN後に残った正負の成分を表します。")

with right:
    top_left, top_right = st.columns(2)
    with top_left:
        st.caption("アンテナのならび")
        st.pyplot(layout_figure(positions, float(max_baseline) / 2.0), width="stretch")
    with top_right:
        st.markdown("#### 3つの表示の見方")
        st.markdown(
            """
            **再構成した画像**

            選んだアンテナ配置で、天体がどこまで再現できたかを示します。

            **アンテナのならび**

            観測に使うアンテナの位置です。配置や台数によって、画像の細かさや形が変わります。

            **元の画像**

            再構成前の天体画像です。再構成した画像と見比べてください。
            """
        )
    st.caption(f"元の画像: {image_label}")
    st.image(source_rgb, width="stretch")

with st.expander("この表示について"):
    st.write(
        "これは電波干渉計の基本を体験するための簡易シミュレーターです。アンテナ同士の組合せから得られる観測情報を使い、"
        "天体画像を簡易的に再構成しています。実際の観測解析では、較正やより高度な画像再構成処理も行います。"
    )
