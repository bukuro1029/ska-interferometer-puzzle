"""
SKA Interferometer Puzzle — Streamlit app

Run:
    pip install streamlit numpy matplotlib pandas pillow
    streamlit run app.py

Purpose:
    Educational outreach app showing how antenna layout and telescope specifications
    affect uv coverage and a simplified dirty image.

Important:
    This is an outreach-oriented educational simulator, not a precision radio
    interferometric imaging pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps


# ============================================================
# Streamlit page config
# ============================================================

st.set_page_config(
    page_title="SKA Interferometer Puzzle",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Presets
# ============================================================

TELESCOPE_PRESETS: Dict[str, Dict[str, object]] = {
    "Educational mini array": {
        "interferometric_elements": 16,
        "physical_antennas": 16,
        "max_baseline": 20.0,
        "unit": "km",
        "display_cap": 32,
        "note": "A small virtual array for outreach. These are not real instrument specifications.",
    },
    "SKA-Low scale": {
        "interferometric_elements": 512,
        "physical_antennas": 131_072,
        "max_baseline": 74.0,
        "unit": "km",
        "display_cap": 96,
        "note": (
            "Low-frequency mode. 131,072 physical antennas are grouped into 512 stations. "
            "The station centers are used as interferometric elements."
        ),
    },
    "SKA-Mid scale": {
        "interferometric_elements": 197,
        "physical_antennas": 197,
        "max_baseline": 150.0,
        "unit": "km",
        "display_cap": 96,
        "note": "Mid-frequency mode. 197 dishes are used as interferometric elements.",
    },
    "Custom": {
        "interferometric_elements": 64,
        "physical_antennas": 64,
        "max_baseline": 40.0,
        "unit": "km",
        "display_cap": 96,
        "note": "Enter any telescope specification you want.",
    },
}


# ============================================================
# Utility functions
# ============================================================

def robust_normalize(img: np.ndarray, symmetric: bool = False) -> np.ndarray:
    arr = np.asarray(img, dtype=float)

    if symmetric:
        vmax = np.percentile(np.abs(arr), 99.0)
        if vmax <= 0:
            return np.zeros_like(arr)
        return np.clip(arr / vmax, -1.0, 1.0)

    lo, hi = np.percentile(arr, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def blur_array(arr: np.ndarray, passes: int = 2) -> np.ndarray:
    """
    Simple dependency-free blur for educational uv filling display.
    This is not a physical gridding kernel.
    """
    out = np.asarray(arr, dtype=float)

    for _ in range(max(0, passes)):
        out = (
            4.0 * out
            + 2.0 * (
                np.roll(out, 1, axis=0)
                + np.roll(out, -1, axis=0)
                + np.roll(out, 1, axis=1)
                + np.roll(out, -1, axis=1)
            )
            + (
                np.roll(np.roll(out, 1, axis=0), 1, axis=1)
                + np.roll(np.roll(out, 1, axis=0), -1, axis=1)
                + np.roll(np.roll(out, -1, axis=0), 1, axis=1)
                + np.roll(np.roll(out, -1, axis=0), -1, axis=1)
            )
        ) / 16.0

    if out.max() > 0:
        out = out / out.max()
    return out


def uv_count_factor(elements: int, reference_elements: int, strength: float = 1.0) -> float:
    """
    Convert interferometric element count to a 0-1 factor for uv filling.
    Uses nonlinear compressed scaling for outreach.
    """
    n = max(float(elements), 2.0)
    ref = max(float(reference_elements), 2.0)

    x = np.log10(n / ref + 1.0)
    response = 1.0 - np.exp(-strength * x)
    return float(np.clip(response, 0.0, 1.0))


def effective_noise_from_count(
    elements: int,
    reference_elements: int,
    base_noise: float,
    floor_fraction: float = 0.15,
    strength: float = 1.5,
) -> Tuple[float, float]:
    """
    Convert physical antenna count to effective noise using nonlinear scaling + noise floor.

    Returns:
        effective_noise, sensitivity_response
    """
    n = max(float(elements), 2.0)
    ref = max(float(reference_elements), 2.0)

    x = np.log10(n / ref + 1.0)
    response = 1.0 - np.exp(-strength * x)

    noise_floor = base_noise * floor_fraction
    effective_noise = base_noise - (base_noise - noise_floor) * response

    return float(effective_noise), float(np.clip(response, 0.0, 1.0))


def gaussian_2d(
    n: int,
    x0: float,
    y0: float,
    sx: float,
    sy: float,
    amp: float = 1.0,
    theta: float = 0.0,
) -> np.ndarray:
    y, x = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
    ct = np.cos(theta)
    st_ = np.sin(theta)
    xp = ct * (x - x0) + st_ * (y - y0)
    yp = -st_ * (x - x0) + ct * (y - y0)
    return amp * np.exp(-0.5 * ((xp / sx) ** 2 + (yp / sy) ** 2))


def make_sky(n: int, model: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sky = np.zeros((n, n), dtype=float)

    if model == "Extended hydrogen gas + small galaxies":
        sky += gaussian_2d(n, -0.18, 0.08, 0.38, 0.21, amp=1.0, theta=0.7)
        sky += gaussian_2d(n, 0.34, -0.28, 0.18, 0.10, amp=0.55, theta=-0.4)
        for _ in range(10):
            sky += gaussian_2d(
                n,
                rng.uniform(-0.82, 0.82),
                rng.uniform(-0.82, 0.82),
                rng.uniform(0.010, 0.030),
                rng.uniform(0.010, 0.030),
                amp=rng.uniform(0.25, 0.85),
            )

    elif model == "Many point sources":
        for _ in range(26):
            sky += gaussian_2d(
                n,
                rng.uniform(-0.88, 0.88),
                rng.uniform(-0.88, 0.88),
                rng.uniform(0.008, 0.022),
                rng.uniform(0.008, 0.022),
                amp=rng.uniform(0.25, 1.25),
            )

    elif model == "Bubble-like cosmic structure":
        sky += 0.50 * gaussian_2d(n, 0.0, 0.0, 0.82, 0.82, amp=1.0)
        for _ in range(14):
            r = rng.uniform(0.07, 0.22)
            sky -= gaussian_2d(
                n,
                rng.uniform(-0.82, 0.82),
                rng.uniform(-0.82, 0.82),
                r,
                r,
                amp=rng.uniform(0.18, 0.72),
            )
        sky -= sky.min()

    elif model == "SKA letters":
        sky += gaussian_2d(n, -0.55, 0.45, 0.16, 0.08, amp=0.9)
        sky += gaussian_2d(n, -0.60, 0.15, 0.10, 0.08, amp=0.9)
        sky += gaussian_2d(n, -0.50, -0.15, 0.10, 0.08, amp=0.9)
        sky += gaussian_2d(n, -0.60, -0.45, 0.16, 0.08, amp=0.9)

        for t in np.linspace(-0.55, 0.55, 12):
            sky += gaussian_2d(n, -0.05, t, 0.02, 0.02, amp=0.9)
        for t in np.linspace(-0.45, 0.45, 12):
            sky += gaussian_2d(n, 0.12 + 0.25 * abs(t), t, 0.02, 0.02, amp=0.9)

        for t in np.linspace(-0.5, 0.5, 18):
            y = -0.5 + t
            sky += gaussian_2d(n, 0.55 - 0.25 * t, y, 0.02, 0.02, amp=0.9)
            sky += gaussian_2d(n, 0.55 + 0.25 * t, y, 0.02, 0.02, amp=0.9)
        for x in np.linspace(0.40, 0.70, 12):
            sky += gaussian_2d(n, x, 0.0, 0.02, 0.02, amp=0.9)

    elif model == "Large disk + small points":
        sky += gaussian_2d(n, 0.0, 0.0, 0.42, 0.42, amp=0.8)
        for _ in range(8):
            sky += gaussian_2d(
                n,
                rng.uniform(-0.80, 0.80),
                rng.uniform(-0.80, 0.80),
                0.015,
                0.015,
                amp=rng.uniform(0.5, 1.2),
            )

    else:
        raise ValueError(f"Unknown sky model: {model}")

    return robust_normalize(sky, symmetric=False)


def apply_manual_rotation(img: Image.Image, rotation_label: str) -> Image.Image:
    if rotation_label == "0° (no rotation)":
        return img
    if rotation_label == "90° clockwise":
        return img.rotate(-90, expand=True)
    if rotation_label == "180°":
        return img.rotate(180, expand=True)
    if rotation_label == "90° counterclockwise":
        return img.rotate(90, expand=True)
    return img


def image_to_square_canvas(img: Image.Image, n: int, keep_aspect: bool = True) -> Image.Image:
    img = img.convert("L")

    if not keep_aspect:
        return img.resize((n, n), resample=Image.Resampling.LANCZOS)

    img.thumbnail((n, n), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("L", (n, n), color=0)
    x = (n - img.size[0]) // 2
    y = (n - img.size[1]) // 2
    canvas.paste(img, (x, y))
    return canvas


def load_uploaded_sky(
    uploaded_file,
    n: int,
    invert: bool = False,
    threshold: float = 0.0,
    gamma: float = 1.0,
    manual_rotation: str = "0° (no rotation)",
    keep_aspect: bool = True,
) -> np.ndarray:
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)
    img = apply_manual_rotation(img, manual_rotation)
    img = image_to_square_canvas(img, n=n, keep_aspect=keep_aspect)

    arr = np.asarray(img, dtype=float)

    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()

    if invert:
        arr = 1.0 - arr

    if threshold > 0:
        arr[arr < threshold] = 0.0

    gamma = max(gamma, 1e-6)
    arr = arr ** gamma

    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()

    return arr


# ============================================================
# Layout generation
# ============================================================

def clip_to_radius(pos: np.ndarray, radius: float) -> np.ndarray:
    r = np.hypot(pos[:, 0], pos[:, 1])
    mask = r > radius
    if np.any(mask):
        pos[mask] *= (radius / r[mask])[:, None]
    return pos


def make_layout(kind: str, n: int, radius: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if kind == "Core concentrated":
        return clip_to_radius(rng.normal(0, 0.18 * radius, size=(n, 2)), radius)

    if kind == "Long-baseline biased":
        n_core = max(3, n // 3)
        core = rng.normal(0, 0.12 * radius, size=(n_core, 2))
        n_outer = n - n_core
        th = rng.uniform(0, 2 * np.pi, n_outer)
        rr = rng.uniform(0.65 * radius, radius, n_outer)
        outer = np.column_stack([rr * np.cos(th), rr * np.sin(th)])
        return np.vstack([core, outer])

    if kind == "Linear":
        x = np.linspace(-radius, radius, n)
        y = rng.normal(0, 0.025 * radius, n)
        return np.column_stack([x, y])

    if kind == "Random":
        th = rng.uniform(0, 2 * np.pi, n)
        rr = radius * np.sqrt(rng.uniform(0, 1, n))
        return np.column_stack([rr * np.cos(th), rr * np.sin(th)])

    if kind == "Three-arm":
        arms = np.arange(n) % 3
        base = arms * 2 * np.pi / 3
        rr = radius * (0.08 + 0.92 * np.linspace(0, 1, n))
        rng.shuffle(rr)
        th = base + 0.60 * rr / max(radius, 1e-12) + rng.normal(0, 0.08, n)
        pos = np.column_stack([rr * np.cos(th), rr * np.sin(th)])
        pos += rng.normal(0, 0.015 * radius, size=pos.shape)
        return clip_to_radius(pos, radius)

    if kind == "SKA-like balanced":
        n_core = max(5, int(0.48 * n))
        core = rng.normal(0, 0.11 * radius, size=(n_core, 2))
        n_arm = n - n_core
        arms = np.arange(n_arm) % 3
        base = arms * 2 * np.pi / 3
        rr = radius * (0.18 + 0.82 * rng.random(n_arm))
        th = base + 0.75 * rr / max(radius, 1e-12) + rng.normal(0, 0.10, n_arm)
        arm = np.column_stack([rr * np.cos(th), rr * np.sin(th)])
        return clip_to_radius(np.vstack([core, arm]), radius)

    raise ValueError(f"Unknown layout: {kind}")


def baselines(pos: np.ndarray) -> np.ndarray:
    out: List[Tuple[float, float]] = []
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = pos[j] - pos[i]
            out.append((dx, dy))
            out.append((-dx, -dy))
    return np.asarray(out, dtype=float)


# ============================================================
# uv coverage and dirty image
# ============================================================

def add_disk(arr: np.ndarray, cx: int, cy: int, r: int, value: float = 1.0) -> None:
    n = arr.shape[0]
    x0, x1 = max(0, cx - r), min(n, cx + r + 1)
    y0, y1 = max(0, cy - r), min(n, cy + r + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    arr[y0:y1, x0:x1][disk] += value


def make_sparse_uv_weight(
    pos: np.ndarray,
    grid: int,
    reference_baseline: float,
    uv_zoom: float,
    point_radius: int,
    density_bonus: int,
    include_zero_spacing_hint: bool,
) -> Tuple[np.ndarray, float]:
    bl = baselines(pos)
    weight = np.zeros((grid, grid), dtype=float)
    c = grid // 2
    scale = (0.45 * grid * uv_zoom) / max(reference_baseline, 1e-12)
    r_pix = max(1, int(point_radius + density_bonus))
    outside = 0

    for u, v in bl:
        ix = int(round(c + u * scale))
        iy = int(round(c + v * scale))
        if 0 <= ix < grid and 0 <= iy < grid:
            add_disk(weight, ix, iy, r_pix, 1.0)
        else:
            outside += 1

    if include_zero_spacing_hint:
        add_disk(weight, c, c, max(1, r_pix), 1.0)

    if weight.max() > 0:
        weight = np.sqrt(weight)
        weight /= weight.max()

    return weight, outside / max(len(bl), 1)


def make_baseline_envelope(
    grid: int,
    max_baseline: float,
    reference_baseline: float,
    uv_zoom: float,
    include_zero_spacing_hint: bool,
) -> np.ndarray:
    """
    Educational Fourier envelope.
    Larger max_baseline -> higher spatial frequency can pass -> sharper image.
    """
    y, x = np.mgrid[0:grid, 0:grid]
    c = grid // 2
    rho = np.hypot(x - c, y - c)

    rmax = 0.45 * grid * uv_zoom * max_baseline / max(reference_baseline, 1e-12)
    rmax = np.clip(rmax, 1.0, 0.49 * grid)

    envelope = np.exp(-((rho / rmax) ** 8))

    if not include_zero_spacing_hint:
        inner = np.exp(-((rho / max(2.0, 0.025 * grid)) ** 4))
        envelope = envelope * (1.0 - inner)

    return envelope


def reconstruct_dirty_image(
    sky: np.ndarray,
    sparse_uv: np.ndarray,
    envelope: np.ndarray,
    educational_mode: bool,
    base_noise: float,
    sensitivity_elements: int,
    sensitivity_reference: int,
    interferometric_elements: int,
    coverage_reference_elements: int,
    count_effect_strength: float,
    noise_floor_fraction: float,
    sensitivity_effect_strength: float,
    signal_strength: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """
    Returns:
        dirty, effective_uv, effective_noise, count_factor, sensitivity_response, snr_proxy
    """
    rng = np.random.default_rng(seed)

    # Reference sky used to set the noise scale
    sky_ref = sky - np.mean(sky)
    ft_ref = np.fft.fftshift(np.fft.fft2(sky_ref))

    # Actual astronomical signal entering the telescope
    ft_signal = signal_strength * ft_ref

    count_factor = uv_count_factor(
        elements=interferometric_elements,
        reference_elements=coverage_reference_elements,
        strength=count_effect_strength,
    )

    effective_noise, sensitivity_response = effective_noise_from_count(
        elements=sensitivity_elements,
        reference_elements=sensitivity_reference,
        base_noise=base_noise,
        floor_fraction=noise_floor_fraction,
        strength=sensitivity_effect_strength,
    )

    alpha = np.clip(count_factor, 0.0, 1.0)

    if educational_mode:
        smooth_uv = blur_array(sparse_uv, passes=1 + int(6 * alpha))
        filled_uv = (1.0 - alpha) * sparse_uv + alpha * smooth_uv
        effective_uv = envelope * np.clip(filled_uv, 0.0, 1.0)
    else:
        effective_uv = envelope * sparse_uv

    sampled = ft_signal * effective_uv

    # Noise is based on the unscaled reference sky,
    # so lowering signal_strength lowers the signal only.
    if effective_noise > 0:
        amp = np.percentile(np.abs(ft_ref), 95) * effective_noise
        noise = amp * (rng.normal(size=ft_ref.shape) + 1j * rng.normal(size=ft_ref.shape))
        sampled += noise * effective_uv

    dirty = np.real(np.fft.ifft2(np.fft.ifftshift(sampled)))
    dirty -= np.mean(dirty)

    signal_rms = float(np.std(signal_strength * sky_ref))
    noise_rms_proxy = float(effective_noise * np.std(sky_ref))
    snr_proxy = signal_rms / max(noise_rms_proxy, 1e-9)

    return (
        dirty,
        effective_uv,
        float(effective_noise),
        float(count_factor),
        float(sensitivity_response),
        float(snr_proxy),
    )


# ============================================================
# Scoring
# ============================================================

@dataclass
class Scores:
    resolution: float
    sensitivity: float
    extended: float
    artifact_control: float
    total: float
    comment: str


def entropy01(p: np.ndarray) -> float:
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))


def compute_scores(
    pos: np.ndarray,
    radius: float,
    mission: str,
    sensitivity_elements: int,
    sensitivity_reference: int,
) -> Scores:
    bl = baselines(pos)
    length = np.hypot(bl[:, 0], bl[:, 1])
    max_possible = max(2 * radius, 1e-12)

    resolution = np.clip(100 * np.max(length) / max_possible, 0, 100)

    sens_ratio = np.sqrt(max(sensitivity_elements, 2) / max(sensitivity_reference, 2))
    sensitivity = np.clip(100 * sens_ratio, 0, 100)

    extended = np.clip(100 * np.mean(length < 0.35 * radius) / 0.45, 0, 100)

    angle = np.mod(np.arctan2(bl[:, 1], bl[:, 0]), np.pi)
    hist, _ = np.histogram(angle, bins=12, range=(0, np.pi))
    angle_score = entropy01(hist / max(hist.sum(), 1))
    radial_balance = 1.0 - min(1.0, abs(np.median(length) - 0.65 * radius) / (0.65 * radius))
    artifact_control = np.clip(100 * (0.75 * angle_score + 0.25 * radial_balance), 0, 100)

    weights = {
        "Resolve distant galaxies": (0.50, 0.20, 0.05, 0.25),
        "Observe extended hydrogen gas": (0.15, 0.20, 0.45, 0.20),
        "Detect faint radio sources": (0.10, 0.50, 0.10, 0.30),
        "Reduce fake patterns": (0.15, 0.20, 0.15, 0.50),
    }[mission]

    total = (
        weights[0] * resolution
        + weights[1] * sensitivity
        + weights[2] * extended
        + weights[3] * artifact_control
    )

    weak = min(
        [
            (resolution, "Weak at resolving fine structure. Increase the maximum baseline."),
            (extended, "Weak at capturing extended structure. Add more short baselines in the core."),
            (artifact_control, "The layout is anisotropic, so fake patterns are likely. Spread directions more evenly."),
            (sensitivity, "Sensitivity to faint signals is insufficient. Increase the number of physical antennas."),
        ],
        key=lambda x: x[0],
    )

    comment = weak[1]
    if total >= 82:
        comment = "A well-balanced array for this mission."
    elif total >= 65:
        comment = "A reasonable configuration, but there is still a clear weakness. " + comment

    return Scores(
        float(resolution),
        float(sensitivity),
        float(extended),
        float(artifact_control),
        float(total),
        comment,
    )


# ============================================================
# Plotting
# ============================================================

def fig_layout(pos: np.ndarray, radius: float, unit: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    circle = plt.Circle((0, 0), radius, fill=False, linestyle="--", linewidth=1.2)
    ax.add_patch(circle)
    ax.scatter(pos[:, 0], pos[:, 1], s=38)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.08 * radius, 1.08 * radius)
    ax.set_ylim(-1.08 * radius, 1.08 * radius)
    ax.set_title("Antenna / station layout")
    ax.set_xlabel(f"East-West ({unit})")
    ax.set_ylabel(f"North-South ({unit})")
    ax.grid(alpha=0.25)
    return fig


def fig_uv(weight: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(weight, origin="lower", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def fig_sky(sky: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(sky, origin="upper", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Input image / true structure")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def fig_dirty(dirty: np.ndarray, display_mode: str, fixed_vmax: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    if display_mode == "Auto contrast (show shape clearly)":
        img = robust_normalize(dirty, symmetric=True)
        ax.imshow(img, origin="upper", interpolation="nearest", vmin=-1, vmax=1)
    else:
        ax.imshow(
            dirty,
            origin="upper",
            interpolation="nearest",
            vmin=-fixed_vmax,
            vmax=fixed_vmax,
        )
    ax.set_title("Reconstructed image (dirty image)")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def metric_bar(label: str, value: float) -> None:
    st.write(f"**{label}: {value:.0f}/100**")
    st.progress(int(np.clip(value, 0, 100)))


def fmt(n: int) -> str:
    return f"{int(n):,}"


# ============================================================
# Main UI
# ============================================================

st.title("SKA Interferometer Puzzle")
st.caption(
    "An outreach app to explore how antenna layout and telescope specifications affect "
    "uv coverage and a simplified dirty image."
)

with st.sidebar:
    st.header("Settings")

    st.subheader("Mission")
    mission = st.selectbox(
        "Observation goal",
        [
            "Resolve distant galaxies",
            "Observe extended hydrogen gas",
            "Detect faint radio sources",
            "Reduce fake patterns",
        ],
        index=1,
    )

    st.divider()
    st.subheader("Input image")

    sky_source = st.selectbox(
        "Input mode",
        ["Use a sample image", "Upload my own image"],
        index=0,
    )

    sample_model = None
    uploaded_file = None
    invert_uploaded = False
    threshold_uploaded = 0.0
    gamma_uploaded = 1.0
    manual_rotation = "0° (no rotation)"
    keep_aspect = True

    if sky_source == "Use a sample image":
        sample_model = st.selectbox(
            "Sample image",
            [
                "Extended hydrogen gas + small galaxies",
                "Many point sources",
                "Bubble-like cosmic structure",
                "SKA letters",
                "Large disk + small points",
            ],
            index=0,
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload an image file",
            type=["png", "jpg", "jpeg"],
        )
        manual_rotation = st.selectbox(
            "Manual rotation correction",
            ["0° (no rotation)", "90° clockwise", "180°", "90° counterclockwise"],
            index=0,
        )
        keep_aspect = st.checkbox("Keep aspect ratio", value=True)
        invert_uploaded = st.checkbox("Invert black/white", value=False)
        threshold_uploaded = st.slider("Background threshold", 0.0, 0.8, 0.05, 0.01)
        gamma_uploaded = st.slider("Gamma correction", 0.3, 3.0, 1.0, 0.1)

    st.divider()
    st.subheader("Telescope specification")

    preset_name = st.selectbox("Preset", list(TELESCOPE_PRESETS.keys()), index=1)
    preset = TELESCOPE_PRESETS[preset_name]

    unit = st.selectbox(
        "Distance unit",
        ["km", "arbitrary unit"],
        index=0 if preset["unit"] == "km" else 1,
        key=f"unit_{preset_name}",
    )

    actual_elements = st.number_input(
        "Interferometric elements (stations / dishes)",
        min_value=2,
        max_value=200_000,
        value=int(preset["interferometric_elements"]),
        step=1,
    )

    physical_antennas = st.number_input(
        "Physical antennas",
        min_value=2,
        max_value=500_000,
        value=int(preset["physical_antennas"]),
        step=1,
    )

    max_baseline = st.number_input(
        f"Maximum baseline ({unit})",
        min_value=0.1,
        max_value=100_000.0,
        value=float(preset["max_baseline"]),
        step=1.0,
    )

    st.caption(str(preset["note"]))

    st.divider()
    st.subheader("Image model parameters")

    signal_strength = st.slider(
        "Signal strength",
        0.01,
        2.0,
        0.20,
        0.01,
        help="Lower values make the astronomical signal weaker, so it can disappear into noise.",
    )

    base_noise = st.slider(
        "Base noise level",
        0.0,
        0.40,
        0.18,
        0.01,
        help="This sets the reference noise before sensitivity improvement is applied.",
    )

    use_physical_sensitivity = st.checkbox(
        "Use physical antenna count for sensitivity",
        value=True,
        help="If on, changing the number of physical antennas changes the noise level.",
    )
    sensitivity_elements = int(physical_antennas if use_physical_sensitivity else actual_elements)

    sensitivity_reference = st.number_input(
        "Reference count for sensitivity scaling",
        min_value=2,
        max_value=500_000,
        value=512,
        step=1,
    )

    sensitivity_effect_strength = st.slider(
        "Sensitivity effect strength",
        0.1,
        5.0,
        1.5,
        0.1,
    )

    noise_floor_fraction = st.slider(
        "Noise floor fraction",
        0.0,
        0.8,
        0.15,
        0.01,
        help="Noise will not drop below this fraction of the base noise.",
    )

    coverage_reference_elements = st.number_input(
        "Reference count for uv-filling scaling",
        min_value=2,
        max_value=500_000,
        value=512,
        step=1,
    )

    count_effect_strength = st.slider(
        "uv-filling effect strength",
        0.0,
        2.5,
        1.0,
        0.1,
    )

    educational_mode = st.checkbox(
        "Emphasize image differences (outreach mode)",
        value=True,
        help="If on, increasing element count helps fill holes in uv coverage more visibly.",
    )

    st.divider()
    st.subheader("Layout")

    layout_kind = st.selectbox(
        "Layout type",
        [
            "Core concentrated",
            "Long-baseline biased",
            "Linear",
            "Random",
            "Three-arm",
            "SKA-like balanced",
            "Manual edit",
        ],
        index=5 if "SKA" in preset_name else 3,
    )

    seed = st.slider("Random seed", 0, 999, 42, 1)

    auto_rep = st.checkbox(
        "Automatically choose displayed representative elements",
        value=True,
        help="Real arrays can be too large to draw directly, so a representative subset is displayed.",
    )

    display_cap = st.slider(
        "Upper limit of representative elements",
        8,
        220,
        int(preset["display_cap"]),
        1,
    )

    if auto_rep:
        n_rep = int(min(max(actual_elements, 4), display_cap))
        st.caption(f"Current representative elements: {n_rep}")
    else:
        n_rep = st.slider(
            "Representative elements",
            4,
            int(min(max(actual_elements, 4), 220)),
            int(min(actual_elements, display_cap)),
            1,
        )

    use_density_bonus = st.checkbox(
        "Let representative density slightly widen uv sample points",
        value=True,
    )
    compression_ratio = max(float(actual_elements) / max(n_rep, 1), 1.0)
    density_bonus = int(np.clip(round(math.log2(compression_ratio) / 1.6), 0, 6)) if use_density_bonus else 0

    st.divider()
    st.subheader("Display and comparison")

    grid = st.select_slider("Image size", options=[64, 96, 128, 160], value=128)

    reference_baseline = st.number_input(
        f"Reference baseline for uv display ({unit})",
        min_value=0.1,
        max_value=100_000.0,
        value=150.0,
        step=1.0,
        help="This changes the mapping of baselines onto the displayed uv plane.",
    )

    uv_zoom = st.slider("uv display zoom", 0.5, 2.5, 1.0, 0.1)
    point_radius = st.slider("Base uv point size", 1, 4, 2, 1)
    zero_spacing_hint = st.checkbox("Add a small central hint", value=False)
    show_true = st.checkbox("Also show the input image", value=True)

    display_mode = st.selectbox(
        "Dirty image display",
        [
            "Auto contrast (show shape clearly)",
            "Fixed contrast (show brightness difference)",
        ],
        index=1,
    )


# ============================================================
# Computation
# ============================================================

radius = max_baseline / 2.0

if layout_kind == "Manual edit":
    default = make_layout("SKA-like balanced", n_rep, radius, seed)
    df = pd.DataFrame(default, columns=[f"x ({unit})", f"y ({unit})"])
    st.subheader("Manual layout editor")
    st.write("You can edit the x and y coordinates directly.")
    edited = st.data_editor(df, num_rows="fixed", use_container_width=True)
    pos = edited[[f"x ({unit})", f"y ({unit})"]].to_numpy(float)
    pos = clip_to_radius(pos, radius)
else:
    pos = make_layout(layout_kind, n_rep, radius, seed)

if sky_source == "Upload my own image" and uploaded_file is not None:
    sky = load_uploaded_sky(
        uploaded_file=uploaded_file,
        n=grid,
        invert=invert_uploaded,
        threshold=threshold_uploaded,
        gamma=gamma_uploaded,
        manual_rotation=manual_rotation,
        keep_aspect=keep_aspect,
    )
else:
    model_to_use = sample_model if sample_model is not None else "Extended hydrogen gas + small galaxies"
    sky = make_sky(grid, model_to_use, seed + 100)

sparse_uv, outside_fraction = make_sparse_uv_weight(
    pos=pos,
    grid=grid,
    reference_baseline=reference_baseline,
    uv_zoom=uv_zoom,
    point_radius=point_radius,
    density_bonus=density_bonus,
    include_zero_spacing_hint=zero_spacing_hint,
)

envelope = make_baseline_envelope(
    grid=grid,
    max_baseline=max_baseline,
    reference_baseline=reference_baseline,
    uv_zoom=uv_zoom,
    include_zero_spacing_hint=zero_spacing_hint,
)

dirty, effective_uv, effective_noise, count_factor, sensitivity_response, snr_proxy = reconstruct_dirty_image(
    sky=sky,
    sparse_uv=sparse_uv,
    envelope=envelope,
    educational_mode=educational_mode,
    base_noise=base_noise,
    sensitivity_elements=sensitivity_elements,
    sensitivity_reference=int(sensitivity_reference),
    interferometric_elements=int(actual_elements),
    coverage_reference_elements=int(coverage_reference_elements),
    count_effect_strength=float(count_effect_strength),
    noise_floor_fraction=float(noise_floor_fraction),
    sensitivity_effect_strength=float(sensitivity_effect_strength),
    signal_strength=float(signal_strength),
    seed=seed + 200,
)

bl = baselines(pos)
bl_len = np.hypot(bl[:, 0], bl[:, 1]) if len(bl) else np.array([0.0])
rep_max_baseline = float(np.max(bl_len))
uv_fill = float(np.mean(effective_uv > 0.05))
dirty_rms = float(np.std(dirty))

# Important: fixed_vmax must NOT depend on signal_strength,
# otherwise the visual difference is canceled out.
fixed_vmax = max(float(np.percentile(np.abs(sky - np.mean(sky)), 99)), 1e-9)

scores = compute_scores(
    pos=pos,
    radius=radius,
    mission=mission,
    sensitivity_elements=sensitivity_elements,
    sensitivity_reference=int(sensitivity_reference),
)


# ============================================================
# Display
# ============================================================

st.markdown("---")
st.subheader("0. Main quantities affecting the dirty image")

mcols = st.columns(11)
with mcols[0]:
    st.metric("Interferometric elements", fmt(actual_elements))
with mcols[1]:
    st.metric("Physical antennas", fmt(physical_antennas))
with mcols[2]:
    st.metric("Max baseline", f"{max_baseline:g} {unit}")
with mcols[3]:
    st.metric("Representative elements", fmt(n_rep))
with mcols[4]:
    st.metric("Signal strength", f"{signal_strength:.2f}")
with mcols[5]:
    st.metric("uv-fill factor", f"{count_factor:.2f}")
with mcols[6]:
    st.metric("Sensitivity response", f"{sensitivity_response:.2f}")
with mcols[7]:
    st.metric("S/N proxy", f"{snr_proxy:.2f}")
with mcols[8]:
    st.metric("uv fill fraction", f"{100 * uv_fill:.2f}%")
with mcols[9]:
    st.metric("Effective noise", f"{effective_noise:.3f}")
with mcols[10]:
    st.metric("Dirty RMS", f"{dirty_rms:.3e}")

st.caption(
    "Image sharpness mainly depends on maximum baseline and layout. "
    "Noise level mainly depends on physical antenna count. "
    "Signal strength determines whether the source emerges above the noise."
)

if outside_fraction > 0.05:
    st.warning(
        f"About {100 * outside_fraction:.1f}% of uv sample points are outside the displayed uv range. "
        "Increase the reference baseline or decrease the uv zoom if needed."
    )

st.markdown("---")
st.subheader("1. Antenna layout and reconstructed image")

if show_true:
    cols = st.columns(4)
    with cols[0]:
        st.pyplot(fig_layout(pos, radius, unit), use_container_width=True)
    with cols[1]:
        st.pyplot(
            fig_uv(effective_uv, "Effective uv coverage\n(used for dirty image)"),
            use_container_width=True,
        )
    with cols[2]:
        st.pyplot(fig_dirty(dirty, display_mode, fixed_vmax), use_container_width=True)
    with cols[3]:
        st.pyplot(fig_sky(sky), use_container_width=True)
else:
    cols = st.columns(3)
    with cols[0]:
        st.pyplot(fig_layout(pos, radius, unit), use_container_width=True)
    with cols[1]:
        st.pyplot(
            fig_uv(effective_uv, "Effective uv coverage\n(used for dirty image)"),
            use_container_width=True,
        )
    with cols[2]:
        st.pyplot(fig_dirty(dirty, display_mode, fixed_vmax), use_container_width=True)

st.markdown("---")
st.subheader("2. Telescope score")

scols = st.columns(4)
with scols[0]:
    metric_bar("Resolution", scores.resolution)
with scols[1]:
    metric_bar("Sensitivity", scores.sensitivity)
with scols[2]:
    metric_bar("Extended-structure response", scores.extended)
with scols[3]:
    metric_bar("Artifact control", scores.artifact_control)

st.metric("Mission score", f"{scores.total:.0f} / 100")
st.info(scores.comment)

st.markdown("---")
st.subheader("3. What each setting actually changes")

st.write(
    """
**Settings that change the image itself**
- **Interferometric elements**: fills uv coverage and reduces fake patterns.
- **Physical antennas**: changes sensitivity, so the noise level changes.
- **Maximum baseline**: changes sharpness and accessible fine structure.
- **Signal strength**: changes how visible the source is above noise.
- **Layout type / manual layout**: changes the uv sampling pattern.

**Settings mainly for display or comparison**
- **Reference baseline for uv display**
- **uv display zoom**
- **Base uv point size**
- **Auto/fixed contrast**
- **Representative element cap**
"""
)

with st.expander("Expert note"):
    st.write(
        f"""
This app is not a precision radio interferometric imaging code.
It combines sparse baseline sampling with an educational Fourier envelope and a simplified noise model.

Current internal values:
- preset: {preset_name}
- layout: {layout_kind}
- input mode: {sky_source}
- signal strength: {signal_strength:.3f}
- actual interferometric elements: {fmt(actual_elements)}
- physical antennas: {fmt(physical_antennas)}
- sensitivity elements used: {fmt(sensitivity_elements)}
- max baseline: {max_baseline:g} {unit}
- representative elements: {fmt(n_rep)}
- representative max baseline: {rep_max_baseline:g} {unit}
- density bonus: {density_bonus}
- uv count factor: {count_factor:.4f}
- sensitivity response: {sensitivity_response:.4f}
- coverage reference elements: {fmt(coverage_reference_elements)}
- sensitivity reference elements: {fmt(sensitivity_reference)}
- noise floor fraction: {noise_floor_fraction:.3f}
- uv filling fraction: {100 * uv_fill:.3f}%
- uv outside fraction: {100 * outside_fraction:.3f}%
- effective noise: {effective_noise:.4f}
- S/N proxy: {snr_proxy:.4f}
- dirty image RMS: {dirty_rms:.4e}
"""
    )

st.caption("Educational prototype for SKA outreach.")
