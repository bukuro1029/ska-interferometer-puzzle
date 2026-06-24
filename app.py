"""
SKA Interferometer Puzzle — Streamlit app with UDP hardware input

Run:
    pip install streamlit numpy matplotlib pandas pillow
    streamlit run app.py

Purpose:
    Educational outreach app showing how interferometric element count,
    maximum baseline, and antenna layout affect uv coverage and a simplified
    dirty image.

Hardware mode:
    - Receives ASCII contact-grid packets via UDP, default localhost:9900.
    - Packet example for an 8x3 array:
          10100100 10100111 11101100
      Each row is a string of 0/1 values. Rows are separated by spaces.
    - Converts active grid cells to antenna/station coordinates using either
      grid_mapping.csv or an automatically generated default mapping.

Important:
    This is an outreach-oriented educational simulator, not a precision radio
    interferometric imaging pipeline.

Language policy:
    - Matplotlib figure labels are in English to avoid font rendering issues online.
    - Streamlit UI and explanations are in Japanese.
"""

from __future__ import annotations

import math
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    "教育用ミニ干渉計": {
        "interferometric_elements": 16,
        "max_baseline": 20.0,
        "display_cap": 64,
        "physical_info": "仮想的な小型配列。実機仕様ではありません。",
    },
    "SKA-Lowスケール": {
        "interferometric_elements": 512,
        "max_baseline": 74.0,
        "display_cap": 128,
        "physical_info": (
            "SKA-Lowでは、131,072本の低周波アンテナが512 stationにまとめられます。"
            "このアプリでは、画像再構成に効く干渉計要素としてstation数を扱います。"
        ),
    },
    "SKA-Midスケール": {
        "interferometric_elements": 197,
        "max_baseline": 150.0,
        "display_cap": 128,
        "physical_info": (
            "SKA-Midでは、197台のディッシュを干渉計要素として扱います。"
            "このアプリでは、その配置と基線長が画像再構成にどう効くかを示します。"
        ),
    },
    "カスタム": {
        "interferometric_elements": 64,
        "max_baseline": 100.0,
        "display_cap": 128,
        "physical_info": "任意の干渉計要素数と最大基線を入力できます。",
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
    """Simple dependency-free blur for educational uv filling display."""
    out = np.asarray(arr, dtype=float)

    for _ in range(max(0, passes)):
        out = (
            4.0 * out
            + 2.0
            * (
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


def uv_count_factor(elements: int, reference_elements: int, strength: float = 2.0) -> float:
    """
    Convert interferometric element count to a 0-1 factor for uv filling.

    This is intentionally tuned for outreach: increasing the element count should
    visibly fill the uv plane.
    """
    n = max(float(elements), 2.0)
    ref = max(float(reference_elements), 2.0)

    pairs = n * (n - 1.0) / 2.0
    ref_pairs = ref * (ref - 1.0) / 2.0

    x = np.log10(pairs + 1.0) / np.log10(ref_pairs + 1.0)
    x = np.clip(x, 0.0, 1.0)

    response = x ** (1.0 / max(strength, 1e-6))
    return float(np.clip(response, 0.0, 1.0))


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

    if model == "広がった水素ガス + 小銀河":
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

    elif model == "多数の点源":
        for _ in range(26):
            sky += gaussian_2d(
                n,
                rng.uniform(-0.88, 0.88),
                rng.uniform(-0.88, 0.88),
                rng.uniform(0.008, 0.022),
                rng.uniform(0.008, 0.022),
                amp=rng.uniform(0.25, 1.25),
            )

    elif model == "泡構造風の宇宙":
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

    elif model == "SKAの文字":
        # S
        sky += gaussian_2d(n, -0.55, 0.45, 0.16, 0.08, amp=0.9)
        sky += gaussian_2d(n, -0.60, 0.15, 0.10, 0.08, amp=0.9)
        sky += gaussian_2d(n, -0.50, -0.15, 0.10, 0.08, amp=0.9)
        sky += gaussian_2d(n, -0.60, -0.45, 0.16, 0.08, amp=0.9)

        # K
        for t in np.linspace(-0.55, 0.55, 12):
            sky += gaussian_2d(n, -0.05, t, 0.02, 0.02, amp=0.9)
        for t in np.linspace(-0.45, 0.45, 12):
            sky += gaussian_2d(n, 0.12 + 0.25 * abs(t), t, 0.02, 0.02, amp=0.9)

        # A
        for t in np.linspace(-0.5, 0.5, 18):
            y = -0.5 + t
            sky += gaussian_2d(n, 0.55 - 0.25 * t, y, 0.02, 0.02, amp=0.9)
            sky += gaussian_2d(n, 0.55 + 0.25 * t, y, 0.02, 0.02, amp=0.9)
        for x in np.linspace(0.40, 0.70, 12):
            sky += gaussian_2d(n, x, 0.0, 0.02, 0.02, amp=0.9)

    elif model == "大きな円 + 小さな点":
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
    if rotation_label == "0°（そのまま）":
        return img
    if rotation_label == "90° 時計回り":
        return img.rotate(-90, expand=True)
    if rotation_label == "180°":
        return img.rotate(180, expand=True)
    if rotation_label == "90° 反時計回り":
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
    manual_rotation: str = "0°（そのまま）",
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
    arr = arr**gamma

    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()

    return arr


# ============================================================
# Hardware packet and grid mapping utilities
# ============================================================


def normalize_packet_text(packet: str) -> str:
    """Normalize whitespace in a contact packet."""
    return " ".join(str(packet).replace("¥n", " ").replace("¥r", " ").split())


def parse_contact_packet(packet: str) -> Tuple[List[Tuple[int, int]], Optional[int], Optional[int], str, Optional[str]]:
    """
    Parse ASCII contact packet.

    Packet format:
        row0 row1 row2 ...
    Example:
        10100100 10100111 11101100

    Returns:
        contacts, rows, cols, normalized_packet, error_message
    """
    normalized = normalize_packet_text(packet)
    if not normalized:
        return [], None, None, "", "パケットが空です。"

    row_strings = normalized.split()
    cols = len(row_strings[0])
    if cols == 0:
        return [], None, None, normalized, "列数が0です。"

    for row in row_strings:
        if len(row) != cols:
            return [], None, None, normalized, "行ごとの文字数が一致していません。"
        bad = [ch for ch in row if ch not in {"0", "1"}]
        if bad:
            return [], None, None, normalized, "0/1以外の文字が含まれています。"

    contacts: List[Tuple[int, int]] = []
    for r, row in enumerate(row_strings):
        for c, ch in enumerate(row):
            if ch == "1":
                contacts.append((r, c))

    return contacts, len(row_strings), cols, normalized, None


def make_default_grid_mapping(rows: int, cols: int, max_baseline: float) -> pd.DataFrame:
    """Create a simple rectangular grid mapping.

    The square side is chosen so that the corner-to-corner distance is roughly
    the maximum baseline. Row 0 is displayed at the top.
    """
    rows = max(int(rows), 1)
    cols = max(int(cols), 1)
    half_side = max_baseline / (2.0 * math.sqrt(2.0))
    xs = np.linspace(-half_side, half_side, cols)
    ys = np.linspace(half_side, -half_side, rows)

    data = []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            data.append({"row": r, "col": c, "x": float(x), "y": float(y), "enabled": 1})
    return pd.DataFrame(data)


def load_grid_mapping(
    mapping_path: str,
    uploaded_mapping,
    rows: int,
    cols: int,
    max_baseline: float,
) -> Tuple[pd.DataFrame, str, Optional[str]]:
    """Load grid mapping from uploaded CSV or local CSV path. Fallback to default mapping."""
    required = {"row", "col", "x", "y"}

    try:
        if uploaded_mapping is not None:
            df = pd.read_csv(uploaded_mapping)
            source = "アップロードされた対応表CSV"
        elif mapping_path and Path(mapping_path).exists():
            df = pd.read_csv(mapping_path)
            source = f"ローカル対応表CSV: {mapping_path}"
        else:
            df = make_default_grid_mapping(rows, cols, max_baseline)
            source = "自動生成したデフォルト対応表"
            return df, source, None

        missing = required - set(df.columns)
        if missing:
            df_default = make_default_grid_mapping(rows, cols, max_baseline)
            return (
                df_default,
                "自動生成したデフォルト対応表",
                f"対応表CSVに必要な列がありません: {sorted(missing)}。デフォルト対応表を使います。",
            )

        df = df.copy()
        df["row"] = df["row"].astype(int)
        df["col"] = df["col"].astype(int)
        df["x"] = df["x"].astype(float)
        df["y"] = df["y"].astype(float)
        if "enabled" not in df.columns:
            df["enabled"] = 1
        df = df[df["enabled"].astype(int) != 0]
        return df, source, None
    except Exception as exc:  # noqa: BLE001
        df_default = make_default_grid_mapping(rows, cols, max_baseline)
        return (
            df_default,
            "自動生成したデフォルト対応表",
            f"対応表CSVの読み込みに失敗しました: {exc}。デフォルト対応表を使います。",
        )


def contacts_to_positions(contacts: List[Tuple[int, int]], mapping_df: pd.DataFrame) -> np.ndarray:
    lookup: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for _, row in mapping_df.iterrows():
        lookup[(int(row["row"]), int(row["col"]))] = (float(row["x"]), float(row["y"]))

    positions: List[Tuple[float, float]] = []
    for key in contacts:
        if key in lookup:
            positions.append(lookup[key])

    if not positions:
        return np.empty((0, 2), dtype=float)
    return np.asarray(positions, dtype=float)


@st.cache_resource(show_spinner=False)
def get_udp_socket(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, int(port)))
    sock.setblocking(False)
    return sock


def read_latest_udp_packet(sock: socket.socket, max_packets: int = 100) -> Tuple[Optional[str], int, Optional[str]]:
    """Read all currently available UDP packets and return only the latest one."""
    latest: Optional[str] = None
    count = 0
    last_addr: Optional[str] = None

    for _ in range(max_packets):
        try:
            data, addr = sock.recvfrom(65535)
        except BlockingIOError:
            break
        except OSError:
            break
        latest = data.decode("ascii", errors="replace").strip()
        count += 1
        last_addr = f"{addr[0]}:{addr[1]}"

    return latest, count, last_addr


def safe_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover - for older Streamlit
        st.experimental_rerun()


# ============================================================
# Layout generation
# ============================================================


def clip_to_radius(pos: np.ndarray, radius: float) -> np.ndarray:
    if len(pos) == 0:
        return np.empty((0, 2), dtype=float)
    r = np.hypot(pos[:, 0], pos[:, 1])
    mask = r > radius
    if np.any(mask):
        pos[mask] *= (radius / r[mask])[:, None]
    return pos


def make_layout(kind: str, n: int, radius: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if kind == "中心集中型":
        return clip_to_radius(rng.normal(0, 0.18 * radius, size=(n, 2)), radius)

    if kind == "長基線重視型":
        n_core = max(3, n // 3)
        core = rng.normal(0, 0.12 * radius, size=(n_core, 2))
        n_outer = n - n_core
        th = rng.uniform(0, 2 * np.pi, n_outer)
        rr = rng.uniform(0.65 * radius, radius, n_outer)
        outer = np.column_stack([rr * np.cos(th), rr * np.sin(th)])
        return np.vstack([core, outer])

    if kind == "直線型":
        x = np.linspace(-radius, radius, n)
        y = rng.normal(0, 0.025 * radius, n)
        return np.column_stack([x, y])

    if kind == "ランダム型":
        th = rng.uniform(0, 2 * np.pi, n)
        rr = radius * np.sqrt(rng.uniform(0, 1, n))
        return np.column_stack([rr * np.cos(th), rr * np.sin(th)])

    if kind == "三本腕型":
        arms = np.arange(n) % 3
        base = arms * 2 * np.pi / 3
        rr = radius * (0.08 + 0.92 * np.linspace(0, 1, n))
        rng.shuffle(rr)
        th = base + 0.60 * rr / max(radius, 1e-12) + rng.normal(0, 0.08, n)
        pos = np.column_stack([rr * np.cos(th), rr * np.sin(th)])
        pos += rng.normal(0, 0.015 * radius, size=pos.shape)
        return clip_to_radius(pos, radius)

    if kind == "SKA風バランス型":
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
    if len(pos) < 2:
        return np.empty((0, 2), dtype=float)

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
    include_zero_spacing_hint: bool,
) -> Tuple[np.ndarray, float]:
    bl = baselines(pos)
    weight = np.zeros((grid, grid), dtype=float)
    if len(bl) == 0:
        return weight, 0.0

    c = grid // 2
    scale = (0.45 * grid * uv_zoom) / max(reference_baseline, 1e-12)
    r_pix = max(1, int(point_radius))
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
    base_noise: float,
    signal_strength: float,
    interferometric_elements: int,
    coverage_reference_elements: int,
    count_effect_strength: float,
    educational_mode: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Returns:
        dirty, effective_uv, count_factor, snr_proxy
    """
    rng = np.random.default_rng(seed)

    sky_ref = sky - np.mean(sky)
    ft_ref = np.fft.fftshift(np.fft.fft2(sky_ref))
    ft_signal = signal_strength * ft_ref

    count_factor = uv_count_factor(
        elements=interferometric_elements,
        reference_elements=coverage_reference_elements,
        strength=count_effect_strength,
    )
    alpha = np.clip(count_factor, 0.0, 1.0)

    if educational_mode:
        # Outreach-oriented effect:
        # low element count -> sparse uv coverage;
        # high element count -> visibly fuller uv coverage.
        smooth_uv_1 = blur_array(sparse_uv, passes=1 + int(4 * alpha))
        smooth_uv_2 = blur_array(sparse_uv, passes=4 + int(8 * alpha))
        filled_uv = (
            (1.0 - alpha) * sparse_uv
            + 0.65 * alpha * smooth_uv_1
            + 0.35 * alpha * smooth_uv_2
        )
        effective_uv = envelope * np.clip(filled_uv, 0.0, 1.0)
    else:
        effective_uv = envelope * sparse_uv

    sampled = ft_signal * effective_uv

    if base_noise > 0:
        amp = np.percentile(np.abs(ft_ref), 95) * base_noise
        noise = amp * (rng.normal(size=ft_ref.shape) + 1j * rng.normal(size=ft_ref.shape))
        sampled += noise * effective_uv

    dirty = np.real(np.fft.ifft2(np.fft.ifftshift(sampled)))
    dirty -= np.mean(dirty)

    signal_rms = float(np.std(signal_strength * sky_ref))
    noise_rms_proxy = float(base_noise * np.std(sky_ref))
    snr_proxy = signal_rms / max(noise_rms_proxy, 1e-9)

    return dirty, effective_uv, float(count_factor), float(snr_proxy)


# ============================================================
# Scoring
# ============================================================


@dataclass
class Scores:
    resolution: float
    uv_coverage: float
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
    count_factor: float,
) -> Scores:
    if len(pos) < 2:
        return Scores(0.0, 0.0, 0.0, 0.0, 0.0, "アンテナは少なくとも2個必要です。")

    bl = baselines(pos)
    length = np.hypot(bl[:, 0], bl[:, 1])
    max_possible = max(2 * radius, 1e-12)

    resolution = np.clip(100 * np.max(length) / max_possible, 0, 100)
    uv_coverage = np.clip(100 * count_factor, 0, 100)
    extended = np.clip(100 * np.mean(length < 0.35 * radius) / 0.45, 0, 100)

    angle = np.mod(np.arctan2(bl[:, 1], bl[:, 0]), np.pi)
    hist, _ = np.histogram(angle, bins=12, range=(0, np.pi))
    angle_score = entropy01(hist / max(hist.sum(), 1))
    radial_balance = 1.0 - min(1.0, abs(np.median(length) - 0.65 * radius) / (0.65 * radius))
    artifact_control = np.clip(100 * (0.75 * angle_score + 0.25 * radial_balance), 0, 100)

    weights = {
        "遠くの銀河を細かく見る": (0.50, 0.20, 0.05, 0.25),
        "広がった水素ガスを見る": (0.15, 0.20, 0.45, 0.20),
        "暗い電波源を見つける": (0.20, 0.35, 0.10, 0.35),
        "偽物の模様を減らす": (0.15, 0.40, 0.05, 0.40),
    }[mission]

    total = (
        weights[0] * resolution
        + weights[1] * uv_coverage
        + weights[2] * extended
        + weights[3] * artifact_control
    )

    weak = min(
        [
            (resolution, "細かい構造を見る力が弱いです。最大基線を長くすると改善します。"),
            (uv_coverage, "uv coverageが疎です。干渉計要素数を増やすと改善します。"),
            (extended, "広がった構造に弱いです。中心部に短い基線を増やすと改善します。"),
            (artifact_control, "配置の偏りが大きく、偽の模様が出やすいです。方向を分散させると改善します。"),
        ],
        key=lambda x: x[0],
    )

    comment = weak[1]
    if total >= 82:
        comment = "このミッションに対して、かなりバランスのよい配置です。"
    elif total >= 65:
        comment = "まずまずの配置ですが、まだ明確な弱点があります。" + weak[1]

    return Scores(
        float(resolution),
        float(uv_coverage),
        float(extended),
        float(artifact_control),
        float(total),
        comment,
    )


# ============================================================
# Plotting
# ============================================================


def fig_layout(pos: np.ndarray, radius: float, unit: str) -> plt.Figure:
    plot_unit = "km" if unit == "km" else "arb. unit"
    fig, ax = plt.subplots(figsize=(4, 4))
    circle = plt.Circle((0, 0), radius, fill=False, linestyle="--", linewidth=1.2)
    ax.add_patch(circle)
    if len(pos) > 0:
        ax.scatter(pos[:, 0], pos[:, 1], s=38)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.08 * radius, 1.08 * radius)
    ax.set_ylim(-1.08 * radius, 1.08 * radius)
    ax.set_title("Antenna / station layout")
    ax.set_xlabel(f"East-West ({plot_unit})")
    ax.set_ylabel(f"North-South ({plot_unit})")
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
    if display_mode == "自動コントラスト（形を見やすく）":
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
# Session state for hardware mode
# ============================================================

if "last_udp_packet" not in st.session_state:
    st.session_state.last_udp_packet = ""
if "last_udp_time" not in st.session_state:
    st.session_state.last_udp_time = None
if "last_udp_addr" not in st.session_state:
    st.session_state.last_udp_addr = None
if "last_udp_count" not in st.session_state:
    st.session_state.last_udp_count = 0
if "last_packet_error" not in st.session_state:
    st.session_state.last_packet_error = None


# ============================================================
# Main UI
# ============================================================

st.title("SKA干渉計パズル")
st.caption(
    "アンテナ配置や望遠鏡仕様を変えると、uv coverage と簡易 dirty image がどう変わるかを体験する展示用アプリです。"
)

with st.sidebar:
    st.header("設定")

    st.subheader("観測ミッション")
    mission = st.selectbox(
        "観測の目的",
        [
            "遠くの銀河を細かく見る",
            "広がった水素ガスを見る",
            "暗い電波源を見つける",
            "偽物の模様を減らす",
        ],
        index=3,
    )

    st.divider()
    st.subheader("入力画像")

    sky_source = st.selectbox(
        "入力モード",
        ["サンプル画像を使う", "自分の画像をアップロード"],
        index=0,
    )

    sample_model = None
    uploaded_file = None
    invert_uploaded = False
    threshold_uploaded = 0.0
    gamma_uploaded = 1.0
    manual_rotation = "0°（そのまま）"
    keep_aspect = True

    if sky_source == "サンプル画像を使う":
        sample_model = st.selectbox(
            "サンプル画像",
            [
                "広がった水素ガス + 小銀河",
                "多数の点源",
                "泡構造風の宇宙",
                "SKAの文字",
                "大きな円 + 小さな点",
            ],
            index=3,
        )
    else:
        uploaded_file = st.file_uploader(
            "画像ファイルをアップロード",
            type=["png", "jpg", "jpeg"],
        )
        manual_rotation = st.selectbox(
            "アップロード画像の回転補正",
            ["0°（そのまま）", "90° 時計回り", "180°", "90° 反時計回り"],
            index=0,
        )
        keep_aspect = st.checkbox("縦横比を保つ", value=True)
        invert_uploaded = st.checkbox("白黒を反転する", value=False)
        threshold_uploaded = st.slider("背景カット閾値", 0.0, 0.8, 0.05, 0.01)
        gamma_uploaded = st.slider("ガンマ補正", 0.3, 3.0, 1.0, 0.1)

    st.divider()
    st.subheader("アンテナ位置入力")

    position_input_mode = st.selectbox(
        "位置入力モード",
        ["手動・プリセット配置", "UDPハードウェア入力", "テキストパケット入力（テスト用）"],
        index=0,
        help="UDPハードウェア入力では、localhost:9900などで受け取った接点情報からアンテナ位置を決めます。",
    )

    st.divider()
    st.subheader("望遠鏡仕様")

    preset_name = st.selectbox("プリセット", list(TELESCOPE_PRESETS.keys()), index=1)
    preset = TELESCOPE_PRESETS[preset_name]

    unit = st.selectbox(
        "距離単位",
        ["km", "任意単位"],
        index=0,
        key=f"unit_{preset_name}",
    )

    if position_input_mode == "手動・プリセット配置":
        actual_elements_input = st.number_input(
            "干渉計要素数（station / dish 数）",
            min_value=2,
            max_value=200_000,
            value=int(preset["interferometric_elements"]),
            step=1,
        )
    else:
        st.caption("UDP/テキスト入力時は、検出された接点数を干渉計要素数として使います。")
        actual_elements_input = 0

    max_baseline = st.number_input(
        f"最大基線・最大分離（{unit}）",
        min_value=0.1,
        max_value=100_000.0,
        value=float(preset["max_baseline"]),
        step=1.0,
    )

    st.caption(str(preset["physical_info"]))

    st.divider()
    st.subheader("画像モデルの設定")

    signal_strength = st.slider(
        "信号の強さ",
        0.01,
        2.0,
        0.50,
        0.01,
        help="小さいほど天体信号が弱くなり、ノイズに埋もれやすくなります。",
    )

    base_noise = st.slider(
        "基準ノイズレベル",
        0.0,
        0.40,
        0.12,
        0.01,
        help="この値を大きくすると、再構成画像にノイズが強く出ます。",
    )

    coverage_default = 16 if position_input_mode != "手動・プリセット配置" else 512
    coverage_reference_elements = st.number_input(
        "uv充填スケーリングの基準要素数",
        min_value=2,
        max_value=500_000,
        value=coverage_default,
        step=1,
        help="この値に近づくほど、uv coverageが埋まるように表示します。展示用の最大アンテナ数を入れると分かりやすいです。",
    )

    count_effect_strength = st.slider(
        "uv充填効果の強さ",
        0.0,
        2.5,
        2.0,
        0.1,
        help="大きくすると、干渉計要素数を増やしたときの変化が見えやすくなります。",
    )

    educational_mode = st.checkbox(
        "画像変化を強調する（展示用）",
        value=True,
        help="オンにすると、要素数が増えたときにuv coverageの穴が埋まる効果を見えやすくします。",
    )

    st.divider()

    # Position-source specific controls
    if position_input_mode == "手動・プリセット配置":
        st.subheader("配置")

        layout_kind = st.selectbox(
            "配置タイプ",
            [
                "中心集中型",
                "長基線重視型",
                "直線型",
                "ランダム型",
                "三本腕型",
                "SKA風バランス型",
                "手動編集",
            ],
            index=5 if "SKA" in preset_name else 3,
        )

        seed = st.slider("乱数シード", 0, 999, 42, 1)

        auto_rep = st.checkbox(
            "表示用の代表要素数を自動設定する",
            value=True,
            help="実機の全要素をそのまま描くと重いため、代表点で表示します。",
        )

        display_cap = st.slider(
            "代表要素数の上限",
            8,
            512,
            int(preset["display_cap"]),
            1,
        )

        if auto_rep:
            n_rep = int(min(max(int(actual_elements_input), 4), display_cap))
            st.caption(f"現在の代表要素数：{n_rep}")
        else:
            n_rep = st.slider(
                "代表要素数",
                4,
                int(min(max(int(actual_elements_input), 4), 512)),
                int(min(int(actual_elements_input), display_cap)),
                1,
            )

        hardware_packet = ""
        hardware_packet_error = None
        mapping_df = pd.DataFrame()
        mapping_source = ""
        mapping_warning = None
        udp_auto_update = False
        udp_update_interval = 0.5

    else:
        st.subheader("接点パケット入力")

        default_rows = st.slider("デフォルト対応表の行数", 3, 16, 8, 1)
        default_cols = st.slider("デフォルト対応表の列数", 3, 16, 8, 1)
        mapping_path = st.text_input("grid_mapping.csv のパス", value="grid_mapping.csv")
        uploaded_mapping = st.file_uploader("対応表CSVをアップロード（任意）", type=["csv"])

        mapping_df, mapping_source, mapping_warning = load_grid_mapping(
            mapping_path=mapping_path,
            uploaded_mapping=uploaded_mapping,
            rows=default_rows,
            cols=default_cols,
            max_baseline=max_baseline,
        )
        sample_csv = make_default_grid_mapping(default_rows, default_cols, max_baseline).to_csv(index=False).encode("utf-8")
        st.download_button(
            "サンプル対応表CSVをダウンロード",
            data=sample_csv,
            file_name="grid_mapping_sample.csv",
            mime="text/csv",
        )

        if position_input_mode == "UDPハードウェア入力":
            udp_host = st.text_input("UDP受信ホスト", value="127.0.0.1")
            udp_port = st.number_input("UDP受信ポート", min_value=1, max_value=65535, value=9900, step=1)
            udp_update_interval = st.slider("画面更新間隔（秒）", 0.2, 2.0, 0.5, 0.1)
            udp_auto_update = st.checkbox("自動更新する", value=True)

            try:
                sock = get_udp_socket(udp_host, int(udp_port))
                latest, packet_count, last_addr = read_latest_udp_packet(sock)
                if latest is not None:
                    st.session_state.last_udp_packet = latest
                    st.session_state.last_udp_time = time.time()
                    st.session_state.last_udp_addr = last_addr
                    st.session_state.last_udp_count += packet_count
                hardware_packet = st.session_state.last_udp_packet
                hardware_packet_error = None
            except Exception as exc:  # noqa: BLE001
                hardware_packet = st.session_state.last_udp_packet
                hardware_packet_error = f"UDP受信ソケットを開けませんでした: {exc}"
        else:
            udp_auto_update = False
            udp_update_interval = 0.5
            hardware_packet = st.text_area(
                "テスト用パケット",
                value="10000001 01000010 00100100 00011000 00011000 00100100 01000010 10000001",
                help="例: 10100100 10100111 11101100",
            )
            hardware_packet_error = None

        seed = st.slider("乱数シード", 0, 999, 42, 1)
        layout_kind = "UDP/テキスト入力"
        n_rep = 0

    st.divider()
    st.subheader("表示・比較設定")

    grid = st.select_slider("画像サイズ", options=[64, 96, 128, 160], value=96 if position_input_mode != "手動・プリセット配置" else 128)

    reference_baseline = st.number_input(
        f"uv表示用の基準最大基線（{unit}）",
        min_value=0.1,
        max_value=100_000.0,
        value=150.0,
        step=1.0,
        help="基線をuv平面上にどう表示するかを決める比較用の基準です。",
    )

    uv_zoom = st.slider("uv表示倍率", 0.5, 2.5, 1.0, 0.1)
    point_radius = st.slider("uv点の基本サイズ", 1, 4, 2, 1)
    zero_spacing_hint = st.checkbox("中心成分を少し補う（教育用）", value=False)
    show_true = st.checkbox("入力画像も表示する", value=True)

    display_mode = st.selectbox(
        "dirty画像の表示",
        [
            "自動コントラスト（形を見やすく）",
            "固定コントラスト（明るさの差を見やすく）",
        ],
        index=1,
    )


# ============================================================
# Computation
# ============================================================

radius = max_baseline / 2.0

if position_input_mode == "手動・プリセット配置":
    if layout_kind == "手動編集":
        default = make_layout("SKA風バランス型", n_rep, radius, seed)
        df = pd.DataFrame(default, columns=[f"x ({unit})", f"y ({unit})"])
        st.subheader("手動配置エディタ")
        st.write("x, y 座標を直接編集できます。")
        edited = st.data_editor(df, num_rows="fixed", use_container_width=True)
        pos = edited[[f"x ({unit})", f"y ({unit})"]].to_numpy(float)
        pos = clip_to_radius(pos, radius)
    else:
        pos = make_layout(layout_kind, n_rep, radius, seed)
    actual_elements = int(actual_elements_input)
    packet_status_message = "手動・プリセット配置を使用中"
    packet_contacts: List[Tuple[int, int]] = []
    packet_rows = None
    packet_cols = None
    normalized_packet = ""
    parse_error = None
else:
    contacts, packet_rows, packet_cols, normalized_packet, parse_error = parse_contact_packet(hardware_packet)
    packet_contacts = contacts
    if parse_error is None:
        pos = contacts_to_positions(contacts, mapping_df)
    else:
        pos = np.empty((0, 2), dtype=float)
    actual_elements = int(len(pos))
    n_rep = actual_elements
    packet_status_message = "UDP/テキスト接点パケットを使用中"

if sky_source == "自分の画像をアップロード" and uploaded_file is not None:
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
    model_to_use = sample_model if sample_model is not None else "SKAの文字"
    sky = make_sky(grid, model_to_use, seed + 100)

sparse_uv, outside_fraction = make_sparse_uv_weight(
    pos=pos,
    grid=grid,
    reference_baseline=reference_baseline,
    uv_zoom=uv_zoom,
    point_radius=point_radius,
    include_zero_spacing_hint=zero_spacing_hint,
)

envelope = make_baseline_envelope(
    grid=grid,
    max_baseline=max_baseline,
    reference_baseline=reference_baseline,
    uv_zoom=uv_zoom,
    include_zero_spacing_hint=zero_spacing_hint,
)

dirty, effective_uv, count_factor, snr_proxy = reconstruct_dirty_image(
    sky=sky,
    sparse_uv=sparse_uv,
    envelope=envelope,
    base_noise=base_noise,
    signal_strength=float(signal_strength),
    interferometric_elements=max(int(actual_elements), 2),
    coverage_reference_elements=int(coverage_reference_elements),
    count_effect_strength=float(count_effect_strength),
    educational_mode=educational_mode,
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
    radius=max(radius, 1e-12),
    mission=mission,
    count_factor=count_factor,
)


# ============================================================
# Display
# ============================================================

st.markdown("---")
st.subheader("0. dirty画像に効いている主な量")

mcols = st.columns(9)
with mcols[0]:
    st.metric("干渉計要素", fmt(actual_elements))
with mcols[1]:
    st.metric("最大基線", f"{max_baseline:g} {unit}")
with mcols[2]:
    st.metric("代表要素数", fmt(n_rep))
with mcols[3]:
    st.metric("信号強度", f"{signal_strength:.2f}")
with mcols[4]:
    st.metric("基準ノイズ", f"{base_noise:.2f}")
with mcols[5]:
    st.metric("uv充填係数", f"{count_factor:.2f}")
with mcols[6]:
    st.metric("S/N指標", f"{snr_proxy:.2f}")
with mcols[7]:
    st.metric("uv充填率", f"{100 * uv_fill:.2f}%")
with mcols[8]:
    st.metric("dirty RMS", f"{dirty_rms:.3e}")

st.caption(
    "画像の細かさは主に最大基線と配置で決まり、干渉計要素数を増やすとuv coverageの穴が埋まり、偽物の模様が減ります。"
)

if position_input_mode != "手動・プリセット配置":
    st.markdown("---")
    st.subheader("接点パケット受信状態")
    status_cols = st.columns(4)
    with status_cols[0]:
        st.metric("検出接点数", len(packet_contacts))
    with status_cols[1]:
        st.metric("有効アンテナ数", len(pos))
    with status_cols[2]:
        st.metric("グリッド", f"{packet_cols or '-'} x {packet_rows or '-'}")
    with status_cols[3]:
        if st.session_state.last_udp_time is None:
            last_time_text = "未受信"
        else:
            last_time_text = f"{time.time() - st.session_state.last_udp_time:.1f}秒前"
        st.metric("最終UDP受信", last_time_text)

    st.caption(packet_status_message)
    st.caption(f"対応表: {mapping_source}")
    if mapping_warning:
        st.warning(mapping_warning)
    if hardware_packet_error:
        st.error(hardware_packet_error)
    if parse_error:
        st.error(parse_error)
    if normalized_packet:
        st.code(normalized_packet, language="text")
    if st.session_state.last_udp_addr:
        st.caption(f"最後のUDP送信元: {st.session_state.last_udp_addr} / 累積受信数: {st.session_state.last_udp_count}")

if outside_fraction > 0.05:
    st.warning(
        f"uv点の約 {100 * outside_fraction:.1f}% が表示範囲外です。"
        "uv表示用の基準最大基線を大きくするか、uv表示倍率を下げると見やすくなります。"
    )

st.markdown("---")
st.subheader("1. アンテナ配置と再構成画像")

if show_true:
    cols = st.columns(4)
    with cols[0]:
        st.pyplot(fig_layout(pos, radius, unit), use_container_width=True)
    with cols[1]:
        st.pyplot(
            fig_uv(effective_uv, "Effective uv coverage¥n(used for dirty image)"),
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
            fig_uv(effective_uv, "Effective uv coverage¥n(used for dirty image)"),
            use_container_width=True,
        )
    with cols[2]:
        st.pyplot(fig_dirty(dirty, display_mode, fixed_vmax), use_container_width=True)

st.markdown("---")
st.subheader("2. 望遠鏡スコア")

scols = st.columns(4)
with scols[0]:
    metric_bar("解像度", scores.resolution)
with scols[1]:
    metric_bar("uv coverage", scores.uv_coverage)
with scols[2]:
    metric_bar("広がった構造への強さ", scores.extended)
with scols[3]:
    metric_bar("偽模様の少なさ", scores.artifact_control)

st.metric("ミッション達成度", f"{scores.total:.0f} / 100")
st.info(scores.comment)

st.markdown("---")
st.subheader("3. UDPハードウェア入力の使い方")

st.write(
    """
接点検出ソフトから、以下のようなASCIIパケットをUDPで送ってください。

```text
10100100 10100111 11101100
```

- 送信先は通常 `127.0.0.1:9900` です。
- 各行は0/1の文字列です。
- `1` の接点をアンテナ模型が置かれた位置として扱います。
- `grid_mapping.csv` がある場合は、その対応表で `(row, col)` を `(x, y)` に変換します。
- `grid_mapping.csv` がない場合は、自動生成した矩形グリッドを使います。
"""
)

st.markdown("---")
st.subheader("4. 各設定が何に効くか")

st.write(
    """
**画像そのものに効く設定**
- **干渉計要素数 / 検出された接点数**：uv coverageを埋め、偽物の模様を減らす。
- **最大基線**：解像度を上げ、細かい構造を見えるようにする。
- **配置タイプ / UDP接点位置 / 手動配置**：uv sampling patternを変える。
- **信号の強さ**：天体信号がノイズの上に現れるかどうかを決める。
- **基準ノイズレベル**：信号の見えにくさを変える。

**説明として残したもの**
- SKA-Lowの131,072本の物理アンテナは、512 stationにまとめられます。
- このアプリでは、画像再構成に直接効く要素としてstation/dish数、または接点で検出されたアンテナ模型の数を操作します。
"""
)

with st.expander("専門家向けメモ"):
    st.write(
        f"""
このアプリは、厳密な電波干渉計イメージングコードではありません。
疎な baseline sampling、最大基線に対応する教育用 Fourier envelope、簡略化したノイズモデルを組み合わせています。

現在の内部量：
- position input mode: {position_input_mode}
- preset: {preset_name}
- layout: {layout_kind}
- input mode: {sky_source}
- signal strength: {signal_strength:.3f}
- actual interferometric elements: {fmt(actual_elements)}
- max baseline: {max_baseline:g} {unit}
- representative elements: {fmt(n_rep)}
- representative max baseline: {rep_max_baseline:g} {unit}
- uv count factor: {count_factor:.4f}
- coverage reference elements: {fmt(coverage_reference_elements)}
- uv filling fraction: {100 * uv_fill:.3f}%
- uv outside fraction: {100 * outside_fraction:.3f}%
- base noise: {base_noise:.4f}
- S/N proxy: {snr_proxy:.4f}
- dirty image RMS: {dirty_rms:.4e}
"""
    )

st.caption("SKAアウトリーチ用の教育プロトタイプです。")


# ============================================================
# Auto refresh for UDP hardware mode
# ============================================================

if position_input_mode == "UDPハードウェア入力" and udp_auto_update:
    time.sleep(float(udp_update_interval))
    safe_rerun()
