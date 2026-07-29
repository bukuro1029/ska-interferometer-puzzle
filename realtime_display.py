"""
Realtime UDP display for the SKA Interferometer Puzzle exhibit.

This script is intentionally separate from the Streamlit app. It runs a small
pygame window with its own 10 FPS render loop, so packet reception and drawing
are not limited by Streamlit reruns.

Examples:
    python -m pip install -r requirements-realtime.txt
    python realtime_display.py
    python realtime_display.py --fps 10
    python realtime_display.py --host 127.0.0.1 --port 9900 --mapping grid_mapping.csv
    python realtime_display.py --image /path/to/input.png
    python realtime_display.py --sample points
    python realtime_display.py --self-test

Pair with:
    python udp_sender_test.py --rows 8 --cols 8 --interval 0.2 --mode moving --seq

Runtime keys:
    H help, D toggle exhibit/science view, I cycle sample image, L reload --image,
    C cycle colors, W cycle reconstruction styles, F11 fullscreen, Q quit.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import socket
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


PacketItem = Tuple[float, str, str]
SAMPLE_MODELS = [
    "gas",
    "points",
    "bubbles",
    "ska",
    "double",
    "jet",
    "ring",
    "spiral",
    "cluster",
    "cross",
    "crescent",
    "resolution",
]
CMAPS = ["thermal", "icefire", "viridis", "RdBu_r", "seismic", "coolwarm", "gray"]
DISPLAY_MODES = ["exhibit", "science"]
EXHIBIT_THEMES = ["aurora", "ember", "tide", "violet", "mint", "mono", "coral"]
RECONSTRUCTION_STYLES = ["exhibit", "clean", "eht", "residual"]
DISPLAY_TRANSITION_SECONDS = 0.28
LAYOUT_PULSE_SECONDS = 0.42
EXHIBIT_LABELS = {
    "ja": {
        "headline": "SKA干渉計パズル  |  天体を再構成中",
        "antennas": "台のアンテナ",
        "baselines": "本の基線",
        "reconstruction": "再構成した画像",
        "layout": "アンテナのならび",
        "uv": "観測情報の分布 (UV)",
        "source": "元の画像",
        "stats": "いまの配列",
        "active_antennas": "アンテナ",
        "baseline_unit": "基線",
    },
    "en": {
        "headline": "SKA INTERFEROMETER PUZZLE  |  RECONSTRUCTING THE SKY",
        "antennas": "ACTIVE ANTENNAS",
        "baselines": "BASELINES",
        "reconstruction": "RECONSTRUCTED SKY",
        "layout": "ANTENNA LAYOUT",
        "uv": "SAMPLED INFORMATION (UV)",
        "source": "REFERENCE SKY",
        "stats": "ACTIVE ARRAY",
        "active_antennas": "ANTENNAS",
        "baseline_unit": "BASELINES",
    },
}
RECONSTRUCTION_STYLE_LABELS = {
    "ja": {
        "exhibit": "展示",
        "clean": "CLEAN",
        "eht": "EHT風",
        "residual": "残差",
    },
    "en": {
        "exhibit": "EXHIBIT",
        "clean": "CLEAN",
        "eht": "EHT-STYLE",
        "residual": "RESIDUAL",
    },
}


@dataclass
class ParsedPacket:
    contact_array: Optional[np.ndarray]
    rows: Optional[int]
    cols: Optional[int]
    seq: Optional[int]
    normalized: str
    error: Optional[str]


class UDPReceiver:
    """Background UDP receiver that stores packets until the render loop drains them."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9900, max_queue: int = 1000) -> None:
        self.host = host
        self.port = int(port)
        self.max_queue = int(max_queue)
        self._queue: Deque[PacketItem] = deque(maxlen=self.max_queue)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._socket: Optional[socket.socket] = None
        self.running = False
        self.total_received = 0
        self.last_packet = ""
        self.last_received_time: Optional[float] = None
        self.last_sender_address: Optional[str] = None
        self.last_error: Optional[str] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="RealtimeUDPReceiver", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.settimeout(0.02)
            self._socket = sock
            with self._lock:
                self.running = True
                self.last_error = None

            while not self._stop_event.is_set():
                try:
                    data, addr = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError as exc:
                    if not self._stop_event.is_set():
                        with self._lock:
                            self.last_error = str(exc)
                    break

                now = time.time()
                packet = data.decode("utf-8", errors="replace").strip()
                sender = f"{addr[0]}:{addr[1]}"
                with self._lock:
                    self._queue.append((now, packet, sender))
                    self.total_received += 1
                    self.last_packet = packet
                    self.last_received_time = now
                    self.last_sender_address = sender
        except OSError as exc:
            with self._lock:
                self.last_error = str(exc)
        finally:
            if self._socket is not None:
                try:
                    self._socket.close()
                except OSError:
                    pass
            with self._lock:
                self.running = False

    def drain_packets(self) -> List[PacketItem]:
        with self._lock:
            packets = list(self._queue)
            self._queue.clear()
            return packets

    def get_status(self) -> Dict[str, object]:
        with self._lock:
            return {
                "running": self.running,
                "queue_length": len(self._queue),
                "total_received": self.total_received,
                "last_packet": self.last_packet,
                "last_received_time": self.last_received_time,
                "last_sender_address": self.last_sender_address,
                "last_error": self.last_error,
            }

    def stop(self) -> None:
        self._stop_event.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=0.3)


def normalize_packet_text(packet: str) -> str:
    return " ".join(str(packet).replace("\n", " ").replace("\r", " ").split())


def parse_contact_packet(packet: str) -> ParsedPacket:
    normalized = normalize_packet_text(packet)
    if not normalized:
        return ParsedPacket(None, None, None, None, "", "empty packet")

    tokens = normalized.split()
    seq: Optional[int] = None
    seq_match = re.match(r"^seq(?:=|:)(\d+)$", tokens[0], flags=re.IGNORECASE)
    if seq_match:
        seq = int(seq_match.group(1))
        tokens = tokens[1:]
        normalized = " ".join([f"seq={seq}", *tokens])

    if not tokens:
        return ParsedPacket(None, None, None, seq, normalized, "no contact rows")

    cols = len(tokens[0])
    if cols == 0:
        return ParsedPacket(None, None, None, seq, normalized, "zero columns")

    for row in tokens:
        if len(row) != cols:
            return ParsedPacket(None, None, None, seq, normalized, "row lengths differ")
        if any(ch not in {"0", "1"} for ch in row):
            return ParsedPacket(None, None, None, seq, normalized, "packet contains non-0/1 characters")

    contact_array = np.array([[ch == "1" for ch in row] for row in tokens], dtype=bool)
    return ParsedPacket(contact_array, len(tokens), cols, seq, normalized, None)


def make_default_grid_mapping(rows: int, cols: int, max_baseline: float) -> Dict[Tuple[int, int], Tuple[float, float]]:
    rows = max(int(rows), 1)
    cols = max(int(cols), 1)
    half_side = max_baseline / (2.0 * math.sqrt(2.0))
    xs = np.linspace(-half_side, half_side, cols)
    ys = np.linspace(half_side, -half_side, rows)
    return {(r, c): (float(x), float(y)) for r, y in enumerate(ys) for c, x in enumerate(xs)}


def load_grid_mapping(
    path: str,
    rows: int,
    cols: int,
    max_baseline: float,
) -> Tuple[Dict[Tuple[int, int], Tuple[float, float]], str, Optional[str]]:
    mapping_path = Path(path)
    if not path or not mapping_path.exists():
        return make_default_grid_mapping(rows, cols, max_baseline), "default rectangular mapping", None

    try:
        with mapping_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            missing = {"row", "col", "x", "y"} - fieldnames
            if missing:
                warning = f"mapping missing columns {sorted(missing)}; using default mapping"
                return make_default_grid_mapping(rows, cols, max_baseline), "default rectangular mapping", warning

            mapping: Dict[Tuple[int, int], Tuple[float, float]] = {}
            for row in reader:
                enabled = int(float(row.get("enabled", "1") or 1))
                if enabled == 0:
                    continue
                r = int(row["row"])
                c = int(row["col"])
                mapping[(r, c)] = (float(row["x"]), float(row["y"]))

        if not mapping:
            return make_default_grid_mapping(rows, cols, max_baseline), "default rectangular mapping", "mapping is empty"
        return mapping, str(mapping_path), None
    except Exception as exc:  # noqa: BLE001
        warning = f"failed to load mapping: {exc}; using default mapping"
        return make_default_grid_mapping(rows, cols, max_baseline), "default rectangular mapping", warning


def positions_from_contacts(
    contact_array: Optional[np.ndarray],
    mapping: Dict[Tuple[int, int], Tuple[float, float]],
) -> Tuple[np.ndarray, int]:
    if contact_array is None:
        return np.empty((0, 2), dtype=float), 0

    positions: List[Tuple[float, float]] = []
    missing = 0
    for r, c in np.argwhere(contact_array):
        key = (int(r), int(c))
        if key in mapping:
            positions.append(mapping[key])
        else:
            missing += 1

    if not positions:
        return np.empty((0, 2), dtype=float), missing
    return np.asarray(positions, dtype=float), missing


def robust_normalize(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=float)
    lo, hi = np.percentile(data, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(data)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0)


def gaussian_2d(n: int, x0: float, y0: float, sx: float, sy: float, amp: float = 1.0) -> np.ndarray:
    y, x = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]
    return amp * np.exp(-0.5 * (((x - x0) / sx) ** 2 + ((y - y0) / sy) ** 2))


def make_sample_sky(n: int, seed: int = 42, model: str = "gas") -> np.ndarray:
    rng = np.random.default_rng(seed)
    sky = np.zeros((n, n), dtype=float)

    if model == "points":
        for _ in range(28):
            sky += gaussian_2d(
                n,
                rng.uniform(-0.88, 0.88),
                rng.uniform(-0.88, 0.88),
                rng.uniform(0.008, 0.020),
                rng.uniform(0.008, 0.020),
                rng.uniform(0.25, 1.2),
            )
    elif model == "bubbles":
        sky += gaussian_2d(n, 0.0, 0.0, 0.80, 0.80, 0.75)
        for _ in range(13):
            r = rng.uniform(0.07, 0.20)
            sky -= gaussian_2d(n, rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), r, r, rng.uniform(0.20, 0.65))
        sky -= sky.min()
    elif model == "ska":
        for t in np.linspace(-0.55, 0.55, 16):
            sky += gaussian_2d(n, -0.58 + 0.10 * np.sin(10 * t), t, 0.020, 0.020, 0.9)
            sky += gaussian_2d(n, -0.05, t, 0.018, 0.018, 0.9)
            sky += gaussian_2d(n, 0.12 + 0.24 * abs(t), t, 0.018, 0.018, 0.9)
            sky += gaussian_2d(n, 0.58 - 0.25 * t, t, 0.018, 0.018, 0.9)
            sky += gaussian_2d(n, 0.58 + 0.25 * t, t, 0.018, 0.018, 0.9)
        for x in np.linspace(0.43, 0.73, 12):
            sky += gaussian_2d(n, x, 0.0, 0.018, 0.018, 0.9)
    elif model == "double":
        sky += gaussian_2d(n, -0.32, 0.10, 0.038, 0.038, 1.0)
        sky += gaussian_2d(n, 0.34, -0.12, 0.050, 0.050, 0.82)
        sky += gaussian_2d(n, 0.02, -0.02, 0.42, 0.18, 0.20)
    elif model == "jet":
        sky += gaussian_2d(n, -0.03, 0.0, 0.055, 0.055, 1.25)
        sky += gaussian_2d(n, -0.36, 0.11, 0.18, 0.052, 0.56)
        sky += gaussian_2d(n, 0.36, -0.11, 0.20, 0.060, 0.70)
        sky += gaussian_2d(n, -0.66, 0.19, 0.075, 0.075, 0.45)
        sky += gaussian_2d(n, 0.67, -0.20, 0.085, 0.085, 0.52)
    elif model == "ring":
        for theta in np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False):
            sky += gaussian_2d(n, 0.43 * np.cos(theta), 0.43 * np.sin(theta), 0.040, 0.040, 0.42)
        sky += gaussian_2d(n, 0.0, 0.0, 0.12, 0.12, 0.14)
    elif model == "spiral":
        for arm in range(3):
            phase = arm * 2.0 * np.pi / 3.0
            for theta in np.linspace(0.15, 2.1 * np.pi, 32):
                radius = 0.07 + 0.070 * theta
                x0 = radius * np.cos(theta + phase)
                y0 = radius * np.sin(theta + phase)
                sky += gaussian_2d(n, x0, y0, 0.030, 0.030, 0.22)
        sky += gaussian_2d(n, 0.0, 0.0, 0.070, 0.070, 1.0)
    elif model == "cluster":
        sky += gaussian_2d(n, 0.0, 0.0, 0.32, 0.25, 0.38)
        for _ in range(18):
            sky += gaussian_2d(
                n,
                rng.normal(0.0, 0.32),
                rng.normal(0.0, 0.24),
                rng.uniform(0.018, 0.050),
                rng.uniform(0.018, 0.050),
                rng.uniform(0.20, 0.95),
            )
    elif model == "cross":
        sky += gaussian_2d(n, 0.0, 0.0, 0.045, 0.045, 1.2)
        for t in np.linspace(-0.62, 0.62, 17):
            if abs(t) > 0.10:
                sky += gaussian_2d(n, t, 0.0, 0.022, 0.022, 0.24)
                sky += gaussian_2d(n, 0.0, t, 0.022, 0.022, 0.24)
        sky += gaussian_2d(n, -0.66, 0.0, 0.07, 0.07, 0.58)
        sky += gaussian_2d(n, 0.66, 0.0, 0.07, 0.07, 0.58)
        sky += gaussian_2d(n, 0.0, -0.66, 0.07, 0.07, 0.58)
        sky += gaussian_2d(n, 0.0, 0.66, 0.07, 0.07, 0.58)
    elif model == "crescent":
        for theta in np.linspace(-0.75 * np.pi, 0.75 * np.pi, 44):
            x0 = 0.38 * np.cos(theta) + 0.10
            y0 = 0.38 * np.sin(theta)
            sky += gaussian_2d(n, x0, y0, 0.038, 0.038, 0.34 + 0.18 * np.cos(theta))
        sky += gaussian_2d(n, -0.14, 0.0, 0.12, 0.16, 0.20)
    elif model == "resolution":
        # Point-source pairs at several scales and orientations make array
        # configuration-dependent resolution differences easy to see.
        for x0, y0, separation, angle in [(-0.45, 0.34, 0.12, 0.0), (0.42, 0.30, 0.12, np.pi / 2), (0.0, -0.36, 0.15, np.pi / 4)]:
            dx = 0.5 * separation * np.cos(angle)
            dy = 0.5 * separation * np.sin(angle)
            sky += gaussian_2d(n, x0 - dx, y0 - dy, 0.026, 0.026, 1.0)
            sky += gaussian_2d(n, x0 + dx, y0 + dy, 0.026, 0.026, 0.78)
        sky += gaussian_2d(n, 0.0, 0.03, 0.18, 0.07, 0.30)
    else:
        sky += gaussian_2d(n, -0.20, 0.10, 0.35, 0.18, 1.0)
        sky += gaussian_2d(n, 0.30, -0.30, 0.16, 0.09, 0.65)
        for _ in range(12):
            sky += gaussian_2d(
                n,
                rng.uniform(-0.85, 0.85),
                rng.uniform(-0.85, 0.85),
                rng.uniform(0.010, 0.028),
                rng.uniform(0.010, 0.028),
                rng.uniform(0.25, 1.1),
            )
    return robust_normalize(sky)


def load_image_sky(path: Optional[str], n: int, sample: str = "gas") -> Tuple[np.ndarray, str]:
    if not path:
        return make_sample_sky(n, model=sample), f"sample:{sample}"
    try:
        from PIL import Image, ImageOps

        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("L")
        img.thumbnail((n, n))
        canvas = Image.new("L", (n, n), color=0)
        canvas.paste(img, ((n - img.size[0]) // 2, (n - img.size[1]) // 2))
        arr = np.asarray(canvas, dtype=float)
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        return arr, str(path)
    except Exception:
        return make_sample_sky(n, model=sample), f"sample:{sample} (image load failed)"


def add_disk(arr: np.ndarray, cx: int, cy: int, radius: int, value: float = 1.0) -> None:
    n = arr.shape[0]
    x0, x1 = max(0, cx - radius), min(n, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(n, cy + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    arr[y0:y1, x0:x1][disk] += value


def baselines(pos: np.ndarray) -> np.ndarray:
    if len(pos) < 2:
        return np.empty((0, 2), dtype=float)
    i, j = np.triu_indices(len(pos), k=1)
    diffs = pos[j] - pos[i]
    return np.vstack([diffs, -diffs])


def blur_preserve_scale(arr: np.ndarray, passes: int = 2) -> np.ndarray:
    """Small separable-like blur that preserves the image scale."""
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
    return out


def blur_array(arr: np.ndarray, passes: int = 2) -> np.ndarray:
    out = blur_preserve_scale(arr, passes)
    if out.max() > 0:
        out = out / out.max()
    return out


def make_uv_weight(
    pos: np.ndarray,
    grid: int,
    reference_baseline: float,
    uv_zoom: float,
    point_radius: int,
) -> np.ndarray:
    weight = np.zeros((grid, grid), dtype=float)
    bl = baselines(pos)
    if len(bl) == 0:
        return weight

    center = grid // 2
    scale = (0.45 * grid * uv_zoom) / max(reference_baseline, 1e-12)
    for u, v in bl:
        ix = int(round(center + u * scale))
        iy = int(round(center + v * scale))
        if 0 <= ix < grid and 0 <= iy < grid:
            add_disk(weight, ix, iy, max(1, point_radius), 1.0)

    if weight.max() > 0:
        weight = np.sqrt(weight)
        weight /= weight.max()
    return weight


def make_baseline_envelope(grid: int, max_baseline: float, reference_baseline: float, uv_zoom: float) -> np.ndarray:
    y, x = np.mgrid[0:grid, 0:grid]
    center = grid // 2
    rho = np.hypot(x - center, y - center)
    rmax = 0.45 * grid * uv_zoom * max_baseline / max(reference_baseline, 1e-12)
    rmax = np.clip(rmax, 1.0, 0.49 * grid)
    return np.exp(-((rho / rmax) ** 8))


def dirty_image_from_uv(sky: np.ndarray, effective_uv: np.ndarray) -> np.ndarray:
    sky_ref = np.asarray(sky, dtype=float) - np.mean(sky)
    sampled = np.fft.fftshift(np.fft.fft2(sky_ref)) * effective_uv
    dirty = np.real(np.fft.ifft2(np.fft.ifftshift(sampled)))
    dirty -= np.mean(dirty)
    return dirty


def reconstruct_dirty_image(
    sky: np.ndarray,
    pos: np.ndarray,
    max_baseline: float,
    reference_baseline: float,
    uv_zoom: float,
    point_radius: int,
    smooth_passes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(pos) < 2:
        zeros = np.zeros_like(sky)
        return zeros, zeros

    grid = sky.shape[0]
    sparse_uv = make_uv_weight(pos, grid, reference_baseline, uv_zoom, point_radius)
    envelope = make_baseline_envelope(grid, max_baseline, reference_baseline, uv_zoom)
    effective_uv = envelope * np.clip((0.82 * sparse_uv + 0.18 * blur_array(sparse_uv, smooth_passes)), 0.0, 1.0)
    return dirty_image_from_uv(sky, effective_uv), effective_uv


def reconstruct_exhibit_image(
    sky: np.ndarray,
    pos: np.ndarray,
    max_baseline: float,
    reference_baseline: float,
    uv_zoom: float,
    point_radius: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a denser display-only UV grid, then apply CLEAN-style restoration.

    The broad gridding kernel approximates combining nearby UV samples over a
    short observation. It is deliberately used only in exhibit mode; the
    science view exposes the sparse instantaneous dirty image instead.
    """
    if len(pos) < 2:
        zeros = np.zeros_like(sky)
        return zeros, zeros, zeros

    grid = sky.shape[0]
    sparse_uv = make_uv_weight(pos, grid, reference_baseline, uv_zoom, point_radius)
    envelope = make_baseline_envelope(grid, max_baseline, reference_baseline, uv_zoom)
    # Keep the individual UV samples dominant so that different antenna
    # configurations remain legible in the reconstructed exhibit image.
    display_uv = envelope * np.clip(0.55 * sparse_uv + 0.45 * blur_array(sparse_uv, passes=6), 0.0, 1.0)
    restored, residual = clean_style_products(dirty_image_from_uv(sky, display_uv), display_uv)
    return restored, display_uv, residual


def clean_style_reconstruction(
    dirty: np.ndarray,
    effective_uv: np.ndarray,
    gain: float = 0.12,
    max_iterations: int = 60,
    threshold_fraction: float = 0.10,
) -> np.ndarray:
    """Return the restored component of a compact Högbom-CLEAN-style image."""
    restored, _ = clean_style_products(dirty, effective_uv, gain, max_iterations, threshold_fraction)
    return restored


def clean_style_products(
    dirty: np.ndarray,
    effective_uv: np.ndarray,
    gain: float = 0.12,
    max_iterations: int = 60,
    threshold_fraction: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a positive restored image and the signed CLEAN residual.

    The science view keeps the raw dirty image. This restoration uses the same
    dirty image and its point-spread function, then suppresses strong sidelobes
    before a small restoring blur is applied. The residual is retained for the
    diagnostic display style, rather than being used to enhance the exhibit.
    """
    residual = np.asarray(dirty, dtype=float).copy()
    peak_scale = float(np.max(np.abs(residual)))
    if peak_scale <= 1e-12 or not np.any(effective_uv):
        return np.zeros_like(residual), residual

    psf = np.real(np.fft.ifft2(np.fft.ifftshift(effective_uv)))
    psf_peak = float(psf[0, 0])
    if abs(psf_peak) <= 1e-12:
        return np.zeros_like(residual), residual
    psf /= psf_peak

    components = np.zeros_like(residual)
    threshold = peak_scale * max(float(threshold_fraction), 0.0)
    clean_gain = np.clip(float(gain), 0.01, 0.95)
    for _ in range(max(0, int(max_iterations))):
        iy, ix = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        component = float(residual[iy, ix])
        if abs(component) < threshold:
            break
        component *= clean_gain
        components[iy, ix] += component
        shifted_psf = np.roll(np.roll(psf, iy, axis=0), ix, axis=1)
        residual -= component * shifted_psf

    restored = blur_preserve_scale(components, passes=2) + 0.08 * blur_preserve_scale(residual, passes=1)
    # Exhibit view focuses on positive recovered brightness. The science view
    # still exposes the signed dirty image, including all sidelobes.
    background = float(np.percentile(restored, 60.0))
    return np.maximum(restored - background, 0.0), residual


def asinh_stretch_signed(arr: np.ndarray, percentile: float = 99.0, stretch: float = 4.0) -> np.ndarray:
    data = np.asarray(arr, dtype=float)
    scale = float(np.percentile(np.abs(data), percentile))
    scale = max(scale, 1e-12)
    stretch = max(float(stretch), 1e-6)
    return np.clip(np.arcsinh(stretch * data / scale) / np.arcsinh(stretch), -1.0, 1.0)


def diverging_rgb(values: np.ndarray, cmap: str = "thermal") -> np.ndarray:
    v = np.clip(values, -1.0, 1.0)
    rgb = np.empty((*v.shape, 3), dtype=float)

    palettes = {
        "thermal": ((39, 55, 99), (13, 17, 24), (250, 190, 86)),
        "icefire": ((48, 108, 170), (18, 20, 26), (232, 103, 76)),
        "viridis": ((68, 1, 84), (35, 137, 142), (253, 231, 37)),
        "RdBu_r": ((49, 130, 189), (247, 247, 247), (202, 0, 32)),
        "seismic": ((0, 0, 160), (255, 255, 255), (160, 0, 0)),
        "coolwarm": ((76, 114, 176), (238, 238, 238), (196, 78, 82)),
    }
    if cmap == "gray":
        gray = (v + 1.0) * 0.5
        return np.repeat((255.0 * gray)[..., None], 3, axis=2).astype(np.uint8)

    neg, mid, pos = [np.array(c, dtype=float) for c in palettes.get(cmap, palettes["RdBu_r"])]
    neg_mask = v < 0
    t_neg = (v[neg_mask] + 1.0)[..., None]
    t_pos = v[~neg_mask][..., None]
    rgb[neg_mask] = neg * (1.0 - t_neg) + mid * t_neg
    rgb[~neg_mask] = mid * (1.0 - t_pos) + pos * t_pos
    return np.clip(rgb, 0, 255).astype(np.uint8)


def scalar_to_rgb(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=float)
    if data.max() > data.min():
        norm = (data - data.min()) / (data.max() - data.min())
    else:
        norm = np.zeros_like(data)
    rgb = np.zeros((*data.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (35 + 180 * norm).astype(np.uint8)
    rgb[..., 1] = (55 + 180 * norm).astype(np.uint8)
    rgb[..., 2] = (80 + 150 * norm).astype(np.uint8)
    return rgb


def _three_stop_rgb(
    values: np.ndarray,
    low: Tuple[int, int, int],
    mid: Tuple[int, int, int],
    high: Tuple[int, int, int],
) -> np.ndarray:
    """Map normalized 0-1 values to a three-stop display palette."""
    v = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    rgb = np.empty((*v.shape, 3), dtype=float)
    low_rgb, mid_rgb, high_rgb = [np.asarray(color, dtype=float) for color in (low, mid, high)]
    lower = v <= 0.55
    lower_t = (v[lower] / 0.55)[..., None]
    upper_t = ((v[~lower] - 0.55) / 0.45)[..., None]
    rgb[lower] = low_rgb * (1.0 - lower_t) + mid_rgb * lower_t
    rgb[~lower] = mid_rgb * (1.0 - upper_t) + high_rgb * upper_t
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _normalize_scalar(arr: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    data = np.asarray(arr, dtype=float)
    lo, hi = np.percentile(data, [1.0, 99.5])
    if hi <= lo:
        return np.zeros_like(data)
    norm = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    return norm ** max(float(gamma), 1e-6)


def exhibit_scalar_rgb(arr: np.ndarray, kind: str, theme: str = "aurora") -> np.ndarray:
    """Display-only palettes for a dark, legible exhibit screen."""
    palettes = {
        "aurora": {
            "sky": ((4, 12, 29), (25, 119, 166), (239, 251, 255)),
            "uv": ((5, 14, 30), (20, 146, 151), (235, 255, 208)),
        },
        "ember": {
            "sky": ((18, 8, 25), (150, 49, 97), (255, 226, 183)),
            "uv": ((20, 9, 24), (192, 77, 48), (255, 238, 174)),
        },
        "tide": {
            "sky": ((3, 16, 30), (27, 107, 180), (215, 248, 255)),
            "uv": ((2, 19, 33), (47, 167, 184), (204, 255, 238)),
        },
        "violet": {
            "sky": ((15, 7, 34), (114, 65, 184), (246, 231, 255)),
            "uv": ((16, 6, 38), (141, 78, 204), (244, 220, 255)),
        },
        "mint": {
            "sky": ((3, 21, 20), (22, 150, 124), (221, 255, 240)),
            "uv": ((2, 25, 23), (26, 179, 150), (214, 255, 234)),
        },
        "mono": {
            "sky": ((10, 13, 18), (112, 131, 151), (248, 250, 252)),
            "uv": ((8, 12, 16), (124, 148, 166), (244, 248, 250)),
        },
        "coral": {
            "sky": ((27, 10, 19), (196, 81, 106), (255, 231, 214)),
            "uv": ((26, 9, 17), (217, 96, 111), (255, 230, 201)),
        },
    }
    colors = palettes.get(theme, palettes["aurora"])[kind]
    gamma = 0.70 if kind == "sky" else 0.58
    return _three_stop_rgb(_normalize_scalar(arr, gamma=gamma), *colors)


def exhibit_dirty_rgb(values: np.ndarray, theme: str = "aurora") -> np.ndarray:
    """Render the positive CLEAN-style exhibit reconstruction on a dark field."""
    palettes = {
        "aurora": ((5, 11, 25), (241, 149, 52), (255, 247, 213)),
        "ember": ((16, 7, 24), (244, 89, 47), (255, 235, 181)),
        "tide": ((3, 14, 28), (74, 213, 188), (231, 255, 232)),
        "violet": ((14, 6, 30), (184, 111, 245), (250, 237, 255)),
        "mint": ((3, 18, 19), (63, 220, 172), (229, 255, 243)),
        "mono": ((9, 12, 17), (174, 190, 204), (255, 255, 255)),
        "coral": ((25, 8, 16), (248, 122, 99), (255, 238, 214)),
    }
    zero, positive, highlight = palettes.get(theme, palettes["aurora"])
    signal = np.maximum(np.asarray(values, dtype=float), 0.0)
    signal = np.clip((signal - 0.08) / 0.92, 0.0, 1.0) ** 0.72
    return _three_stop_rgb(signal, zero, positive, highlight)


def clean_restored_rgb(values: np.ndarray) -> np.ndarray:
    """Neutral restored-image palette used in standard radio-imaging figures."""
    signal = _normalize_scalar(np.maximum(values, 0.0), gamma=0.62)
    return _three_stop_rgb(signal, (8, 11, 16), (118, 132, 145), (255, 255, 250))


def eht_style_rgb(values: np.ndarray) -> np.ndarray:
    """Warm, high-dynamic-range display inspired by public EHT image releases."""
    signal = _normalize_scalar(np.maximum(values, 0.0), gamma=0.42)
    signal = np.clip((signal - 0.025) / 0.975, 0.0, 1.0)
    return _three_stop_rgb(signal, (0, 0, 0), (198, 45, 12), (255, 242, 190))


def render_exhibit_reconstruction_rgb(
    reconstruction: np.ndarray,
    residual: np.ndarray,
    style: str,
    theme: str,
    contrast_percentile: float,
    stretch: float,
) -> np.ndarray:
    """Select a display treatment without altering the reconstruction itself."""
    if style == "clean":
        return clean_restored_rgb(reconstruction)
    if style == "eht":
        return eht_style_rgb(reconstruction)
    if style == "residual":
        return diverging_rgb(asinh_stretch_signed(residual, contrast_percentile, stretch), "icefire")
    return exhibit_dirty_rgb(asinh_stretch_signed(reconstruction, contrast_percentile, stretch), theme)


def render_dirty_rgb(
    dirty: np.ndarray,
    display_mode: str,
    cmap: str,
    exhibit_theme: str,
    contrast_percentile: float,
    stretch: float,
) -> np.ndarray:
    stretched = asinh_stretch_signed(dirty, contrast_percentile, stretch)
    if display_mode == "exhibit":
        return exhibit_dirty_rgb(stretched, exhibit_theme)
    return diverging_rgb(stretched, cmap)


def render_sky_rgb(sky: np.ndarray, display_mode: str, exhibit_theme: str) -> np.ndarray:
    if display_mode == "exhibit":
        return exhibit_scalar_rgb(sky, "sky", exhibit_theme)
    return scalar_to_rgb(sky)


def render_uv_rgb(uv: np.ndarray, display_mode: str, exhibit_theme: str) -> np.ndarray:
    if display_mode == "exhibit":
        return exhibit_scalar_rgb(uv, "uv", exhibit_theme)
    return scalar_to_rgb(uv)


def blend_rgb(first: np.ndarray, second: np.ndarray, amount: float) -> np.ndarray:
    t = np.clip(float(amount), 0.0, 1.0)
    return np.clip(first.astype(float) * (1.0 - t) + second.astype(float) * t, 0, 255).astype(np.uint8)


def resize_array_nearest(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    y_idx = np.linspace(0, arr.shape[0] - 1, height).astype(int)
    x_idx = np.linspace(0, arr.shape[1] - 1, width).astype(int)
    return arr[np.ix_(y_idx, x_idx)]


def draw_text(surface, font, text: str, x: int, y: int, color=(235, 238, 241)) -> None:
    rendered = font.render(text, True, color)
    surface.blit(rendered, (x, y))


def font_supports_japanese(pygame, font_path: str) -> bool:
    """Check whether a font renders Japanese glyphs rather than replacement boxes."""
    try:
        font = pygame.font.Font(font_path, 24)
        sample = font.render("アンテナ", True, (255, 255, 255))
        missing = font.render("□□□□", True, (255, 255, 255))
        return sample.get_size() != missing.get_size() or pygame.image.tostring(sample, "RGBA") != pygame.image.tostring(missing, "RGBA")
    except Exception:  # noqa: BLE001
        return False


def resolve_ui_font(pygame, requested: Optional[str]) -> Tuple[Optional[str], bool]:
    """Find a CJK font by file path; pygame's font-name lookup misses some macOS fonts."""
    candidates: List[Path] = []
    if requested:
        requested_path = Path(requested).expanduser()
        if requested_path.exists():
            candidates.append(requested_path)
        else:
            matched = pygame.font.match_font(requested)
            if matched:
                candidates.append(Path(matched))

    candidates.extend(
        Path(path)
        for path in (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKJP-Regular.otf",
            "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
            "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        )
    )

    mac_font_dir = Path("/System/Library/Fonts")
    if mac_font_dir.exists():
        candidates.extend(
            path
            for path in sorted(mac_font_dir.glob("*.ttc"))
            if "ゴシック" in unicodedata.normalize("NFC", path.name)
        )

    fallback_source: Optional[str] = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        source = str(candidate)
        if fallback_source is None:
            fallback_source = source
        if font_supports_japanese(pygame, source):
            return source, True
    return fallback_source, False


def make_ui_font(pygame, source: Optional[str], size: int):
    return pygame.font.Font(source, size) if source else pygame.font.Font(None, size)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def square_panel_rects(width: int, height: int, top_h: int, margin: int) -> List[Tuple[int, int, int, int]]:
    content_w = max(1, int(width) - 3 * margin)
    content_h = max(1, int(height) - top_h - 3 * margin)
    side = max(1, min(content_w // 2, content_h // 2))
    total_w = 2 * side + margin
    total_h = 2 * side + margin
    start_x = max(margin, (int(width) - total_w) // 2)
    start_y = top_h + max(margin, (int(height) - top_h - total_h) // 2)
    return [
        (start_x, start_y, side, side),
        (start_x + side + margin, start_y, side, side),
        (start_x, start_y + side + margin, side, side),
        (start_x + side + margin, start_y + side + margin, side, side),
    ]


def exhibit_panel_rects(width: int, height: int, top_h: int, margin: int) -> Dict[str, Tuple[int, int, int, int]]:
    """One large square reconstruction panel plus four square supporting panels."""
    content_w = max(1, int(width) - 2 * margin)
    content_h = max(1, int(height) - top_h - 2 * margin)
    small_side = max(1, min((content_w - 3 * margin) // 4, (content_h - margin) // 2))
    large_side = 2 * small_side + margin
    block_w = 4 * small_side + 3 * margin
    block_h = 2 * small_side + margin
    start_x = max(margin, (int(width) - block_w) // 2)
    start_y = top_h + max(margin, (int(height) - top_h - block_h) // 2)
    right_x = start_x + large_side + margin
    lower_y = start_y + small_side + margin
    return {
        "reconstruction": (start_x, start_y, large_side, large_side),
        "layout": (right_x, start_y, small_side, small_side),
        "uv": (right_x + small_side + margin, start_y, small_side, small_side),
        "source": (right_x, lower_y, small_side, small_side),
        "stats": (right_x + small_side + margin, lower_y, small_side, small_side),
    }


def square_inner_rect(
    rect,
    top_pad: int = 36,
    bottom_pad: int = 30,
    side_pad: int = 12,
) -> Tuple[int, int, int, int]:
    area_x = rect.x + side_pad
    area_y = rect.y + top_pad
    area_w = max(1, rect.w - 2 * side_pad)
    area_h = max(1, rect.h - top_pad - bottom_pad)
    side = max(1, min(area_w, area_h))
    return area_x + (area_w - side) // 2, area_y + (area_h - side) // 2, side, side


def draw_help_overlay(pygame, screen, font, small_font, args: argparse.Namespace, image_source: str) -> None:
    panel = pygame.Rect(44, 72, screen.get_width() - 88, screen.get_height() - 144)
    pygame.draw.rect(screen, (18, 21, 27), panel)
    pygame.draw.rect(screen, (105, 115, 130), panel, 1)
    x = panel.x + 20
    y = panel.y + 18
    lines = [
        "Realtime Display Help",
        "",
        "Input image:",
        f"  current: {image_source}",
        "  external file: python realtime_display.py --image /path/to/input.png",
        f"  built-in sample: use I to cycle (current default: {args.sample})",
        "",
        "Runtime keys:",
        "  H: show/hide this help    Q: quit",
        "  F11: fullscreen/window    Esc: leave fullscreen or quit window",
        "  D: switch exhibit/science view",
        "  I: cycle built-in sample image    L: reload --image file",
        "  C: cycle exhibit color theme or science colormap",
        "  W: cycle reconstructed-image style (exhibit / CLEAN / EHT / residual)",
        "  Z/X: decrease/increase asinh stretch",
        "  N/M: decrease/increase contrast percentile",
        "  B/V: decrease/increase max baseline",
        "  R/T: decrease/increase reference baseline",
        "  U/J: increase/decrease uv zoom",
        "  O/P: decrease/increase uv point size",
        "  A/S: decrease/increase uv smoothing",
        "",
        "This Pygame version is intentionally for fast exhibit display.",
        "Streamlit remains the richer adjustment/prototype interface.",
    ]
    for i, line in enumerate(lines):
        draw_text(screen, font if i == 0 else small_font, line, x, y)
        y += 25 if i == 0 else 21


def draw_contour_overlay(pygame, screen, inner, values: np.ndarray, color=(213, 227, 229)) -> None:
    """Draw lightweight marching-squares contours for the CLEAN display."""
    positive = np.maximum(np.asarray(values, dtype=float), 0.0)
    scale = float(np.percentile(positive, 99.5))
    if scale <= 1e-12:
        return
    sample_side = min(72, positive.shape[0], positive.shape[1])
    sampled = resize_array_nearest(positive / scale, sample_side, sample_side)
    levels = (0.20, 0.38, 0.58, 0.80)
    step_x = (inner.w - 1) / max(sample_side - 1, 1)
    step_y = (inner.h - 1) / max(sample_side - 1, 1)
    for level in levels:
        for iy in range(sample_side - 1):
            for ix in range(sample_side - 1):
                corners = (
                    float(sampled[iy, ix]),
                    float(sampled[iy, ix + 1]),
                    float(sampled[iy + 1, ix + 1]),
                    float(sampled[iy + 1, ix]),
                )
                edges = ((0, 1), (1, 2), (2, 3), (3, 0))
                edge_points = []
                for first, second in edges:
                    a, b = corners[first], corners[second]
                    if (a >= level) == (b >= level):
                        continue
                    delta = b - a
                    if abs(delta) <= 1e-12:
                        continue
                    fraction = (level - a) / delta
                    corner_points = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
                    ax, ay = corner_points[first]
                    bx, by = corner_points[second]
                    edge_points.append(
                        (
                            int(inner.x + (ix + ax + fraction * (bx - ax)) * step_x),
                            int(inner.y + (iy + ay + fraction * (by - ay)) * step_y),
                        )
                    )
                if len(edge_points) == 2:
                    pygame.draw.line(screen, color, edge_points[0], edge_points[1], 1)
                elif len(edge_points) == 4:
                    pygame.draw.line(screen, color, edge_points[0], edge_points[1], 1)
                    pygame.draw.line(screen, color, edge_points[2], edge_points[3], 1)


def draw_restoring_beam(pygame, screen, inner, uv: np.ndarray) -> None:
    """Draw a small synthesized-beam marker, as used on radio image figures."""
    weight = np.maximum(np.asarray(uv, dtype=float), 0.0)
    total = float(weight.sum())
    if total <= 1e-12:
        return
    yy, xx = np.mgrid[0 : weight.shape[0], 0 : weight.shape[1]]
    xx = xx - (weight.shape[1] - 1) / 2.0
    yy = yy - (weight.shape[0] - 1) / 2.0
    covariance = np.array(
        [
            [float((weight * xx * xx).sum() / total), float((weight * xx * yy).sum() / total)],
            [float((weight * xx * yy).sum() / total), float((weight * yy * yy).sum() / total)],
        ]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-9)
    aspect = clamp(math.sqrt(float(eigenvalues[1] / eigenvalues[0])), 1.0, 2.5)
    base = max(16.0, 0.105 * min(inner.w, inner.h))
    major = base * math.sqrt(aspect)
    minor = base / math.sqrt(aspect)
    direction = eigenvectors[:, 0]
    perpendicular = np.array((-direction[1], direction[0]))
    center = np.array((inner.x + 0.13 * inner.w, inner.bottom - 0.13 * inner.h))
    angles = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
    points = [
        tuple(
            np.rint(
                center
                + 0.5 * major * math.cos(angle) * direction
                + 0.5 * minor * math.sin(angle) * perpendicular
            ).astype(int)
        )
        for angle in angles
    ]
    pygame.draw.polygon(screen, (210, 222, 217), points)
    pygame.draw.polygon(screen, (20, 28, 31), points, 1)


def draw_image_panel(
    pygame,
    screen,
    rect,
    title: str,
    rgb: np.ndarray,
    font,
    accent: Optional[Tuple[int, int, int]] = None,
    smooth: bool = False,
    contours: Optional[np.ndarray] = None,
    show_restoring_beam: bool = False,
    beam_uv: Optional[np.ndarray] = None,
) -> None:
    background = (10, 17, 29) if accent is not None else (22, 25, 30)
    border = accent if accent is not None else (80, 88, 100)
    pygame.draw.rect(screen, background, rect)
    if accent is not None:
        pygame.draw.rect(screen, accent, (rect.x, rect.y, rect.w, 3))
    draw_text(screen, font, title, rect.x + 10, rect.y + 8)
    inner = pygame.Rect(*square_inner_rect(rect))
    if smooth:
        native = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        surf = pygame.transform.smoothscale(native, (inner.w, inner.h))
    else:
        resized = resize_array_nearest(rgb, inner.w, inner.h)
        surf = pygame.surfarray.make_surface(np.transpose(resized, (1, 0, 2)))
    screen.blit(surf, inner)
    if contours is not None:
        draw_contour_overlay(pygame, screen, inner, contours)
    if show_restoring_beam and beam_uv is not None:
        draw_restoring_beam(pygame, screen, inner, beam_uv)
    pygame.draw.rect(screen, border, rect, 1)


def draw_layout_panel(
    pygame,
    screen,
    rect,
    title: str,
    pos: np.ndarray,
    radius: float,
    font,
    accent: Optional[Tuple[int, int, int]] = None,
    pulse_strength: float = 0.0,
) -> None:
    background = (10, 17, 29) if accent is not None else (22, 25, 30)
    border = accent if accent is not None else (80, 88, 100)
    pygame.draw.rect(screen, background, rect)
    if accent is not None:
        pygame.draw.rect(screen, accent, (rect.x, rect.y, rect.w, 3))
    draw_text(screen, font, title, rect.x + 10, rect.y + 8)
    inner = pygame.Rect(*square_inner_rect(rect, top_pad=42, bottom_pad=24, side_pad=18))
    pygame.draw.rect(screen, (8, 10, 14), inner)
    center = (inner.centerx, inner.centery)
    scale = 0.45 * min(inner.w, inner.h) / max(radius, 1e-12)
    pygame.draw.circle(screen, (92, 102, 116), center, int(radius * scale), 1)
    pygame.draw.line(screen, (45, 50, 60), (inner.left, inner.centery), (inner.right, inner.centery), 1)
    pygame.draw.line(screen, (45, 50, 60), (inner.centerx, inner.top), (inner.centerx, inner.bottom), 1)
    for x, y in pos:
        px = int(center[0] + x * scale)
        py = int(center[1] - y * scale)
        if pulse_strength > 0:
            pulse_radius = int(7 + 7 * (1.0 - pulse_strength))
            pygame.draw.circle(screen, (255, 225, 128), (px, py), pulse_radius, 1)
        pygame.draw.circle(screen, (246, 197, 67), (px, py), 5)
    pygame.draw.rect(screen, border, rect, 1)


def draw_live_stats_panel(
    pygame,
    screen,
    rect,
    antennas: int,
    font,
    small_font,
    big_font,
    accent: Tuple[int, int, int],
    labels: Dict[str, str],
) -> None:
    pygame.draw.rect(screen, (10, 17, 29), rect)
    pygame.draw.rect(screen, accent, (rect.x, rect.y, rect.w, 3))
    draw_text(screen, font, labels["stats"], rect.x + 10, rect.y + 8)
    pygame.draw.circle(screen, (102, 227, 163), (rect.right - 20, rect.y + 19), 5)
    number = str(max(0, antennas))
    rendered = big_font.render(number, True, (250, 204, 94))
    screen.blit(rendered, (rect.centerx - rendered.get_width() // 2, rect.y + rect.h // 3 - rendered.get_height() // 2))
    antenna_label = small_font.render(labels["active_antennas"], True, (220, 229, 238))
    screen.blit(antenna_label, (rect.centerx - antenna_label.get_width() // 2, rect.y + rect.h // 2 + 16))
    baseline_label = f"{antennas * max(0, antennas - 1) // 2} {labels['baseline_unit']}"
    baseline_text = small_font.render(baseline_label, True, (147, 211, 210))
    screen.blit(baseline_text, (rect.centerx - baseline_text.get_width() // 2, rect.bottom - 33))
    pygame.draw.rect(screen, accent, rect, 1)


def run_self_test() -> None:
    parsed = parse_contact_packet("SEQ:12 101 010 101")
    assert parsed.error is None
    assert parsed.seq == 12
    assert parsed.contact_array is not None
    mapping = make_default_grid_mapping(parsed.rows or 3, parsed.cols or 3, 20.0)
    pos, missing = positions_from_contacts(parsed.contact_array, mapping)
    assert len(pos) == 5
    assert missing == 0
    sky = make_sample_sky(64, model="points")
    dirty, uv = reconstruct_dirty_image(sky, pos, 20.0, 20.0, 1.0, 1, 1)
    assert dirty.shape == (64, 64)
    assert uv.shape == (64, 64)
    restored = clean_style_reconstruction(dirty, uv)
    assert restored.shape == (64, 64)
    assert np.all(np.isfinite(restored))
    exhibit_image, exhibit_uv, exhibit_residual = reconstruct_exhibit_image(sky, pos, 20.0, 20.0, 1.0, 1)
    assert exhibit_image.shape == (64, 64)
    assert exhibit_uv.shape == (64, 64)
    assert exhibit_residual.shape == (64, 64)
    for model in SAMPLE_MODELS:
        assert make_sample_sky(64, model=model).shape == (64, 64)
    assert exhibit_dirty_rgb(asinh_stretch_signed(dirty)).shape == (64, 64, 3)
    assert exhibit_scalar_rgb(uv, "uv").shape == (64, 64, 3)
    for style in RECONSTRUCTION_STYLES:
        assert render_exhibit_reconstruction_rgb(exhibit_image, exhibit_residual, style, "aurora", 99.0, 4.0).shape == (64, 64, 3)
    rects = square_panel_rects(1040, 960, 96, 12)
    assert len(rects) == 4
    assert all(w == h for _, _, w, h in rects)
    exhibit_rects = exhibit_panel_rects(1040, 960, 96, 12)
    assert set(exhibit_rects) == {"reconstruction", "layout", "uv", "source", "stats"}
    assert all(w == h for _, _, w, h in exhibit_rects.values())
    print("self-test ok")


def run_display(args: argparse.Namespace) -> None:
    import pygame

    pygame.init()
    fullscreen = bool(args.fullscreen)
    window_size = (args.width, args.height)

    def make_screen(use_fullscreen: bool):
        if use_fullscreen:
            return pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        return pygame.display.set_mode(window_size, pygame.RESIZABLE)

    screen = make_screen(fullscreen)
    pygame.display.set_caption("SKA Interferometer Puzzle - Realtime UDP Display")
    font_source, japanese_ui = resolve_ui_font(pygame, args.font)
    font = make_ui_font(pygame, font_source, 18)
    small_font = make_ui_font(pygame, font_source, 15)
    title_font = make_ui_font(pygame, font_source, 24)
    stats_font = make_ui_font(pygame, font_source, 58)
    exhibit_labels = EXHIBIT_LABELS["ja" if japanese_ui else "en"]
    clock = pygame.time.Clock()

    sample_index = SAMPLE_MODELS.index(args.sample) if args.sample in SAMPLE_MODELS else 0
    cmap_index = CMAPS.index(args.cmap) if args.cmap in CMAPS else 0
    display_mode = args.display_mode
    exhibit_theme_index = EXHIBIT_THEMES.index(args.exhibit_theme) if args.exhibit_theme in EXHIBIT_THEMES else 0
    reconstruction_style_index = (
        RECONSTRUCTION_STYLES.index(args.reconstruction_style)
        if args.reconstruction_style in RECONSTRUCTION_STYLES
        else 0
    )
    max_baseline = float(args.max_baseline)
    reference_baseline = float(args.reference_baseline)
    uv_zoom = float(args.uv_zoom)
    point_radius = int(args.point_radius)
    smooth_passes = int(args.smooth_passes)
    contrast_percentile = float(args.contrast_percentile)
    stretch = float(args.stretch)
    show_help = bool(args.show_help)

    sky, image_source = load_image_sky(args.image, args.grid, SAMPLE_MODELS[sample_index])
    mapping, mapping_source, mapping_warning = load_grid_mapping(args.mapping, args.rows, args.cols, max_baseline)
    receiver = UDPReceiver(args.host, args.port, args.max_queue)
    receiver.start()

    pos = np.empty((0, 2), dtype=float)
    dirty = np.zeros((args.grid, args.grid), dtype=float)
    uv = np.zeros((args.grid, args.grid), dtype=float)
    showcase_reconstruction = np.zeros((args.grid, args.grid), dtype=float)
    showcase_residual = np.zeros((args.grid, args.grid), dtype=float)
    dirty_rgb = render_exhibit_reconstruction_rgb(
        showcase_reconstruction,
        showcase_residual,
        RECONSTRUCTION_STYLES[reconstruction_style_index],
        EXHIBIT_THEMES[exhibit_theme_index],
        contrast_percentile,
        stretch,
    )
    if display_mode == "science":
        dirty_rgb = render_dirty_rgb(dirty, display_mode, CMAPS[cmap_index], EXHIBIT_THEMES[exhibit_theme_index], contrast_percentile, stretch)
    previous_dirty_rgb = dirty_rgb.copy()
    uv_rgb = render_uv_rgb(uv, display_mode, EXHIBIT_THEMES[exhibit_theme_index])
    sky_rgb = render_sky_rgb(sky, display_mode, EXHIBIT_THEMES[exhibit_theme_index])
    latest_contact_array: Optional[np.ndarray] = None
    latest_packet = ""
    latest_seq: Optional[int] = None
    latest_shape = "-"
    latest_sender = "-"
    latest_error = ""
    missing_count = 0
    valid_count = 0
    parse_errors = 0
    drained_this_frame = 0
    last_recompute = time.time()
    last_visual_change = 0.0
    last_layout_change = 0.0
    frame_count = 0
    running = True

    def refresh_visuals(animate: bool = False) -> None:
        nonlocal dirty_rgb, previous_dirty_rgb, uv_rgb, sky_rgb, last_visual_change
        now = time.time()
        if animate:
            elapsed = now - last_visual_change
            if 0.0 < elapsed < DISPLAY_TRANSITION_SECONDS:
                previous_dirty_rgb = blend_rgb(previous_dirty_rgb, dirty_rgb, elapsed / DISPLAY_TRANSITION_SECONDS)
            else:
                previous_dirty_rgb = dirty_rgb.copy()

        theme = EXHIBIT_THEMES[exhibit_theme_index]
        # The UV panel always shows the instantaneous measurement pattern,
        # rather than the smoother display-only grid used for reconstruction.
        uv_to_render = uv
        if display_mode == "exhibit":
            dirty_rgb = render_exhibit_reconstruction_rgb(
                showcase_reconstruction,
                showcase_residual,
                RECONSTRUCTION_STYLES[reconstruction_style_index],
                theme,
                contrast_percentile,
                stretch,
            )
        else:
            dirty_rgb = render_dirty_rgb(dirty, display_mode, CMAPS[cmap_index], theme, contrast_percentile, stretch)
        uv_rgb = render_uv_rgb(uv_to_render, display_mode, theme)
        sky_rgb = render_sky_rgb(sky, display_mode, theme)
        if animate:
            last_visual_change = now

    try:
        while running:
            recompute_needed = False
            image_changed = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE and not fullscreen:
                    window_size = event.size
                    screen = make_screen(False)
                elif event.type == pygame.KEYDOWN:
                    key = event.key
                    if key == pygame.K_q:
                        running = False
                    elif key == pygame.K_ESCAPE:
                        if fullscreen:
                            fullscreen = False
                            screen = make_screen(False)
                        else:
                            running = False
                    elif key == pygame.K_F11:
                        fullscreen = not fullscreen
                        screen = make_screen(fullscreen)
                    elif key == pygame.K_h:
                        show_help = not show_help
                    elif key == pygame.K_d:
                        display_mode = "science" if display_mode == "exhibit" else "exhibit"
                        refresh_visuals(animate=True)
                    elif key == pygame.K_i:
                        sample_index = (sample_index + 1) % len(SAMPLE_MODELS)
                        sky, image_source = load_image_sky(None, args.grid, SAMPLE_MODELS[sample_index])
                        image_changed = True
                    elif key == pygame.K_l:
                        sky, image_source = load_image_sky(args.image, args.grid, SAMPLE_MODELS[sample_index])
                        image_changed = True
                    elif key == pygame.K_c:
                        if display_mode == "exhibit":
                            exhibit_theme_index = (exhibit_theme_index + 1) % len(EXHIBIT_THEMES)
                        else:
                            cmap_index = (cmap_index + 1) % len(CMAPS)
                        refresh_visuals(animate=True)
                    elif key == pygame.K_w:
                        reconstruction_style_index = (reconstruction_style_index + 1) % len(RECONSTRUCTION_STYLES)
                        refresh_visuals(animate=True)
                    elif key == pygame.K_z:
                        stretch = clamp(stretch - 0.5, 1.0, 10.0)
                        refresh_visuals(animate=True)
                    elif key == pygame.K_x:
                        stretch = clamp(stretch + 0.5, 1.0, 10.0)
                        refresh_visuals(animate=True)
                    elif key == pygame.K_n:
                        contrast_percentile = clamp(contrast_percentile - 0.5, 90.0, 99.9)
                        refresh_visuals(animate=True)
                    elif key == pygame.K_m:
                        contrast_percentile = clamp(contrast_percentile + 0.5, 90.0, 99.9)
                        refresh_visuals(animate=True)
                    elif key == pygame.K_b:
                        max_baseline = clamp(max_baseline - 5.0, 1.0, 100000.0)
                        mapping, mapping_source, mapping_warning = load_grid_mapping(args.mapping, args.rows, args.cols, max_baseline)
                        recompute_needed = True
                    elif key == pygame.K_v:
                        max_baseline = clamp(max_baseline + 5.0, 1.0, 100000.0)
                        mapping, mapping_source, mapping_warning = load_grid_mapping(args.mapping, args.rows, args.cols, max_baseline)
                        recompute_needed = True
                    elif key == pygame.K_r:
                        reference_baseline = clamp(reference_baseline - 5.0, 1.0, 100000.0)
                        recompute_needed = True
                    elif key == pygame.K_t:
                        reference_baseline = clamp(reference_baseline + 5.0, 1.0, 100000.0)
                        recompute_needed = True
                    elif key == pygame.K_u:
                        uv_zoom = clamp(uv_zoom + 0.1, 0.2, 4.0)
                        recompute_needed = True
                    elif key == pygame.K_j:
                        uv_zoom = clamp(uv_zoom - 0.1, 0.2, 4.0)
                        recompute_needed = True
                    elif key == pygame.K_o:
                        point_radius = max(1, point_radius - 1)
                        recompute_needed = True
                    elif key == pygame.K_p:
                        point_radius = min(8, point_radius + 1)
                        recompute_needed = True
                    elif key == pygame.K_a:
                        smooth_passes = max(0, smooth_passes - 1)
                        recompute_needed = True
                    elif key == pygame.K_s:
                        smooth_passes = min(12, smooth_passes + 1)
                        recompute_needed = True

            if image_changed:
                recompute_needed = True
                refresh_visuals(animate=True)

            packets = receiver.drain_packets()
            drained_this_frame = len(packets)
            latest_valid: Optional[ParsedPacket] = None
            latest_time = None
            latest_sender_in_batch = None
            for packet_time, raw_packet, sender in packets:
                parsed = parse_contact_packet(raw_packet)
                if parsed.error is None:
                    latest_valid = parsed
                    latest_time = packet_time
                    latest_sender_in_batch = sender
                    valid_count += 1
                else:
                    parse_errors += 1
                    latest_error = parsed.error

            if latest_valid is not None and latest_valid.normalized != latest_packet:
                latest_packet = latest_valid.normalized
                latest_seq = latest_valid.seq
                latest_shape = f"{latest_valid.cols} x {latest_valid.rows}"
                latest_sender = latest_sender_in_batch or "-"
                latest_contact_array = latest_valid.contact_array
                recompute_needed = True

            if recompute_needed and latest_contact_array is not None:
                pos, missing_count = positions_from_contacts(latest_contact_array, mapping)
                if len(pos) >= 2:
                    dirty, uv = reconstruct_dirty_image(
                        sky,
                        pos,
                        max_baseline,
                        reference_baseline,
                        uv_zoom,
                        point_radius,
                        smooth_passes,
                    )
                    showcase_reconstruction, _, showcase_residual = reconstruct_exhibit_image(
                        sky,
                        pos,
                        max_baseline,
                        reference_baseline,
                        uv_zoom,
                        point_radius,
                    )
                    refresh_visuals(animate=True)
                    latest_error = ""
                    last_recompute = time.time()
                    last_layout_change = last_recompute
                else:
                    dirty = np.zeros((args.grid, args.grid), dtype=float)
                    uv = np.zeros((args.grid, args.grid), dtype=float)
                    showcase_reconstruction = np.zeros((args.grid, args.grid), dtype=float)
                    showcase_residual = np.zeros((args.grid, args.grid), dtype=float)
                    refresh_visuals(animate=True)
                    latest_error = "need at least 2 active antennas"
                    last_layout_change = time.time()

            status = receiver.get_status()
            margin = 12
            top_h = 96

            recv_age = "-"
            if status["last_received_time"] is not None:
                recv_age = f"{time.time() - float(status['last_received_time']):.2f}s ago"

            if display_mode == "exhibit":
                screen.fill((3, 9, 20))
                rects = {
                    name: pygame.Rect(*values)
                    for name, values in exhibit_panel_rects(screen.get_width(), screen.get_height(), top_h, margin).items()
                }
                baseline_count = len(pos) * max(0, len(pos) - 1) // 2
                update_state = "LIVE" if status["running"] else "WAITING"
                reconstruction_style = RECONSTRUCTION_STYLES[reconstruction_style_index]
                reconstruction_label = RECONSTRUCTION_STYLE_LABELS["ja" if japanese_ui else "en"][reconstruction_style]
                draw_text(screen, title_font, exhibit_labels["headline"], margin, 12, (243, 246, 251))
                draw_text(
                    screen,
                    small_font,
                    f"{update_state}   {len(pos)} {exhibit_labels['antennas']}   {baseline_count} {exhibit_labels['baselines']}   {args.fps:g} FPS   {reconstruction_label}   {EXHIBIT_THEMES[exhibit_theme_index].upper()}   W",
                    margin,
                    45,
                    (142, 211, 210) if status["running"] else (250, 181, 77),
                )
                if latest_error or status["last_error"]:
                    draw_text(screen, small_font, (latest_error or str(status["last_error"]))[:110], margin, 68, (250, 181, 77))

                transition = clamp((time.time() - last_visual_change) / DISPLAY_TRANSITION_SECONDS, 0.0, 1.0)
                visible_dirty_rgb = blend_rgb(previous_dirty_rgb, dirty_rgb, transition)
                pulse_strength = clamp(1.0 - (time.time() - last_layout_change) / LAYOUT_PULSE_SECONDS, 0.0, 1.0)
                draw_image_panel(
                    pygame,
                    screen,
                    rects["reconstruction"],
                    f"{exhibit_labels['reconstruction']}  |  {reconstruction_label}",
                    visible_dirty_rgb,
                    font,
                    accent=(246, 180, 75),
                    smooth=True,
                    contours=showcase_reconstruction if reconstruction_style == "clean" else None,
                    show_restoring_beam=reconstruction_style == "clean",
                    beam_uv=uv,
                )
                draw_layout_panel(
                    pygame,
                    screen,
                    rects["layout"],
                    exhibit_labels["layout"],
                    pos,
                    max_baseline / 2.0,
                    font,
                    accent=(246, 197, 67),
                    pulse_strength=pulse_strength,
                )
                draw_image_panel(pygame, screen, rects["uv"], exhibit_labels["uv"], uv_rgb, font, accent=(72, 201, 191))
                draw_image_panel(
                    pygame,
                    screen,
                    rects["source"],
                    exhibit_labels["source"],
                    sky_rgb,
                    font,
                    accent=(92, 172, 221),
                    smooth=True,
                )
                draw_live_stats_panel(
                    pygame,
                    screen,
                    rects["stats"],
                    len(pos),
                    font,
                    small_font,
                    stats_font,
                    accent=(102, 227, 163),
                    labels=exhibit_labels,
                )
            else:
                screen.fill((12, 14, 18))
                rects = [
                    pygame.Rect(*values)
                    for values in square_panel_rects(screen.get_width(), screen.get_height(), top_h, margin)
                ]
                draw_text(
                    screen,
                    font,
                    f"Realtime UDP display | {args.host}:{args.port} | target {args.fps:g} FPS | receiver "
                    f"{'running' if status['running'] else 'stopped'}",
                    margin,
                    8,
                )
                draw_text(
                    screen,
                    small_font,
                    f"total={status['total_received']} drained={drained_this_frame} valid={valid_count} "
                    f"errors={parse_errors} queue={status['queue_length']} seq={latest_seq if latest_seq is not None else '-'} "
                    f"shape={latest_shape} antennas={len(pos)} missing_map={missing_count} sender={latest_sender} "
                    f"last={recv_age} recompute={time.time() - last_recompute:.2f}s ago",
                    margin,
                    32,
                    (185, 193, 204),
                )
                draw_text(
                    screen,
                    small_font,
                    f"image={image_source} | cmap={CMAPS[cmap_index]} | stretch={stretch:.1f} pct={contrast_percentile:.1f} "
                    f"maxBL={max_baseline:.1f} refBL={reference_baseline:.1f} uvZoom={uv_zoom:.1f} "
                    f"point={point_radius} smooth={smooth_passes} | H:help",
                    margin,
                    56,
                    (185, 193, 204),
                )
                if mapping_warning or latest_error or status["last_error"]:
                    draw_text(
                        screen,
                        small_font,
                        f"{mapping_warning or ''} {latest_error or ''} {status['last_error'] or ''}".strip(),
                        margin,
                        76,
                        (250, 176, 70),
                    )

                draw_layout_panel(pygame, screen, rects[0], "Antenna / station layout", pos, max_baseline / 2.0, font)
                draw_image_panel(pygame, screen, rects[1], "Effective uv coverage", uv_rgb, font)
                draw_image_panel(pygame, screen, rects[2], "Reconstructed image (dirty image)", dirty_rgb, font)
                draw_image_panel(pygame, screen, rects[3], "Input image / true structure", sky_rgb, font)

                if latest_packet:
                    draw_text(screen, small_font, f"packet: {latest_packet[:150]}", rects[2].x + 10, rects[2].bottom - 22)
                draw_text(screen, small_font, f"mapping: {mapping_source}", rects[3].x + 10, rects[3].bottom - 22)
            if show_help:
                draw_help_overlay(pygame, screen, font, small_font, args, image_source)

            pygame.display.flip()
            frame_count += 1
            if args.max_frames > 0 and frame_count >= args.max_frames:
                running = False
            clock.tick(args.fps)
    finally:
        receiver.stop()
        pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime UDP display for SKA Interferometer Puzzle.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9900)
    parser.add_argument("--mapping", default="grid_mapping.csv")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument(
        "--grid",
        type=int,
        default=192,
        help="Image/reconstruction resolution. Default 192; use 96 for slower computers or 256 for a sharper display.",
    )
    parser.add_argument("--width", type=int, default=1040)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-queue", type=int, default=1000)
    parser.add_argument("--max-baseline", type=float, default=74.0)
    parser.add_argument("--reference-baseline", type=float, default=150.0)
    parser.add_argument("--uv-zoom", type=float, default=1.0)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--smooth-passes", type=int, default=2)
    parser.add_argument("--contrast-percentile", type=float, default=99.0)
    parser.add_argument("--stretch", type=float, default=4.0)
    parser.add_argument(
        "--display-mode",
        choices=DISPLAY_MODES,
        default="exhibit",
        help="Default is the public-facing exhibit view. Press D to switch at runtime.",
    )
    parser.add_argument(
        "--exhibit-theme",
        choices=EXHIBIT_THEMES,
        default="aurora",
        help="Exhibit-only display color theme. Press C to change at runtime.",
    )
    parser.add_argument(
        "--reconstruction-style",
        choices=RECONSTRUCTION_STYLES,
        default="exhibit",
        help="Exhibit reconstruction style. Press W to change at runtime.",
    )
    parser.add_argument("--cmap", choices=CMAPS, default="thermal")
    parser.add_argument("--image", default=None, help="Optional input image path. Press L at runtime to reload it.")
    parser.add_argument("--sample", choices=SAMPLE_MODELS, default="gas", help="Built-in input image when --image is omitted.")
    parser.add_argument("--font", default=None, help="Optional pygame font name.")
    parser.add_argument("--show-help", action="store_true", help="Show runtime key help at startup.")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
    else:
        run_display(args)


if __name__ == "__main__":
    main()
