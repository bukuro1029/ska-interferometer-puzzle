"""
Realtime UDP display for the SKA Interferometer Puzzle exhibit.

This script is intentionally separate from the Streamlit app. It runs a small
pygame window with its own 10 FPS render loop, so packet reception and drawing
are not limited by Streamlit reruns.

Examples:
    python -m pip install -r requirements-realtime.txt
    python realtime_display.py
    python realtime_display.py --fullscreen --fps 10
    python realtime_display.py --host 127.0.0.1 --port 9900 --mapping grid_mapping.csv
    python realtime_display.py --image /path/to/input.png
    python realtime_display.py --sample points
    python realtime_display.py --self-test

Pair with:
    python udp_sender_test.py --rows 8 --cols 8 --interval 0.2 --mode moving --seq

Runtime keys:
    H help, I cycle sample image, L reload --image, C cycle colormap, Q/Esc quit.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


PacketItem = Tuple[float, str, str]
SAMPLE_MODELS = ["gas", "points", "bubbles", "ska"]
CMAPS = ["thermal", "icefire", "viridis", "RdBu_r", "seismic", "coolwarm", "gray"]


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


def blur_array(arr: np.ndarray, passes: int = 2) -> np.ndarray:
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

    sky_ref = sky - np.mean(sky)
    sampled = np.fft.fftshift(np.fft.fft2(sky_ref)) * effective_uv
    dirty = np.real(np.fft.ifft2(np.fft.ifftshift(sampled)))
    dirty -= np.mean(dirty)
    return dirty, effective_uv


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


def resize_array_nearest(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    y_idx = np.linspace(0, arr.shape[0] - 1, height).astype(int)
    x_idx = np.linspace(0, arr.shape[1] - 1, width).astype(int)
    return arr[np.ix_(y_idx, x_idx)]


def draw_text(surface, font, text: str, x: int, y: int, color=(235, 238, 241)) -> None:
    rendered = font.render(text, True, color)
    surface.blit(rendered, (x, y))


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
        f"  built-in sample: python realtime_display.py --sample gas|points|bubbles|ska  (current default: {args.sample})",
        "",
        "Runtime keys:",
        "  H: show/hide this help    Q or Esc: quit",
        "  I: cycle built-in sample image    L: reload --image file",
        "  C: cycle dirty-image colormap",
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


def draw_image_panel(pygame, screen, rect, title: str, rgb: np.ndarray, font) -> None:
    pygame.draw.rect(screen, (22, 25, 30), rect)
    draw_text(screen, font, title, rect.x + 10, rect.y + 8)
    inner = pygame.Rect(*square_inner_rect(rect))
    resized = resize_array_nearest(rgb, inner.w, inner.h)
    surf = pygame.surfarray.make_surface(np.transpose(resized, (1, 0, 2)))
    screen.blit(surf, inner)
    pygame.draw.rect(screen, (80, 88, 100), rect, 1)


def draw_layout_panel(pygame, screen, rect, title: str, pos: np.ndarray, radius: float, font) -> None:
    pygame.draw.rect(screen, (22, 25, 30), rect)
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
        pygame.draw.circle(screen, (246, 197, 67), (px, py), 5)
    pygame.draw.rect(screen, (80, 88, 100), rect, 1)


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
    rects = square_panel_rects(1040, 960, 96, 12)
    assert len(rects) == 4
    assert all(w == h for _, _, w, h in rects)
    print("self-test ok")


def run_display(args: argparse.Namespace) -> None:
    import pygame

    pygame.init()
    flags = pygame.FULLSCREEN if args.fullscreen else 0
    screen = pygame.display.set_mode((args.width, args.height), flags)
    pygame.display.set_caption("SKA Interferometer Puzzle - Realtime UDP Display")
    font = pygame.font.SysFont(args.font, 18)
    small_font = pygame.font.SysFont(args.font, 15)
    clock = pygame.time.Clock()

    sample_index = SAMPLE_MODELS.index(args.sample) if args.sample in SAMPLE_MODELS else 0
    cmap_index = CMAPS.index(args.cmap) if args.cmap in CMAPS else 0
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
    dirty_rgb = diverging_rgb(dirty, CMAPS[cmap_index])
    uv_rgb = scalar_to_rgb(uv)
    sky_rgb = scalar_to_rgb(sky)
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
    frame_count = 0
    running = True

    try:
        while running:
            recompute_needed = False
            image_changed = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in {pygame.K_ESCAPE, pygame.K_q}:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    key = event.key
                    if key == pygame.K_h:
                        show_help = not show_help
                    elif key == pygame.K_i:
                        sample_index = (sample_index + 1) % len(SAMPLE_MODELS)
                        sky, image_source = load_image_sky(None, args.grid, SAMPLE_MODELS[sample_index])
                        sky_rgb = scalar_to_rgb(sky)
                        image_changed = True
                    elif key == pygame.K_l:
                        sky, image_source = load_image_sky(args.image, args.grid, SAMPLE_MODELS[sample_index])
                        sky_rgb = scalar_to_rgb(sky)
                        image_changed = True
                    elif key == pygame.K_c:
                        cmap_index = (cmap_index + 1) % len(CMAPS)
                        dirty_rgb = diverging_rgb(asinh_stretch_signed(dirty, contrast_percentile, stretch), CMAPS[cmap_index])
                    elif key == pygame.K_z:
                        stretch = clamp(stretch - 0.5, 1.0, 10.0)
                        dirty_rgb = diverging_rgb(asinh_stretch_signed(dirty, contrast_percentile, stretch), CMAPS[cmap_index])
                    elif key == pygame.K_x:
                        stretch = clamp(stretch + 0.5, 1.0, 10.0)
                        dirty_rgb = diverging_rgb(asinh_stretch_signed(dirty, contrast_percentile, stretch), CMAPS[cmap_index])
                    elif key == pygame.K_n:
                        contrast_percentile = clamp(contrast_percentile - 0.5, 90.0, 99.9)
                        dirty_rgb = diverging_rgb(asinh_stretch_signed(dirty, contrast_percentile, stretch), CMAPS[cmap_index])
                    elif key == pygame.K_m:
                        contrast_percentile = clamp(contrast_percentile + 0.5, 90.0, 99.9)
                        dirty_rgb = diverging_rgb(asinh_stretch_signed(dirty, contrast_percentile, stretch), CMAPS[cmap_index])
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
                    dirty_rgb = diverging_rgb(asinh_stretch_signed(dirty, contrast_percentile, stretch), CMAPS[cmap_index])
                    uv_rgb = scalar_to_rgb(uv)
                    latest_error = ""
                    last_recompute = time.time()
                else:
                    dirty = np.zeros((args.grid, args.grid), dtype=float)
                    uv = np.zeros((args.grid, args.grid), dtype=float)
                    dirty_rgb = diverging_rgb(dirty, CMAPS[cmap_index])
                    uv_rgb = scalar_to_rgb(uv)
                    latest_error = "need at least 2 active antennas"

            status = receiver.get_status()
            screen.fill((12, 14, 18))
            margin = 12
            top_h = 96
            rects = [
                pygame.Rect(*values)
                for values in square_panel_rects(args.width, args.height, top_h, margin)
            ]

            recv_age = "-"
            if status["last_received_time"] is not None:
                recv_age = f"{time.time() - float(status['last_received_time']):.2f}s ago"

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
    parser.add_argument("--grid", type=int, default=96)
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
