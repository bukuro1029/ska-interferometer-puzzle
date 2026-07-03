"""
Test UDP sender for SKA Interferometer Puzzle hardware-input mode.

Run examples:
    python udp_sender_test.py
    python udp_sender_test.py --rows 8 --cols 8 --interval 0.2 --mode random
    python udp_sender_test.py --rows 8 --cols 8 --interval 0.8 --mode layout_demo --seq
    python udp_sender_test.py --packet "10100100 10100111 11101100"
    python udp_sender_test.py --rows 8 --cols 8 --interval 0.2 --mode moving --count 3
    python udp_sender_test.py --rows 8 --cols 8 --interval 0.2 --mode moving --seq
"""

from __future__ import annotations

import argparse
import random
import socket
import time


def packet_from_active(rows: int, cols: int, active: set[tuple[int, int]]) -> str:
    lines = []
    for r in range(rows):
        line = ''.join('1' if (r, c) in active else '0' for c in range(cols))
        lines.append(line)
    return ' '.join(lines)


def random_packet(rows: int, cols: int, n_active: int) -> str:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    active = set(random.sample(cells, k=min(n_active, len(cells))))
    return packet_from_active(rows, cols, active)


def moving_packet(rows: int, cols: int, step: int) -> str:
    active = set()
    for k in range(min(rows, cols)):
        active.add((k, (k + step) % cols))
        active.add((k, (cols - 1 - k + step) % cols))
    return packet_from_active(rows, cols, active)


def take_cells(candidates: list[tuple[int, int]], rows: int, cols: int, n_active: int) -> set[tuple[int, int]]:
    active = []
    seen = set()
    for r, c in candidates:
        key = (int(max(0, min(rows - 1, r))), int(max(0, min(cols - 1, c))))
        if key not in seen:
            seen.add(key)
            active.append(key)
        if len(active) >= n_active:
            return set(active)

    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    for key in all_cells:
        if key not in seen:
            seen.add(key)
            active.append(key)
        if len(active) >= n_active:
            break
    return set(active)


def layout_demo_cells(rows: int, cols: int, n_active: int, step: int) -> tuple[str, set[tuple[int, int]]]:
    n_active = max(2, min(n_active, rows * cols))
    center_r = (rows - 1) / 2.0
    center_c = (cols - 1) / 2.0
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    name = ""
    candidates: list[tuple[int, int]]

    pattern = step % 7
    if pattern == 0:
        name = "compact_core"
        candidates = sorted(cells, key=lambda rc: (rc[0] - center_r) ** 2 + (rc[1] - center_c) ** 2)
    elif pattern == 1:
        name = "horizontal_line"
        mid = rows // 2
        candidates = [(mid, c) for c in range(cols)]
        candidates += [(max(0, mid - 1), c) for c in range(cols)]
        candidates += [(min(rows - 1, mid + 1), c) for c in range(cols)]
    elif pattern == 2:
        name = "vertical_line"
        mid = cols // 2
        candidates = [(r, mid) for r in range(rows)]
        candidates += [(r, max(0, mid - 1)) for r in range(rows)]
        candidates += [(r, min(cols - 1, mid + 1)) for r in range(rows)]
    elif pattern == 3:
        name = "diagonal_cross"
        n = min(rows, cols)
        candidates = [(k, k) for k in range(n)]
        candidates += [(k, cols - 1 - k) for k in range(n)]
    elif pattern == 4:
        name = "outer_baselines"
        candidates = sorted(cells, key=lambda rc: -((rc[0] - center_r) ** 2 + (rc[1] - center_c) ** 2))
    elif pattern == 5:
        name = "two_clusters"
        anchors = [(rows * 0.25, cols * 0.25), (rows * 0.75, cols * 0.75)]
        candidates = []
        ranked_groups = [
            sorted(cells, key=lambda rc: (rc[0] - anchor_r) ** 2 + (rc[1] - anchor_c) ** 2)
            for anchor_r, anchor_c in anchors
        ]
        for k in range(rows * cols):
            for ranked in ranked_groups:
                if k < len(ranked):
                    candidates.append(ranked[k])
    else:
        name = "three_arms"
        candidates = [(round(center_r), round(center_c))]
        max_len = max(rows, cols)
        for k in range(1, max_len):
            candidates.append((round(center_r), round(center_c + k)))
            candidates.append((round(center_r + 0.86 * k), round(center_c - 0.50 * k)))
            candidates.append((round(center_r - 0.86 * k), round(center_c - 0.50 * k)))

    return name, take_cells(candidates, rows, cols, n_active)


def layout_demo_packet(rows: int, cols: int, n_active: int, step: int) -> tuple[str, str]:
    name, active = layout_demo_cells(rows, cols, n_active, step)
    return name, packet_from_active(rows, cols, active)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9900)
    parser.add_argument('--rows', type=int, default=8)
    parser.add_argument('--cols', type=int, default=8)
    parser.add_argument('--active', type=int, default=8)
    parser.add_argument('--interval', type=float, default=0.2)
    parser.add_argument('--mode', choices=['moving', 'random', 'layout_demo'], default='moving')
    parser.add_argument('--packet', default=None, help='Send this one packet repeatedly.')
    parser.add_argument('--count', type=int, default=0, help='Number of packets to send. 0 means run until Ctrl+C.')
    parser.add_argument('--seq', action='store_true', help='Prefix packets with seq=N for receiver debugging.')
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    step = 0
    print(f'Sending UDP packets to {args.host}:{args.port}. Press Ctrl+C to stop.')
    try:
        while True:
            if args.packet is not None:
                packet = args.packet
                label = 'fixed'
            elif args.mode == 'random':
                packet = random_packet(args.rows, args.cols, args.active)
                label = 'random'
            elif args.mode == 'layout_demo':
                label, packet = layout_demo_packet(args.rows, args.cols, args.active, step)
            else:
                packet = moving_packet(args.rows, args.cols, step)
                label = 'moving'
            packet_to_send = f'seq={step + 1} {packet}' if args.seq else packet
            sock.sendto(packet_to_send.encode('ascii'), (args.host, args.port))
            print(f'[{label}] {packet_to_send}')
            step += 1
            if args.count > 0 and step >= args.count:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('Stopped.')


if __name__ == '__main__':
    main()
