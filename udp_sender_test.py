"""
Test UDP sender for SKA Interferometer Puzzle hardware-input mode.

Run examples:
    python udp_sender_test.py
    python udp_sender_test.py --rows 8 --cols 8 --interval 0.2 --mode random
    python udp_sender_test.py --packet "10100100 10100111 11101100"
    python udp_sender_test.py --rows 8 --cols 8 --interval 0.2 --mode moving --count 3
"""

from __future__ import annotations

import argparse
import random
import socket
import time


def random_packet(rows: int, cols: int, n_active: int) -> str:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    active = set(random.sample(cells, k=min(n_active, len(cells))))
    lines = []
    for r in range(rows):
        line = ''.join('1' if (r, c) in active else '0' for c in range(cols))
        lines.append(line)
    return ' '.join(lines)


def moving_packet(rows: int, cols: int, step: int) -> str:
    active = set()
    for k in range(min(rows, cols)):
        active.add((k, (k + step) % cols))
        active.add((k, (cols - 1 - k + step) % cols))
    lines = []
    for r in range(rows):
        line = ''.join('1' if (r, c) in active else '0' for c in range(cols))
        lines.append(line)
    return ' '.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9900)
    parser.add_argument('--rows', type=int, default=8)
    parser.add_argument('--cols', type=int, default=8)
    parser.add_argument('--active', type=int, default=8)
    parser.add_argument('--interval', type=float, default=0.2)
    parser.add_argument('--mode', choices=['moving', 'random'], default='moving')
    parser.add_argument('--packet', default=None, help='Send this one packet repeatedly.')
    parser.add_argument('--count', type=int, default=0, help='Number of packets to send. 0 means run until Ctrl+C.')
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    step = 0
    print(f'Sending UDP packets to {args.host}:{args.port}. Press Ctrl+C to stop.')
    try:
        while True:
            if args.packet is not None:
                packet = args.packet
            elif args.mode == 'random':
                packet = random_packet(args.rows, args.cols, args.active)
            else:
                packet = moving_packet(args.rows, args.cols, step)
            sock.sendto(packet.encode('ascii'), (args.host, args.port))
            print(packet)
            step += 1
            if args.count > 0 and step >= args.count:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('Stopped.')


if __name__ == '__main__':
    main()
