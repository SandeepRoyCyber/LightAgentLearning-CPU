#!/usr/bin/env python3
"""Lightweight CPU benchmark helper for low-memory Agent Lightning setups."""

from __future__ import annotations

import argparse
import math
import os
import statistics
import time


def synthetic_benchmark(iterations: int, width: int) -> dict[str, float]:
    """Run a deterministic CPU workload to estimate baseline throughput."""
    samples = []
    x = 0.123456789

    for _ in range(iterations):
        t0 = time.perf_counter()
        for i in range(width):
            x = math.sin(x + i * 1e-5) * math.cos(x - i * 1e-5)
        samples.append(time.perf_counter() - t0)

    # keep optimizer away from x becoming unused
    if x == float('inf'):
        raise RuntimeError('unexpected numerical value')

    mean_s = statistics.mean(samples)
    p95_s = statistics.quantiles(samples, n=20)[18] if len(samples) >= 20 else max(samples)
    return {
        'iterations': float(iterations),
        'width': float(width),
        'mean_step_ms': mean_s * 1000,
        'p95_step_ms': p95_s * 1000,
        'steps_per_sec': 1.0 / mean_s if mean_s else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='CPU benchmark helper for 8GB laptop profiles')
    parser.add_argument('--iterations', type=int, default=30, help='Number of timed steps')
    parser.add_argument('--width', type=int, default=50000, help='Work per step')
    args = parser.parse_args()

    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')

    results = synthetic_benchmark(args.iterations, args.width)
    print('CPU benchmark (synthetic):')
    for key, value in results.items():
        if key in {'iterations', 'width'}:
            print(f'  {key}: {int(value)}')
        else:
            print(f'  {key}: {value:.3f}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
