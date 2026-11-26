#!/usr/bin/env python3
"""Headless ViZDoom / doom_env throughput benchmark (Task V.2).

Measures decision steps per second (SPS) across single-env and multiprocess
configurations, with and without pixel buffer rendering (`render_buffers`).

Usage:
    uv run python train/benchmark_sps.py
    uv run python train/benchmark_sps.py --quick
    uv run python train/benchmark_sps.py --map maps/two_rooms.wad:MAP01 --workers 1,2,4,8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from doom_env import make_env

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BenchResult:
    label: str
    workers: int
    render_buffers: bool
    total_steps: int
    elapsed_s: float
    sps: float

    @property
    def projected_9h_millions(self) -> float:
        return self.sps * 9.0 * 3600.0 / 1e6


def _run_worker(
    worker_id: int,
    wad: str,
    map_name: str,
    steps_per_worker: int,
    episode_ticks: int,
    render_buffers: bool,
    barrier,
    queue,
) -> None:
    rng = np.random.default_rng(1000 + worker_id)
    env = make_env(
        wad,
        map_name,
        episode_ticks=episode_ticks,
        render_buffers=render_buffers,
    )
    try:
        env.reset(seed=worker_id)
        # Warm up 10 steps before synchronized timing.
        for _ in range(10):
            act = rng.integers(0, 2, size=env.action_space.shape, dtype=np.int8)
            _, _, term, trunc, _ = env.step(act)
            if term or trunc:
                env.reset(seed=worker_id)

        barrier.wait(timeout=60)
        t0 = time.perf_counter()
        for _ in range(steps_per_worker):
            act = rng.integers(0, 2, size=env.action_space.shape, dtype=np.int8)
            _, _, term, trunc, _ = env.step(act)
            if term or trunc:
                env.reset()
        elapsed = time.perf_counter() - t0
        queue.put((worker_id, steps_per_worker, elapsed))
    finally:
        env.close()


def bench_single(
    wad: str,
    map_name: str,
    total_steps: int,
    episode_ticks: int,
    render_buffers: bool,
) -> BenchResult:
    rng = np.random.default_rng(0)
    env = make_env(
        wad,
        map_name,
        episode_ticks=episode_ticks,
        render_buffers=render_buffers,
    )
    try:
        env.reset(seed=0)
        for _ in range(20):
            act = rng.integers(0, 2, size=env.action_space.shape, dtype=np.int8)
            _, _, term, trunc, _ = env.step(act)
            if term or trunc:
                env.reset(seed=0)

        t0 = time.perf_counter()
        for _ in range(total_steps):
            act = rng.integers(0, 2, size=env.action_space.shape, dtype=np.int8)
            _, _, term, trunc, _ = env.step(act)
            if term or trunc:
                env.reset()
        elapsed = time.perf_counter() - t0
    finally:
        env.close()

    return BenchResult(
        label=f"single (render={render_buffers})",
        workers=1,
        render_buffers=render_buffers,
        total_steps=total_steps,
        elapsed_s=elapsed,
        sps=total_steps / max(elapsed, 1e-9),
    )


def bench_multiprocess(
    wad: str,
    map_name: str,
    workers: int,
    steps_per_worker: int,
    episode_ticks: int,
    render_buffers: bool,
) -> BenchResult:
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(workers + 1)
    queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_run_worker,
            args=(i, wad, map_name, steps_per_worker, episode_ticks, render_buffers, barrier, queue),
        )
        for i in range(workers)
    ]
    for p in procs:
        p.start()

    try:
        barrier.wait(timeout=60)
        t0 = time.perf_counter()
        results = [queue.get(timeout=180) for _ in range(workers)]
        wall = time.perf_counter() - t0
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
                p.join()

    total_steps = sum(r[1] for r in results)
    return BenchResult(
        label=f"mp x{workers} (render={render_buffers})",
        workers=workers,
        render_buffers=render_buffers,
        total_steps=total_steps,
        elapsed_s=wall,
        sps=total_steps / max(wall, 1e-9),
    )


def print_system_info() -> None:
    print("=== Hardware & Environment Profile ===")
    print(f"OS:       {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python:   {platform.python_version()}")
    print(f"CPU cores (logical): {os.cpu_count()}")
    try:
        import torch  # noqa: PLC0415

        cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda else "none"
        print(f"PyTorch:  {torch.__version__} | CUDA: {cuda} ({gpu_name})")
    except ImportError:
        print("PyTorch:  not installed in this environment")
    print()


def parse_map_spec(spec: str) -> tuple[str, str]:
    wad_part, map_name = spec, "MAP01"
    if ":" in spec:
        head, _, tail = spec.rpartition(":")
        # Ignore Windows drive-letter colons like "C:\path\to\file.wad"
        if not (len(head) == 1 and head.isalpha() and tail.startswith(("\\", "/"))):
            wad_part, map_name = head, (tail or "MAP01")
    if wad_part.lower() in {"freedoom2", "freedoom2.wad"}:
        import vizdoom  # noqa: PLC0415

        return str(Path(vizdoom.__file__).resolve().parent / "freedoom2.wad"), map_name
    wad_path = Path(wad_part)
    if not wad_path.is_absolute():
        wad_path = ROOT / wad_path
    return str(wad_path), map_name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="maps/two_rooms.wad:MAP01", help="WAD[:MAP] to benchmark")
    ap.add_argument("--steps", type=int, default=2000, help="Decision steps per worker")
    ap.add_argument("--episode-ticks", type=int, default=2100, help="Episode tick cap")
    ap.add_argument(
        "--workers",
        default=None,
        help="Comma-separated worker counts for multiprocess sweep (default: auto up to cpu_count)",
    )
    ap.add_argument("--quick", action="store_true", help="Short smoke benchmark (500 steps/worker)")
    args = ap.parse_args()

    steps = 500 if args.quick else args.steps
    wad, map_name = parse_map_spec(args.map)
    cpus = os.cpu_count() or 2

    if args.workers:
        worker_counts = [int(x.strip()) for x in args.workers.split(",") if x.strip()]
    elif args.quick:
        worker_counts = [w for w in (2, 4) if w <= cpus]
    else:
        candidates = [2, 4, 8, 12, 16]
        worker_counts = [w for w in candidates if w <= cpus]
        if not worker_counts and cpus >= 2:
            worker_counts = [2]

    print_system_info()
    print(f"Map: {wad} ({map_name}) | Steps/worker: {steps} | Episode ticks: {args.episode_ticks}\n")

    results: list[BenchResult] = []
    results.append(bench_single(wad, map_name, steps, args.episode_ticks, render_buffers=True))
    results.append(bench_single(wad, map_name, steps, args.episode_ticks, render_buffers=False))
    for w in worker_counts:
        results.append(
            bench_multiprocess(wad, map_name, w, steps, args.episode_ticks, render_buffers=False)
        )

    print(f"{'Configuration':<28} {'Workers':>7} {'Steps':>8} {'Wall (s)':>9} {'SPS':>9} {'9h Est (M)':>11}")
    print("-" * 78)
    for r in results:
        print(
            f"{r.label:<28} {r.workers:>7} {r.total_steps:>8} "
            f"{r.elapsed_s:>9.2f} {r.sps:>9.1f} {r.projected_9h_millions:>11.1f}"
        )

    best = max(results, key=lambda r: r.sps)
    print(f"\nPeak throughput: {best.sps:.1f} SPS ({best.label}, ~{best.projected_9h_millions:.1f}M steps in 9h)")
    if best.sps >= 1000:
        print("Status: IDEAL (>= 1000 SPS; full curriculum stages 1-5 feasible overnight).")
    elif best.sps >= 500:
        print("Status: PASS (>= 500 SPS target met).")
    else:
        print("Status: BELOW TARGET (< 500 SPS; prefer Colab/Linux or train stages 1-3 per team plan).")
    return 0


if __name__ == "__main__":
    main()
