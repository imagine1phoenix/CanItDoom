"""Offline evaluation harness and scoring pipeline for DOOM agents.

Tracks competition scoring metrics:
- Exit reached (primary goal)
- Progression (keys found, doors opened)
- Spatial exploration (unique 64x64 cells visited)
- Combat & survival (kills, damage taken, deaths)
- Inference latency (per-decision time in ms, official budget 40ms)
- Policy mode comparison (greedy vs. stochastic sampling across temperatures)
- Optional OpenCV video recording for demo submission

Exports structured results to CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add repo root to import path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from doom_env import make_env
from submission.model import DoomActorCritic


def evaluate_episode(
    model: DoomActorCritic,
    wad_path: str,
    map_name: str = "MAP01",
    episode_ticks: int = 2100,
    deterministic: bool = True,
    temperature: float = 1.0,
    record_video_path: str | None = None,
    device: str = "cpu",
) -> dict:
    """Evaluates a single map episode and returns comprehensive metrics."""
    dev = torch.device(device)
    model.eval()

    need_buffers = record_video_path is not None
    env = make_env(wad_path, map_name, episode_ticks=episode_ticks, render_buffers=need_buffers)

    obs, _ = env.reset(seed=42)
    hx = model.initial_state(batch_size=1, device=dev)

    # Metrics trackers
    visited_cells: set[tuple[int, int]] = set()
    step_latencies: list[float] = []
    total_steps = 0
    video_writer = None

    if record_video_path:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Observation resolution is (160, 120)
        video_writer = cv2.VideoWriter(record_video_path, fourcc, 35.0 / 4.0, (160, 120))

    initial_priv = env.privileged_state()
    start_x, start_y = initial_priv.get("x", 0.0), initial_priv.get("y", 0.0)
    visited_cells.add((int(start_x // 64), int(start_y // 64)))

    terminated, truncated = False, False
    start_wall_time = time.perf_counter()

    while not (terminated or truncated):
        if video_writer is not None and "rgb" in obs:
            # OpenCV expects BGR uint8
            bgr = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR)
            video_writer.write(bgr)

        # Prepare observation tensors
        obs_tensor = {
            "rays": {
                "distance": torch.from_numpy(obs["rays"]["distance"]).unsqueeze(0).to(dev),
                "hit_type": torch.from_numpy(obs["rays"]["hit_type"]).unsqueeze(0).to(dev),
            },
            "visible_entities": {
                "features": torch.from_numpy(obs["visible_entities"]["features"]).unsqueeze(0).to(dev),
                "entity_type": torch.from_numpy(obs["visible_entities"]["entity_type"]).unsqueeze(0).to(dev),
                "mask": torch.from_numpy(obs["visible_entities"]["mask"]).unsqueeze(0).to(dev),
            },
            "vitals": torch.from_numpy(obs["vitals"]).unsqueeze(0).to(dev),
        }

        # Measure pure agent inference latency
        t0 = time.perf_counter()
        with torch.no_grad():
            if deterministic:
                action, _, _, _, hx = model.step(obs_tensor, hx, deterministic=True)
            else:
                # Temperature scaled sampling
                features = model.extract_features(obs_tensor).unsqueeze(1)
                out, hx = model.gru(features, hx)
                logits = model.actor_head(out.squeeze(1)) / max(1e-4, temperature)
                dist = torch.distributions.Bernoulli(logits=logits)
                action = dist.sample()
        step_ms = (time.perf_counter() - t0) * 1000.0
        step_latencies.append(step_ms)

        action_np = action.squeeze(0).cpu().numpy().astype(np.int8)
        obs, reward, terminated, truncated, info = env.step(action_np)
        total_steps += 1

        priv = env.privileged_state()
        x, y = priv.get("x", 0.0), priv.get("y", 0.0)
        visited_cells.add((int(x // 64), int(y // 64)))

    total_wall_s = time.perf_counter() - start_wall_time
    final_priv = env.privileged_state()

    if video_writer is not None:
        video_writer.release()

    keys = final_priv.get("keys", set())
    keys_count = len(keys) if isinstance(keys, (set, list)) else 0

    return {
        "map": f"{wad_path}:{map_name}",
        "exit_reached": final_priv.get("exit_reached", False),
        "dead": final_priv.get("dead", False),
        "kills": final_priv.get("kills", 0),
        "damage_taken": final_priv.get("damage_taken", 0.0),
        "keys_found": keys_count,
        "unique_cells": len(visited_cells),
        "total_steps": total_steps,
        "wall_time_s": round(total_wall_s, 2),
        "avg_latency_ms": round(float(np.mean(step_latencies)), 2) if step_latencies else 0.0,
        "p99_latency_ms": round(float(np.percentile(step_latencies, 99)), 2) if step_latencies else 0.0,
        "max_latency_ms": round(float(np.max(step_latencies)), 2) if step_latencies else 0.0,
        "mode": "greedy" if deterministic else f"sample(T={temperature})",
    }


def run_evaluation_suite(
    checkpoint_path: str,
    map_specs: list[tuple[str, str]],
    output_csv: str = "eval_results.csv",
    mode: str = "greedy",
    temperature: float = 1.0,
    video_dir: str | None = None,
    device: str = "cpu",
) -> list[dict]:
    """Evaluates a checkpoint across a suite of maps and saves results to CSV."""
    dev = torch.device(device)
    model = DoomActorCritic(num_actions=9).to(dev)

    ckpt_file = Path(checkpoint_path)
    if ckpt_file.is_file():
        state = torch.load(ckpt_file, map_location=dev, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict):
            model.load_state_dict(state)
        print(f"Loaded evaluation checkpoint from {ckpt_file}")
    else:
        print(f"Warning: Checkpoint {ckpt_file} not found; evaluating uninitialized model")

    deterministic = (mode == "greedy")
    results = []

    if video_dir:
        Path(video_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 95)
    print(f"{'Map':<24} | {'Exit':<5} | {'Dead':<5} | {'Kills':<5} | {'Keys':<5} | {'Cells':<6} | {'Steps':<6} | {'Avg ms':<6} | {'Mode':<10}")
    print("-" * 95)

    for i, (wad, map_name) in enumerate(map_specs):
        vid_path = None
        if video_dir:
            vid_path = str(Path(video_dir) / f"eval_{Path(wad).stem}_{map_name}.mp4")

        res = evaluate_episode(
            model=model,
            wad_path=wad,
            map_name=map_name,
            deterministic=deterministic,
            temperature=temperature,
            record_video_path=vid_path,
            device=device,
        )
        results.append(res)

        exit_str = "YES" if res["exit_reached"] else "no"
        dead_str = "YES" if res["dead"] else "no"
        print(
            f"{res['map']:<24} | {exit_str:<5} | {dead_str:<5} | {res['kills']:<5} | "
            f"{res['keys_found']:<5} | {res['unique_cells']:<6} | {res['total_steps']:<6} | "
            f"{res['avg_latency_ms']:<6} | {res['mode']:<10}"
        )

    print("=" * 95)

    # Write CSV
    if results:
        csv_path = Path(output_csv)
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved evaluation metrics CSV to: {csv_path.resolve()}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation harness for DOOM agents")
    parser.add_argument("--checkpoint", default="submission/weights/model.pt", help="Model weights path")
    parser.add_argument("--map", default="maps/two_rooms.wad:MAP01", help="Target map (wad:MAPxx)")
    parser.add_argument("--maps-json", default=None, help="Optional JSON file listing map suite")
    parser.add_argument("--output", default="eval_results.csv", help="Output metrics CSV file")
    parser.add_argument("--mode", default="greedy", choices=["greedy", "sample"], help="Policy inference mode")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--video-dir", default=None, help="Directory to save replay MP4 videos")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    map_specs = []
    if args.maps_json and Path(args.maps_json).is_file():
        with open(args.maps_json, encoding="utf-8") as f:
            catalog = json.load(f)
            for entry in catalog:
                map_specs.append((entry["file"], entry.get("map", "MAP01")))
    else:
        parts = args.map.split(":")
        wad = parts[0]
        map_name = parts[1] if len(parts) > 1 else "MAP01"
        map_specs.append((wad, map_name))

    run_evaluation_suite(
        checkpoint_path=args.checkpoint,
        map_specs=map_specs,
        output_csv=args.output,
        mode=args.mode,
        temperature=args.temperature,
        video_dir=args.video_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
