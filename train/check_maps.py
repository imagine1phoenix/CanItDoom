#!/usr/bin/env python3
"""Verify all maps registered in train/maps.json.

For each entry, this script:
  1. Resolves the WAD path (relative to repo root, or bundled freedoom2.wad).
  2. Parses WAD metadata via doom_env.mapinfo and checks sector count <= 2048.
  3. Launches DoomEnv, resets, and steps up to 100 random valid actions.
  4. Validates observations against observation_space and prints a pass/fail table.

Run from repository root:
    python train/check_maps.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import vizdoom as vd

from doom_env import make_env
from doom_env.mapinfo import load_mapinfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "train" / "maps.json"
MAX_SECTORS = 2048
VALID_SPLITS = {"train", "held_out"}


def resolve_wad(file_ref: str) -> Path:
    """Resolve a catalog file path against repo root or bundled ViZDoom IWAD."""
    candidate = Path(file_ref)
    if candidate.is_absolute():
        return candidate
    repo_path = ROOT / candidate
    if repo_path.is_file():
        return repo_path
    if file_ref == "freedoom2.wad":
        iwad = Path(vd.__file__).resolve().parent / "freedoom2.wad"
        if iwad.is_file():
            return iwad
    return repo_path


def check_entry(entry: dict, steps: int, seed: int, render_buffers: bool) -> dict:
    """Validate one map catalog entry and run a random-action smoke test."""
    file_ref = str(entry.get("file", ""))
    map_name = str(entry.get("map", "MAP01")).upper()
    stage = entry.get("stage", "?")
    split = entry.get("split", "?")

    result = {
        "file": file_ref,
        "map": map_name,
        "stage": stage,
        "split": split,
        "sectors": "-",
        "doors": "-",
        "steps": 0,
        "status": "FAIL",
        "detail": "",
    }

    if not file_ref or not map_name:
        result["detail"] = "missing 'file' or 'map' field"
        return result
    if split not in VALID_SPLITS:
        result["detail"] = f"invalid split {split!r} (expected train|held_out)"
        return result
    if not isinstance(stage, int) or stage < 0:
        result["detail"] = f"invalid stage {stage!r}"
        return result

    wad_path = resolve_wad(file_ref)
    if not wad_path.is_file():
        result["detail"] = f"WAD not found: {wad_path}"
        return result

    try:
        info = load_mapinfo(wad_path, map_name)
    except Exception as exc:
        result["detail"] = f"mapinfo failed: {exc}"
        return result

    result["sectors"] = info.num_sectors
    doors = len(info.door_sectors)
    locked = sum(1 for lock in info.door_sectors.values() if lock > 0)
    result["doors"] = f"{doors} ({locked} locked)"

    if info.num_sectors <= 0:
        result["detail"] = "map has 0 sectors"
        return result
    if info.num_sectors > MAX_SECTORS:
        result["detail"] = f"{info.num_sectors} sectors exceeds {MAX_SECTORS} limit"
        return result

    env = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            episode_ticks = max(2100, steps * 4 + 40)
            env = make_env(
                wad_path,
                map_name,
                episode_ticks=episode_ticks,
                render_buffers=render_buffers,
            )
            rng = np.random.default_rng(seed)
            obs, _ = env.reset(seed=seed)
            if not env.observation_space.contains(obs):
                result["detail"] = "reset observation failed space check"
                return result

            steps_run = 0
            outcome = "ok"
            for _ in range(steps):
                action = rng.integers(0, 2, size=env.action_space.shape, dtype=np.int8)
                obs, _, term, trunc, step_info = env.step(action)
                steps_run += 1
                if not env.observation_space.contains(obs):
                    result["steps"] = steps_run
                    result["detail"] = f"step {steps_run} observation failed space check"
                    return result
                if term or trunc:
                    if step_info.get("exit_reached"):
                        outcome = "exit"
                    elif term:
                        outcome = "died"
                    else:
                        outcome = "trunc"
                    break

            result["steps"] = steps_run
            if caught:
                result["detail"] = f"warning: {caught[0].message}"
                return result
            result["status"] = "PASS"
            result["detail"] = f"{info.format}, {outcome}"
            return result
    except Exception as exc:
        result["detail"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        if env is not None:
            env.close()


def print_table(rows: list[dict]) -> None:
    """Print a clean tabular summary of map verification results."""
    headers = ["STATUS", "STAGE", "SPLIT", "FILE", "MAP", "SECTORS", "DOORS", "STEPS", "DETAIL"]
    table_rows = [
        [
            r["status"],
            str(r["stage"]),
            str(r["split"]),
            r["file"],
            r["map"],
            str(r["sectors"]),
            str(r["doors"]),
            str(r["steps"]),
            r["detail"],
        ]
        for r in rows
    ]
    widths = [
        max(len(h), *(len(row[i]) for row in table_rows)) if table_rows else len(h)
        for i, h in enumerate(headers)
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for row in table_rows:
        print(fmt.format(*row))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="Path to maps.json")
    parser.add_argument("--steps", type=int, default=100, help="Random action steps per map")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--render-buffers", action="store_true", help="Also render RGB/depth buffers")
    args = parser.parse_args()

    if not args.catalog.is_file():
        print(f"ERROR: map catalog not found: {args.catalog}", file=sys.stderr)
        return 1

    try:
        entries = json.loads(args.catalog.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR: failed to parse {args.catalog}: {exc}", file=sys.stderr)
        return 1

    if not isinstance(entries, list) or not entries:
        print(f"ERROR: {args.catalog} must be a non-empty JSON list", file=sys.stderr)
        return 1

    # Check for duplicate (file, map) pairs or held-out leakage.
    seen_splits: dict[tuple[str, str], set[str]] = {}
    duplicates: list[str] = []
    for entry in entries:
        key = (str(entry.get("file", "")), str(entry.get("map", "")).upper())
        split = str(entry.get("split", ""))
        if key in seen_splits:
            duplicates.append(f"{key[0]}:{key[1]}")
        seen_splits.setdefault(key, set()).add(split)

    t0 = time.perf_counter()
    rows = [check_entry(entry, args.steps, args.seed, args.render_buffers) for entry in entries]
    elapsed = time.perf_counter() - t0

    print_table(rows)
    print()

    leakage = [f"{k[0]}:{k[1]}" for k, splits in seen_splits.items() if len(splits) > 1]
    if duplicates:
        print(f"ERROR: duplicate catalog entries: {', '.join(sorted(set(duplicates)))}", file=sys.stderr)
    if leakage:
        print(f"ERROR: train/held_out split leakage: {', '.join(sorted(leakage))}", file=sys.stderr)

    passed = sum(1 for r in rows if r["status"] == "PASS")
    failed = len(rows) - passed
    print(f"Summary: {passed}/{len(rows)} passed, {failed} failed ({elapsed:.2f}s)")

    if failed > 0 or duplicates or leakage:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
