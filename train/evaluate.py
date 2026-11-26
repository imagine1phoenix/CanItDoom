#!/usr/bin/env python3
"""Offline evaluation harness and replay recorder for CanItDoom.

Runs a trained (or baseline) DoomAgent on held-out or specified maps without
reward shaping, computes official competition metrics from DoomEnv's
privileged_state(), exports a structured CSV, and optionally benchmarks
greedy vs. temperature-sampled action selection or records OpenCV replays.

Usage examples:
    python train/evaluate.py
    python train/evaluate.py --checkpoint submission/weights/model.pt --output eval_results.csv
    python train/evaluate.py --map maps/two_rooms.wad:MAP01 --episodes 3 --record-dir replays/
    python train/evaluate.py --sweep-temperatures 0.2 0.4 0.6 0.8 1.0 --output sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_DIR = REPO_ROOT / "env"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SUBMISSION_DIR = REPO_ROOT / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))
if ENV_DIR.is_dir() and str(ENV_DIR) not in sys.path:
    sys.path.append(str(ENV_DIR))

PLAYER_OPENING_HEIGHT = 56.0
DEFAULT_CELL_SIZE = 64.0
DEFAULT_TICKS = 2100
DEFAULT_MAPS_JSON = REPO_ROOT / "train" / "maps.json"
DEFAULT_FALLBACK_MAP = "maps/two_rooms.wad:MAP01"

CSV_COLUMNS = (
    "map_name",
    "tier",
    "exit_reached",
    "keys_found",
    "doors_opened",
    "unique_cells_visited",
    "kills",
    "damage_taken",
    "deaths",
    "agent_time_ms",
)


@dataclass(frozen=True)
class MapSpec:
    wad: Path
    map_name: str
    tier: int
    episode_ticks: int

    @property
    def label(self) -> str:
        return f"{self.wad.name}:{self.map_name}"


@dataclass
class EpisodeMetrics:
    map_name: str
    tier: int
    exit_reached: int
    keys_found: int
    doors_opened: int
    unique_cells_visited: int
    kills: int
    damage_taken: float
    deaths: int
    agent_time_ms: float
    steps: int = 0
    p99_time_ms: float = 0.0
    first_act_ms: float = 0.0
    max_subsequent_act_ms: float = 0.0
    reset_time_ms: float = 0.0
    policy_label: str = "default"
    seed: int = 0

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "map_name": self.map_name,
            "tier": self.tier,
            "exit_reached": self.exit_reached,
            "keys_found": self.keys_found,
            "doors_opened": self.doors_opened,
            "unique_cells_visited": self.unique_cells_visited,
            "kills": self.kills,
            "damage_taken": round(self.damage_taken, 3),
            "deaths": self.deaths,
            "agent_time_ms": round(self.agent_time_ms, 3),
        }


def resolve_wad_path(wad_raw: str) -> Path:
    """Resolve a WAD path relative to repo root, or fall back to bundled freedoom2.wad."""
    wad_path = Path(wad_raw)
    if wad_path.is_absolute():
        return wad_path
    candidate = (REPO_ROOT / wad_path).resolve()
    if candidate.is_file() or wad_raw.lower() not in ("freedoom2.wad", "iwad"):
        return candidate
    try:
        import vizdoom as vd

        iwad = Path(vd.__file__).resolve().parent / "freedoom2.wad"
        if iwad.is_file():
            return iwad
    except ImportError:
        pass
    return candidate


def _normalize_split(split: str) -> str:
    return split.strip().lower().replace("-", "_")


def parse_map_spec(spec: str, tier: int, ticks: int) -> MapSpec:
    wad_part, _, lump = spec.partition(":")
    return MapSpec(
        wad=resolve_wad_path(wad_part),
        map_name=(lump or "MAP01").upper(),
        tier=tier,
        episode_ticks=ticks,
    )


def load_map_specs(
    maps_arg: list[str] | None,
    maps_json: Path | None,
    split: str,
    default_tier: int,
    default_ticks: int,
) -> list[MapSpec]:
    if maps_arg:
        return [parse_map_spec(spec, default_tier, default_ticks) for spec in maps_arg]

    catalog_path = maps_json if maps_json is not None else DEFAULT_MAPS_JSON
    if catalog_path.is_file():
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        entries = raw["maps"] if isinstance(raw, dict) and isinstance(raw.get("maps"), list) else raw
        if not isinstance(entries, list):
            raise ValueError(f"Expected a JSON list or {{'maps': [...]}} in {catalog_path}")

        target_split = _normalize_split(split)
        specs: list[MapSpec] = []
        for entry in entries:
            entry_split = _normalize_split(str(entry.get("split", "train")))
            if target_split != "all" and entry_split != target_split:
                continue
            wad_raw = str(entry.get("file") or entry.get("wad") or "")
            if not wad_raw:
                raise ValueError(f"Map catalog entry missing 'file' in {catalog_path}: {entry}")
            specs.append(
                MapSpec(
                    wad=resolve_wad_path(wad_raw),
                    map_name=str(entry.get("map", "MAP01")).upper(),
                    tier=int(entry.get("tier", entry.get("stage", default_tier))),
                    episode_ticks=int(entry.get("episode_ticks", default_ticks)),
                )
            )
        if specs:
            return specs
        raise ValueError(f"No maps matched split={split!r} in {catalog_path}")

    print(
        f"[evaluate] Note: {catalog_path} not found; falling back to {DEFAULT_FALLBACK_MAP}",
        file=sys.stderr,
    )
    return [parse_map_spec(DEFAULT_FALLBACK_MAP, default_tier, default_ticks)]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _apply_checkpoint(agent: Any, checkpoint_path: Path, device: str) -> None:
    if hasattr(agent, "load_checkpoint") and callable(agent.load_checkpoint):
        agent.load_checkpoint(checkpoint_path)
        return

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} provided, but PyTorch is not installed."
        ) from exc

    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = payload
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model", "policy"):
            if key in payload and isinstance(payload[key], dict):
                state_dict = payload[key]
                break

    for attr in ("model", "policy", "net", "network"):
        module = getattr(agent, attr, None)
        if module is not None and hasattr(module, "load_state_dict"):
            module.load_state_dict(state_dict)
            if hasattr(module, "eval"):
                module.eval()
            return

    print(
        f"[evaluate] Warning: {type(agent).__name__} has no checkpoint hook or model attribute; "
        f"skipping explicit load of {checkpoint_path.name}.",
        file=sys.stderr,
    )


def build_agent(
    observation_space: Any,
    action_space: Any,
    device: str,
    checkpoint: Path | None,
) -> Any:
    from submission.agent import DoomAgent

    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Omit --checkpoint to evaluate the baseline agent."
        )

    sig = inspect.signature(DoomAgent.__init__)
    if checkpoint is not None and "checkpoint" in sig.parameters:
        agent = DoomAgent(observation_space, action_space, device, checkpoint=checkpoint)
    else:
        agent = DoomAgent(observation_space, action_space, device)
        if checkpoint is not None:
            _apply_checkpoint(agent, checkpoint, device)
    return agent


def configure_policy_mode(agent: Any, mode: str, temperature: float) -> bool:
    """Configure greedy/sampling mode if the agent exposes evaluation controls.

    Returns True if the agent accepted the configuration, False otherwise.
    """
    if mode == "auto":
        return True

    deterministic = mode == "greedy"
    if hasattr(agent, "set_eval_mode") and callable(agent.set_eval_mode):
        agent.set_eval_mode(deterministic=deterministic, temperature=temperature)
        return True

    configured = False
    if hasattr(agent, "deterministic"):
        agent.deterministic = deterministic
        configured = True
    if hasattr(agent, "greedy"):
        agent.greedy = deterministic
        configured = True
    if hasattr(agent, "temperature"):
        agent.temperature = float(temperature)
        configured = True
    return configured


class ReplayRecorder:
    """Writes an upscaled MP4 replay with an informative telemetry bar below the frame."""

    def __init__(self, output_path: Path, fps: int = 10, scale: int = 4):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV (cv2) is required for --record-dir. Install opencv-python-headless."
            ) from exc

        self._cv2 = cv2
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.scale = scale
        self.bar_height = 68
        self.width = 160 * scale
        self.height = 120 * scale + self.bar_height
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (self.width, self.height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open OpenCV VideoWriter for {output_path}")

    def write(
        self,
        rgb: np.ndarray,
        *,
        map_label: str,
        step: int,
        health: float,
        keys_found: int,
        doors_opened: int,
        cells: int,
        kills: int,
        action: np.ndarray,
    ) -> None:
        cv2 = self._cv2
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(bgr, (160 * self.scale, 120 * self.scale), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[: 120 * self.scale, :] = frame
        canvas[120 * self.scale :, :] = (24, 24, 28)

        line1 = (
            f"{map_label} | step {step:4d} | HP {health:3.0f} | "
            f"Keys {keys_found} | Doors {doors_opened} | Cells {cells} | Kills {kills}"
        )
        btn_names = ("FWD", "BCK", "STL", "STR", "TRL", "TRR", "ATK", "USE", "WPN")
        pressed = [name for name, val in zip(btn_names, action) if int(val) == 1]
        line2 = "Action: " + (" ".join(pressed) if pressed else "NONE")

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, line1, (10, 120 * self.scale + 25), font, 0.48, (235, 235, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, line2, (10, 120 * self.scale + 52), font, 0.48, (130, 220, 150), 1, cv2.LINE_AA)
        self._writer.write(canvas)

    def close(self) -> None:
        self._writer.release()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))
    return ordered[idx]


def evaluate_episode(
    agent: Any,
    spec: MapSpec,
    *,
    seed: int,
    cell_size: float,
    render_buffers: bool,
    policy_label: str,
    record_path: Path | None = None,
    fps: int = 10,
) -> EpisodeMetrics:
    from doom_env import make_env

    need_rgb = render_buffers or (record_path is not None)
    env = make_env(
        str(spec.wad),
        spec.map_name,
        episode_ticks=spec.episode_ticks,
        render_buffers=need_rgb,
    )
    recorder = ReplayRecorder(record_path, fps=fps) if record_path is not None else None
    try:
        obs, _ = env.reset(seed=seed)
        t_reset = time.perf_counter()
        agent.reset()
        reset_time_ms = (time.perf_counter() - t_reset) * 1000.0

        priv = env.privileged_state()
        init_ceil = priv["sector_ceiling"]
        init_floor = priv["sector_floor"]
        door_lock = priv["door_lock"]
        if door_lock is not None and init_ceil is not None and init_floor is not None:
            door_mask = door_lock >= 0
            initially_closed = door_mask & ((init_ceil - init_floor) < PLAYER_OPENING_HEIGHT)
            opened_mask = np.zeros_like(door_mask, dtype=bool)
        else:
            initially_closed = None
            opened_mask = None

        visited_cells: set[tuple[int, int]] = {
            (int(math.floor(priv["x"] / cell_size)), int(math.floor(priv["y"] / cell_size)))
        }
        max_keys = int(np.sum(priv["keys"]))
        latencies_ms: list[float] = []
        steps = 0

        while True:
            t0 = time.perf_counter()
            action = np.asarray(agent.act(obs))
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            obs, _reward, terminated, truncated, info = env.step(action)
            steps += 1
            priv = env.privileged_state()

            visited_cells.add(
                (int(math.floor(priv["x"] / cell_size)), int(math.floor(priv["y"] / cell_size)))
            )
            max_keys = max(max_keys, int(np.sum(priv["keys"])))

            if initially_closed is not None and opened_mask is not None:
                ceil = priv["sector_ceiling"]
                floor = priv["sector_floor"]
                if ceil is not None and floor is not None and init_ceil is not None:
                    now_open = (ceil - floor) >= PLAYER_OPENING_HEIGHT
                    raised = ceil > (init_ceil + 4.0)
                    opened_mask |= initially_closed & (now_open | raised)

            if recorder is not None:
                recorder.write(
                    obs["rgb"],
                    map_label=spec.label,
                    step=steps,
                    health=float(priv["health"]),
                    keys_found=max_keys,
                    doors_opened=int(opened_mask.sum()) if opened_mask is not None else 0,
                    cells=len(visited_cells),
                    kills=int(priv["kills"]),
                    action=action,
                )

            if terminated or truncated:
                break

        mean_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
        p99_ms = _percentile(latencies_ms, 0.99)
        first_act_ms = latencies_ms[0] if latencies_ms else 0.0
        max_sub_ms = max(latencies_ms[1:]) if len(latencies_ms) > 1 else 0.0
        return EpisodeMetrics(
            map_name=spec.label,
            tier=spec.tier,
            exit_reached=int(bool(info.get("exit_reached", priv["exit_reached"]))),
            keys_found=max_keys,
            doors_opened=int(opened_mask.sum()) if opened_mask is not None else 0,
            unique_cells_visited=len(visited_cells),
            kills=int(priv["kills"]),
            damage_taken=float(priv["damage_taken"]),
            deaths=int(bool(priv["dead"])),
            agent_time_ms=mean_ms,
            steps=steps,
            p99_time_ms=p99_ms,
            first_act_ms=first_act_ms,
            max_subsequent_act_ms=max_sub_ms,
            reset_time_ms=reset_time_ms,
            policy_label=policy_label,
            seed=seed,
        )
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


def write_csv(rows: list[EpisodeMetrics], output_path: Path, include_policy: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(CSV_COLUMNS)
    if include_policy:
        fieldnames.extend(["policy_mode", "seed"])
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = row.to_csv_row()
            if include_policy:
                data["policy_mode"] = row.policy_label
                data["seed"] = row.seed
            writer.writerow(data)


def _aggregate_group(group: list[EpisodeMetrics]) -> dict[str, Any]:
    n = len(group)
    return {
        "episodes": n,
        "exit_rate_pct": round(100.0 * sum(r.exit_reached for r in group) / n, 2),
        "mean_keys": round(sum(r.keys_found for r in group) / n, 3),
        "mean_doors": round(sum(r.doors_opened for r in group) / n, 3),
        "mean_cells": round(sum(r.unique_cells_visited for r in group) / n, 2),
        "mean_kills": round(sum(r.kills for r in group) / n, 3),
        "mean_damage": round(sum(r.damage_taken for r in group) / n, 2),
        "death_rate_pct": round(100.0 * sum(r.deaths for r in group) / n, 2),
        "mean_agent_time_ms": round(sum(r.agent_time_ms for r in group) / n, 3),
        "max_p99_time_ms": round(max(r.p99_time_ms for r in group), 3),
        "max_first_act_ms": round(max(r.first_act_ms for r in group), 3),
        "max_subsequent_act_ms": round(max(r.max_subsequent_act_ms for r in group), 3),
        "max_reset_time_ms": round(max(r.reset_time_ms for r in group), 3),
    }


def build_summary(
    rows: list[EpisodeMetrics],
    init_time_ms: float,
    *,
    policy_controls_supported: bool = True,
) -> dict[str, Any]:
    by_policy: dict[str, list[EpisodeMetrics]] = {}
    by_policy_tier: dict[str, dict[int, list[EpisodeMetrics]]] = {}
    for row in rows:
        by_policy.setdefault(row.policy_label, []).append(row)
        by_policy_tier.setdefault(row.policy_label, {}).setdefault(row.tier, []).append(row)

    return {
        "init_time_ms": round(init_time_ms, 3),
        "policy_controls_supported": policy_controls_supported,
        "policies": {
            label: {
                "overall": _aggregate_group(group),
                "by_tier": {
                    str(tier): _aggregate_group(tgroup)
                    for tier, tgroup in sorted(by_policy_tier[label].items())
                },
            }
            for label, group in by_policy.items()
        },
    }


def print_summary(
    rows: list[EpisodeMetrics],
    budget_ms: float,
    init_time_ms: float = 0.0,
    *,
    policy_controls_supported: bool = True,
) -> dict[str, Any]:
    if not rows:
        print("No episodes evaluated.")
        return {
            "init_time_ms": round(init_time_ms, 3),
            "policy_controls_supported": policy_controls_supported,
            "policies": {},
        }

    summary = build_summary(
        rows,
        init_time_ms,
        policy_controls_supported=policy_controls_supported,
    )
    print("\n=== Evaluation Summary ===")
    if init_time_ms > 0.0:
        print(f"Agent __init__ time: {init_time_ms / 1000.0:.3f} s")

    for label, pdata in summary["policies"].items():
        ov = pdata["overall"]
        print(
            f"[{label}] episodes={ov['episodes']} | exit={ov['exit_rate_pct']:.1f}% | "
            f"keys={ov['mean_keys']:.2f} | doors={ov['mean_doors']:.2f} | "
            f"cells={ov['mean_cells']:.1f} | kills={ov['mean_kills']:.2f} | "
            f"dmg={ov['mean_damage']:.1f} | deaths={ov['death_rate_pct']:.1f}% | "
            f"latency={ov['mean_agent_time_ms']:.2f} ms (p99 max {ov['max_p99_time_ms']:.2f} ms, "
            f"first max {ov['max_first_act_ms']:.2f} ms, sub max {ov['max_subsequent_act_ms']:.2f} ms, "
            f"reset max {ov['max_reset_time_ms']:.2f} ms, budget {budget_ms:.0f} ms)"
        )
        if len(pdata["by_tier"]) > 1:
            for tier_str, tdata in pdata["by_tier"].items():
                print(
                    f"    Tier {tier_str}: n={tdata['episodes']} | exit={tdata['exit_rate_pct']:.1f}% | "
                    f"keys={tdata['mean_keys']:.2f} | doors={tdata['mean_doors']:.2f} | "
                    f"cells={tdata['mean_cells']:.1f} | kills={tdata['mean_kills']:.2f} | "
                    f"dmg={tdata['mean_damage']:.1f} | deaths={tdata['death_rate_pct']:.1f}%"
                )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to saved model weights (.pt).")
    parser.add_argument("--output", type=Path, default=Path("eval_results.csv"), help="Destination CSV path.")
    parser.add_argument("--maps-json", type=Path, default=None, help="Path to map catalog JSON (default: train/maps.json).")
    parser.add_argument("--split", choices=("held_out", "train", "all"), default="held_out", help="Catalog split to evaluate.")
    parser.add_argument("--map", dest="maps", action="append", metavar="WAD[:MAP]", help="Explicit map(s) to evaluate.")
    parser.add_argument("--tier", type=int, default=0, help="Tier metadata when --map is passed directly.")
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS, help="Default episode tick cap.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of seeds per map.")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting random seed.")
    parser.add_argument("--device", default="auto", help="Inference device: auto | cpu | cuda.")
    parser.add_argument("--cell-size", type=float, default=DEFAULT_CELL_SIZE, help="Grid cell size in map units.")
    parser.add_argument("--no-render", action="store_true", help="Skip rendering rgb/depth buffers when not recording.")
    parser.add_argument("--policy-mode", choices=("auto", "greedy", "sample"), default="auto", help="Action selection mode.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature when --policy-mode=sample.")
    parser.add_argument(
        "--sweep-temperatures",
        type=float,
        nargs="+",
        default=None,
        help="Benchmark greedy plus categorical sampling across the listed temperatures (e.g. 0.2 0.4 0.6 0.8 1.0).",
    )
    parser.add_argument("--record-dir", type=Path, default=None, help="Optional directory to save OpenCV MP4 replays.")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS when --record-dir is used.")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch intra-op CPU threads (competition limit: 1).")
    parser.add_argument(
        "--include-meta",
        action="store_true",
        help="Append policy_mode and seed columns to the output CSV (always enabled for --sweep-temperatures).",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write aggregated per-policy, per-tier, and timing compliance JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        import torch

        torch.set_num_threads(max(1, args.threads))
    except ImportError:
        pass

    from doom_env import action_space, limits, observation_space

    device = resolve_device(args.device)
    specs = load_map_specs(args.maps, args.maps_json, args.split, args.tier, args.ticks)
    t_init = time.perf_counter()
    agent = build_agent(observation_space(), action_space(), device, args.checkpoint)
    init_time_ms = (time.perf_counter() - t_init) * 1000.0

    policies: list[tuple[str, str, float]] = []
    if args.sweep_temperatures:
        policies.append(("greedy", "greedy", 0.0))
        for temp in args.sweep_temperatures:
            policies.append((f"sample_T={temp:.2f}", "sample", float(temp)))
    else:
        label = args.policy_mode if args.policy_mode != "sample" else f"sample_T={args.temperature:.2f}"
        policies.append((label, args.policy_mode, float(args.temperature)))

    all_rows: list[EpisodeMetrics] = []
    warned_unconfigurable = False
    policy_controls_supported = True

    for policy_label, mode, temp in policies:
        if not configure_policy_mode(agent, mode, temp):
            policy_controls_supported = False
            if not warned_unconfigurable:
                extra = (
                    " Temperature sweep results will be identical until deterministic/temperature "
                    "controls are exposed on DoomAgent."
                    if args.sweep_temperatures
                    else ""
                )
                print(
                    f"[evaluate] Warning: {type(agent).__name__} has no deterministic/temperature control; "
                    f"using default act() behavior.{extra}",
                    file=sys.stderr,
                )
                warned_unconfigurable = True

        for spec in specs:
            for ep_idx in range(args.episodes):
                seed = args.seed_start + ep_idx
                record_path = None
                if args.record_dir is not None:
                    safe_name = spec.label.replace(":", "_").replace("/", "_")
                    safe_policy = policy_label.replace("=", "")
                    record_path = args.record_dir / f"{safe_name}_{safe_policy}_seed{seed}.mp4"
                metrics = evaluate_episode(
                    agent,
                    spec,
                    seed=seed,
                    cell_size=args.cell_size,
                    render_buffers=not args.no_render,
                    policy_label=policy_label,
                    record_path=record_path,
                    fps=args.fps,
                )
                all_rows.append(metrics)
                print(
                    f"  [{policy_label}] {metrics.map_name} (tier {metrics.tier}, seed {seed}): "
                    f"exit={metrics.exit_reached} keys={metrics.keys_found} doors={metrics.doors_opened} "
                    f"cells={metrics.unique_cells_visited} kills={metrics.kills} "
                    f"dmg={metrics.damage_taken:.1f} deaths={metrics.deaths} "
                    f"time={metrics.agent_time_ms:.2f}ms/step"
                )

    include_policy = bool(args.sweep_temperatures) or args.include_meta
    write_csv(all_rows, args.output, include_policy=include_policy)
    summary = print_summary(
        all_rows,
        budget_ms=limits.AGENT_MS_PER_STEP,
        init_time_ms=init_time_ms,
        policy_controls_supported=policy_controls_supported,
    )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Saved summary JSON to {args.summary_json}")
    print(f"\nSaved {len(all_rows)} row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
