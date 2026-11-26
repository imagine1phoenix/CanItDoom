#!/usr/bin/env python3
"""Recurrent Behavioral Cloning (BC) pretrainer from ExpertSpeedrunTeacher.

Collects clean, non-circling, goal-directed speedrun demonstrations across
Curriculum Stages 1-4 and two_rooms.wad using an expert teacher bot (in train/),
and trains DoomActorCritic's feature extractors + GRU + actor head via BCEWithLogitsLoss.

Result: Warm-starts submission/weights/model.pt so the neural network policy
natively knows how to sprint forward, navigate corridors, interact with doors,
pick up keycards, and engage enemies without spinning in circles.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ENV_DIR = REPO_ROOT / "env"
if str(ENV_DIR) not in sys.path:
    sys.path.append(str(ENV_DIR))

from doom_env import make_env
from submission.model import DoomActorCritic
from train.ppo import collate_obs
from train.run import _atomic_torch_save, load_curriculum_catalog


def get_sector_center(env, sec_id: int) -> tuple[float, float]:
    """Computes mean (x, y) center coordinates for a sector polygon."""
    lines = env.mapinfo.sector_lines().get(sec_id, [])
    if len(lines) == 0:
        return (0.0, 0.0)
    pts = np.vstack([lines[:, :2], lines[:, 2:]])
    return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))


def get_portals(env) -> dict[tuple[int, int], tuple[float, float]]:
    """Extracts midpoint coordinates of 2-sided portals between adjacent sectors."""
    lines = env.mapinfo.lines
    portals = {}
    for i in range(len(lines.front)):
        f, b = lines.front[i], lines.back[i]
        if f != -1 and b != -1:
            mid = tuple((lines.p1[i] + lines.p2[i]) / 2.0)
            pair = tuple(sorted([f, b]))
            if pair not in portals:
                portals[pair] = (float(mid[0]), float(mid[1]))
    return portals


class ExpertSpeedrunTeacher:
    """Fast, goal-directed speedrun teacher bot (training-only helper)."""

    def __init__(self, env):
        self.env = env
        self.portals = get_portals(env)
        self.sec_centers = {s: get_sector_center(env, s) for s in range(env.mapinfo.num_sectors)}
        self.use_cooldown = 0
        self.door_idx = -1
        self.wps: list[tuple[float, float]] = []
        self.wp_idx = 0
        self.plan_waypoints()

    def plan_waypoints(self) -> None:
        self.wps = []
        exit_pos = self.sec_centers.get(3, (900.0, 300.0))
        is_two_rooms = exit_pos[0] < 800.0

        if is_two_rooms:
            self.door_idx = 0
            self.wps = [
                (430.0, 352.0),  # In front of door west
                (540.0, 352.0),  # Past door east
                (750.0, 352.0),  # East side of room 2 (past dividing wall)
                (750.0, 110.0),  # South corridor
                (672.0, 96.0),   # Exit pad
            ]
        else:
            # Procedural curriculum map (Stages 1-4)
            self.door_idx = 0
            state = self.env._game.get_state()
            if state is not None:
                for obj in state.objects:
                    if "Card" in obj.name or "Key" in obj.name:
                        self.wps.append((float(obj.position_x), float(obj.position_y)))
                        self.door_idx = 1
                        break

            p_door_in = self.portals.get((0, 2), (480.0, 240.0))
            p_door_out = self.portals.get((1, 2), (540.0, 240.0))
            self.wps.append((p_door_in[0] - 25.0, p_door_in[1]))
            self.wps.append((p_door_out[0] + 25.0, p_door_out[1]))
            self.wps.append(exit_pos)

        self.wp_idx = 0

    def reset(self) -> None:
        self.use_cooldown = 0
        self.wp_idx = 0
        self.plan_waypoints()

    def act(self, obs: dict) -> np.ndarray:
        action = np.zeros(9, dtype=np.int8)
        priv = self.env.privileged_state()
        px, py, pang = priv["x"], priv["y"], priv["angle"]
        v_ents = obs["visible_entities"]

        # Combat: attack enemy in FOV while advancing
        for i in range(len(v_ents["mask"])):
            if v_ents["mask"][i] == 1 and v_ents["entity_type"][i] == 1:
                feat = v_ents["features"][i]
                sin_r, cos_r = feat[1], feat[2]
                if cos_r > 0.05 and abs(sin_r) < 0.35:
                    action[6] = 1  # ATTACK!
                break

        ceil = priv.get("sector_ceiling")
        has_door = 2 in self.env.mapinfo.door_sectors

        # Door interaction: if door is closed and close to door linedef
        if has_door and ceil is not None and len(ceil) > 2 and ceil[2] < 50.0:
            lines = self.env.mapinfo.sector_lines().get(2, [])
            if len(lines):
                d_x = lines[:, [0, 2]].mean()
                d_y = lines[:, [1, 3]].mean()
                if math.hypot(d_x - px, d_y - py) < 110.0:
                    if self.use_cooldown == 0:
                        action[7] = 1  # USE
                        self.use_cooldown = 14

        if self.use_cooldown > 0:
            self.use_cooldown -= 1

        # Advance along waypoints
        while self.wp_idx < len(self.wps):
            tx, ty = self.wps[self.wp_idx]
            dist = math.hypot(tx - px, ty - py)
            # Only hold at the door waypoint if the door is currently closed
            if self.wp_idx == self.door_idx and has_door and ceil is not None and len(ceil) > 2 and ceil[2] < 50.0:
                break
            if dist < 36.0 and self.wp_idx < len(self.wps) - 1:
                self.wp_idx += 1
            else:
                break

        if self.wp_idx < len(self.wps):
            tx, ty = self.wps[self.wp_idx]
        else:
            tx, ty = self.wps[-1]

        dx = tx - px
        dy = ty - py
        target_deg = math.degrees(math.atan2(dy, dx)) % 360
        diff = (target_deg - pang + 180) % 360 - 180

        if diff > 4:
            action[4] = 1  # TURN_LEFT
        elif diff < -4:
            action[5] = 1  # TURN_RIGHT

        if abs(diff) < 45:
            action[0] = 1  # MOVE_FORWARD

        return action


# Backward compatibility alias
ObsTeacherBot = ExpertSpeedrunTeacher


def collect_and_pretrain(
    steps: int = 40960,
    seq_len: int = 64,
    bc_epochs: int = 4,
    batch_size: int = 16,
    lr: float = 5e-4,
    device_name: str = "auto",
) -> None:
    """Collects expert speedrun demonstrations and pre-trains DoomActorCritic."""
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"\n=======================================================")
    print(f"  Behavioral Cloning (BC) Pretrainer")
    print(f"  Target: {steps:,} steps | seq_len: {seq_len} | device: {device}")
    print(f"=======================================================")

    catalog = load_curriculum_catalog()
    mixed_pool = [("maps/two_rooms.wad", "MAP01")]
    for s in (1, 2, 3, 4):
        mixed_pool.extend(catalog.get(s, [])[:15])

    print(f"Loaded {len(mixed_pool)} diverse training maps across stages 1-4 + two_rooms.")

    # Initialize model and optimizer
    model = DoomActorCritic(num_actions=9).to(device)
    optimizer = Adam(model.parameters(), lr=lr, eps=1e-5)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    collected_steps = 0
    exits_seen = 0
    kills_seen = 0
    map_idx = 0
    t0 = time.time()

    # Pre-allocation buffer for sequence training
    seq_obs_list = []
    seq_act_list = []
    seq_done_list = []

    print("\nCollecting expert demonstrations and training recurrent trunk...")

    while collected_steps < steps:
        wad, map_name = mixed_pool[map_idx % len(mixed_pool)]
        map_idx += 1

        env = make_env(wad, map_name=map_name, episode_ticks=2100)
        obs, _ = env.reset(seed=100 + map_idx)
        bot = ExpertSpeedrunTeacher(env)

        ep_obs, ep_act, ep_done = [], [], []

        for step in range(350):
            act = bot.act(obs)
            next_obs, reward, term, trunc, info = env.step(act)
            done = term or trunc

            ep_obs.append(obs)
            ep_act.append(act)
            ep_done.append(float(done))

            obs = next_obs
            collected_steps += 1

            if done:
                if info.get("exit_reached"):
                    exits_seen += 1
                kills_seen += int(info.get("kills", 0))
                break

        env.close()

        # Chunk episode into seq_len segments
        for start_idx in range(0, len(ep_obs) - seq_len + 1, seq_len // 2):
            seq_obs_list.append(ep_obs[start_idx : start_idx + seq_len])
            seq_act_list.append(np.stack(ep_act[start_idx : start_idx + seq_len], axis=0))
            seq_done_list.append(np.array(ep_done[start_idx : start_idx + seq_len], dtype=np.float32))

        # Train mini-batches when enough sequences are accumulated
        if len(seq_obs_list) >= batch_size:
            indices = np.random.choice(len(seq_obs_list), size=batch_size, replace=False)
            b_actions_np = np.stack([seq_act_list[i] for i in indices], axis=0)  # (B, T, 9)
            b_dones_np = np.stack([seq_done_list[i] for i in indices], axis=0)    # (B, T)

            b_actions = torch.from_numpy(b_actions_np).float().to(device)
            b_dones = torch.from_numpy(b_dones_np).float().to(device)
            done_mask = 1.0 - b_dones

            # Collate batch sequences for model forward
            flat_obs_list = []
            for i in indices:
                flat_obs_list.extend(seq_obs_list[i])
            b_obs = collate_obs(flat_obs_list, device)

            obs_seq = {
                "rays": {
                    "distance": b_obs["rays"]["distance"].view(batch_size, seq_len, 64),
                    "hit_type": b_obs["rays"]["hit_type"].view(batch_size, seq_len, 64),
                },
                "visible_entities": {
                    "features": b_obs["visible_entities"]["features"].view(batch_size, seq_len, 32, 5),
                    "entity_type": b_obs["visible_entities"]["entity_type"].view(batch_size, seq_len, 32),
                    "mask": b_obs["visible_entities"]["mask"].view(batch_size, seq_len, 32),
                },
                "vitals": b_obs["vitals"].view(batch_size, seq_len, 17),
            }

            model.train()
            total_loss_val = 0.0

            for _ in range(bc_epochs):
                mem_grids, mem_lms, mem_odom = model.spatial_memory.build_sequence_inputs(
                    obs_seq, actions_seq=b_actions, done_mask_seq=done_mask
                )
                feats = model.extract_features(
                    obs_seq, mem_grid=mem_grids, mem_landmarks=mem_lms, mem_odom=mem_odom
                )  # (B, T, hidden_dim)

                init_hx = model.initial_state(batch_size=batch_size, device=device)
                outputs = []
                hx = init_hx
                for t in range(seq_len):
                    if t > 0:
                        hx = hx * done_mask[:, t - 1].view(1, batch_size, 1)
                    out, hx = model.gru(feats[:, t : t + 1, :], hx)
                    outputs.append(out)

                gru_out = torch.cat(outputs, dim=1)  # (B, T, hidden_dim)
                logits = model.actor_head(gru_out)   # (B, T, 9)

                loss = bce_loss_fn(logits, b_actions)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                total_loss_val = float(loss.item())

            # Prune buffer to keep recent demonstrations
            if len(seq_obs_list) > 256:
                seq_obs_list = seq_obs_list[-128:]
                seq_act_list = seq_act_list[-128:]
                seq_done_list = seq_done_list[-128:]

            sps = int(collected_steps / max(0.1, time.time() - t0))
            if collected_steps % 4000 < 350 or collected_steps >= steps:
                print(
                    f"BC Step {collected_steps:>6d}/{steps} | SPS {sps:>4d} | "
                    f"BCE Loss: {total_loss_val:.4f} | Exits: {exits_seen} | Kills: {kills_seen}"
                )

    # Save to submission weights
    sub_weights = REPO_ROOT / "submission" / "weights" / "model.pt"
    sub_weights.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(model.state_dict(), sub_weights)

    # Save initial checkpoint for PPO resumption
    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": 0,
            "stage": 1,
            "stage_steps": 0,
            "mean_return": 0.0,
            "best_return": -999.0,
            "timestamp": float(time.time()),
        },
        ckpt_dir / "ckpt_latest.pt",
    )
    print(f"\n=======================================================")
    print(f"  BC Pretraining Complete!")
    print(f"  Total Steps: {collected_steps:,} | Exits: {exits_seen} | Kills: {kills_seen}")
    print(f"  Weights saved to: {sub_weights}")
    print(f"  Checkpoint saved to: {ckpt_dir / 'ckpt_latest.pt'}")
    print(f"=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="BC Pretrainer from ExpertSpeedrunTeacher")
    parser.add_argument("--steps", type=int, default=40960, help="Total demonstration steps to collect")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    args = parser.parse_args()
    collect_and_pretrain(steps=args.steps, device_name=args.device)


if __name__ == "__main__":
    main()
