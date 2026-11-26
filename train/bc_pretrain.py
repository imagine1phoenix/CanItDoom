#!/usr/bin/env python3
"""Recurrent Behavioral Cloning (BC) pretrainer from ExpertSpeedrunTeacher.

Collects clean, non-circling, goal-directed speedrun demonstrations across
Curriculum Stages 1-4 and two_rooms.wad using a training-only teacher bot,
and pre-trains the DoomActorCritic neural cluster:
- Blended cluster logits vs teacher actions (BCEWithLogitsLoss)
- Active specialist expert head (nav/enemy/door/progression) vs teacher actions
- ClusterRouter gating weights vs training-time situation labels (CrossEntropyLoss)
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

EXPERT_NAV = 0
EXPERT_ENEMY = 1
EXPERT_DOOR = 2
EXPERT_PROG = 3


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

    def act_with_expert(self, obs: dict) -> tuple[np.ndarray, int]:
        """Returns (action, active_expert_id) for training-only specialist supervision."""
        action = np.zeros(9, dtype=np.int8)
        priv = self.env.privileged_state()
        px, py, pang = priv["x"], priv["y"], priv["angle"]
        v_ents = obs["visible_entities"]

        enemy_active = False
        door_active = False
        prog_active = False

        # Combat: attack any visible hostile enemy in FOV (melee=1, hitscan=2, projectile=3)
        for i in range(len(v_ents["mask"])):
            if v_ents["mask"][i] == 1:
                etype = int(v_ents["entity_type"][i])
                if etype in (1, 2, 3):
                    enemy_active = True
                    feat = v_ents["features"][i]
                    sin_r, cos_r = feat[1], feat[2]
                    if cos_r > 0.05 and abs(sin_r) < 0.35:
                        action[6] = 1  # ATTACK
                    break
                if etype in (5, 6, 7, 8, 9) and v_ents["features"][i][0] < 0.45:
                    prog_active = True

        ceil = priv.get("sector_ceiling")
        has_door = 2 in self.env.mapinfo.door_sectors

        # Door interaction: if door is closed and close to door linedef
        if has_door and ceil is not None and len(ceil) > 2 and ceil[2] < 50.0:
            lines = self.env.mapinfo.sector_lines().get(2, [])
            if len(lines):
                d_x = lines[:, [0, 2]].mean()
                d_y = lines[:, [1, 3]].mean()
                door_dist = math.hypot(d_x - px, d_y - py)
                if door_dist < 90.0:
                    door_active = True
                if door_dist < 110.0 and self.use_cooldown == 0:
                    action[7] = 1  # USE
                    self.use_cooldown = 14

        if self.use_cooldown > 0:
            door_active = True
            self.use_cooldown -= 1

        # Key/item ray or keycard waypoint tracking
        ray_ht = obs["rays"]["hit_type"]
        ray_dist = obs["rays"]["distance"]
        if (self.door_idx > 0 and self.wp_idx < self.door_idx) or np.any((ray_ht == 5) & (ray_dist < 0.35)):
            prog_active = True
        if np.any(((ray_ht == 2) | (ray_ht == 3)) & (ray_dist < 0.09)):
            door_active = True

        # Advance along waypoints
        while self.wp_idx < len(self.wps):
            tx, ty = self.wps[self.wp_idx]
            dist = math.hypot(tx - px, ty - py)
            if self.wp_idx == self.door_idx and has_door and ceil is not None and len(ceil) > 2 and ceil[2] < 50.0:
                break
            if dist < 36.0 and self.wp_idx < len(self.wps) - 1:
                self.wp_idx += 1
            else:
                break

        tx, ty = self.wps[self.wp_idx] if self.wp_idx < len(self.wps) else self.wps[-1]
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

        if enemy_active:
            expert_id = EXPERT_ENEMY
        elif door_active:
            expert_id = EXPERT_DOOR
        elif prog_active:
            expert_id = EXPERT_PROG
        else:
            expert_id = EXPERT_NAV

        return action, expert_id

    def act(self, obs: dict) -> np.ndarray:
        action, _ = self.act_with_expert(obs)
        return action


ObsTeacherBot = ExpertSpeedrunTeacher


def _chunk_episode(
    ep_obs: list[dict],
    ep_act: list[np.ndarray],
    ep_done: list[float],
    ep_exp: list[int],
    seq_len: int,
) -> list[tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Chunks an episode into fixed-length segments, padding short episodes cleanly."""
    n = len(ep_obs)
    if n == 0:
        return []
    chunks = []
    if n < seq_len:
        pad_n = seq_len - n
        obs_seg = ep_obs + [ep_obs[-1]] * pad_n
        act_seg = np.concatenate([np.stack(ep_act, axis=0), np.zeros((pad_n, 9), dtype=np.int8)], axis=0)
        done_seg = np.concatenate([np.array(ep_done, dtype=np.float32), np.ones(pad_n, dtype=np.float32)], axis=0)
        exp_seg = np.concatenate([np.array(ep_exp, dtype=np.int64), np.zeros(pad_n, dtype=np.int64)], axis=0)
        mask_seg = np.concatenate([np.ones(n, dtype=np.float32), np.zeros(pad_n, dtype=np.float32)], axis=0)
        chunks.append((obs_seg, act_seg, done_seg, exp_seg, mask_seg))
        return chunks

    stride = max(1, seq_len // 2)
    for start_idx in range(0, n - seq_len + 1, stride):
        end_idx = start_idx + seq_len
        chunks.append(
            (
                ep_obs[start_idx:end_idx],
                np.stack(ep_act[start_idx:end_idx], axis=0),
                np.array(ep_done[start_idx:end_idx], dtype=np.float32),
                np.array(ep_exp[start_idx:end_idx], dtype=np.int64),
                np.ones(seq_len, dtype=np.float32),
            )
        )
    if (n - seq_len) % stride != 0:
        start_idx = n - seq_len
        chunks.append(
            (
                ep_obs[start_idx:n],
                np.stack(ep_act[start_idx:n], axis=0),
                np.array(ep_done[start_idx:n], dtype=np.float32),
                np.array(ep_exp[start_idx:n], dtype=np.int64),
                np.ones(seq_len, dtype=np.float32),
            )
        )
    return chunks


def collect_and_pretrain(
    steps: int = 40960,
    seq_len: int = 32,
    bc_epochs: int = 4,
    batch_size: int = 16,
    lr: float = 5e-4,
    device_name: str = "auto",
) -> None:
    """Collects expert demonstrations and pre-trains the DoomActorCritic neural cluster."""
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("\n=======================================================")
    print("  Behavioral Cloning (BC) Pretrainer — Neural Cluster")
    print(f"  Target: {steps:,} steps | seq_len: {seq_len} | device: {device}")
    print("=======================================================")

    catalog = load_curriculum_catalog()
    mixed_pool = [("maps/two_rooms.wad", "MAP01")]
    for s in (1, 2, 3, 4):
        mixed_pool.extend(catalog.get(s, [])[:15])

    print(f"Loaded {len(mixed_pool)} diverse training maps across stages 1-4 + two_rooms.")

    model = DoomActorCritic(num_actions=9).to(device)
    optimizer = Adam(model.parameters(), lr=lr, eps=1e-5)
    bce_none = nn.BCEWithLogitsLoss(reduction="none")
    ce_none = nn.CrossEntropyLoss(reduction="none")

    collected_steps = 0
    exits_seen = 0
    kills_seen = 0
    map_idx = 0
    t0 = time.time()

    seq_buffer: list[tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    print("\nCollecting expert demonstrations and specializing cluster engines...")

    while collected_steps < steps:
        wad, map_name = mixed_pool[map_idx % len(mixed_pool)]
        map_idx += 1

        env = make_env(wad, map_name=map_name, episode_ticks=2100, render_buffers=False)
        obs, _ = env.reset(seed=100 + map_idx)
        bot = ExpertSpeedrunTeacher(env)

        ep_obs, ep_act, ep_done, ep_exp = [], [], [], []
        for _ in range(350):
            act, exp_id = bot.act_with_expert(obs)
            next_obs, _, term, trunc, info = env.step(act)
            done = term or trunc

            ep_obs.append(obs)
            ep_act.append(act)
            ep_done.append(float(done))
            ep_exp.append(exp_id)

            obs = next_obs
            collected_steps += 1

            if done:
                if info.get("exit_reached"):
                    exits_seen += 1
                kills_seen += int(info.get("kills", 0))
                break

        env.close()
        seq_buffer.extend(_chunk_episode(ep_obs, ep_act, ep_done, ep_exp, seq_len))

        if len(seq_buffer) >= batch_size:
            indices = np.random.choice(len(seq_buffer), size=batch_size, replace=False)
            b_actions = torch.from_numpy(np.stack([seq_buffer[i][1] for i in indices], axis=0)).float().to(device)
            b_dones = torch.from_numpy(np.stack([seq_buffer[i][2] for i in indices], axis=0)).float().to(device)
            b_exp_ids = torch.from_numpy(np.stack([seq_buffer[i][3] for i in indices], axis=0)).long().to(device)
            b_valid = torch.from_numpy(np.stack([seq_buffer[i][4] for i in indices], axis=0)).float().to(device)
            done_mask = 1.0 - b_dones

            flat_obs_list = []
            for i in indices:
                flat_obs_list.extend(seq_buffer[i][0])
            flat_obs = collate_obs(flat_obs_list, device)
            b_obs_seq = {
                "rays": {
                    "distance": flat_obs["rays"]["distance"].view(batch_size, seq_len, -1),
                    "hit_type": flat_obs["rays"]["hit_type"].view(batch_size, seq_len, -1),
                },
                "visible_entities": {
                    "features": flat_obs["visible_entities"]["features"].view(batch_size, seq_len, 32, 5),
                    "entity_type": flat_obs["visible_entities"]["entity_type"].view(batch_size, seq_len, 32),
                    "mask": flat_obs["visible_entities"]["mask"].view(batch_size, seq_len, 32),
                },
                "vitals": flat_obs["vitals"].view(batch_size, seq_len, -1),
            }

            model.train()
            total_loss_val = 0.0
            gate_loss_val = 0.0
            valid_denom = b_valid.sum().clamp(min=1.0)

            for _ in range(bc_epochs):
                init_hx = model.initial_state(batch_size=batch_size, device=device)
                _, _, _, aux = model.evaluate_sequence(
                    b_obs_seq,
                    init_hx,
                    b_actions,
                    done_mask_seq=done_mask,
                    return_aux=True,
                )

                # 1. Blended cluster policy loss
                blend_bce = bce_none(aux["logits"], b_actions).mean(dim=-1)
                blend_loss = (blend_bce * b_valid).sum() / valid_denom

                # 2. Active specialist expert loss (sharpens each engine on its own domain)
                gather_idx = b_exp_ids.view(batch_size, seq_len, 1, 1).expand(-1, -1, 1, 9)
                active_expert_logits = aux["expert_logits"].gather(dim=2, index=gather_idx).squeeze(2)
                spec_bce = bce_none(active_expert_logits, b_actions).mean(dim=-1)
                spec_loss = (spec_bce * b_valid).sum() / valid_denom

                # 3. Neural router situation supervision + entropy floor to keep gates soft for PPO
                gate_ce = ce_none(aux["gate_logits"].view(-1, 4), b_exp_ids.view(-1)).view(batch_size, seq_len)
                gate_loss = (gate_ce * b_valid).sum() / valid_denom
                gate_ent = (aux["gate_entropy"] * b_valid).sum() / valid_denom

                loss = blend_loss + 0.5 * spec_loss + 0.2 * gate_loss - 0.05 * gate_ent
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                total_loss_val = float(blend_loss.item())
                gate_loss_val = float(gate_loss.item())

            if len(seq_buffer) > 256:
                seq_buffer = seq_buffer[-128:]

            sps = int(collected_steps / max(0.1, time.time() - t0))
            if collected_steps % 4000 < 350 or collected_steps >= steps:
                print(
                    f"BC Step {collected_steps:>6d}/{steps} | SPS {sps:>4d} | "
                    f"BCE {total_loss_val:.4f} | GateCE {gate_loss_val:.4f} | "
                    f"Exits {exits_seen} | Kills {kills_seen}"
                )

    sub_weights = REPO_ROOT / "submission" / "weights" / "model.pt"
    sub_weights.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(model.state_dict(), sub_weights)

    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        {
            "arch": "cluster",
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
    print("\n=======================================================")
    print("  BC Pretraining Complete!")
    print(f"  Total Steps: {collected_steps:,} | Exits: {exits_seen} | Kills: {kills_seen}")
    print(f"  Weights saved to: {sub_weights}")
    print(f"  Checkpoint saved to: {ckpt_dir / 'ckpt_latest.pt'}")
    print("=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="BC Pretrainer from ExpertSpeedrunTeacher")
    parser.add_argument("--steps", type=int, default=40960, help="Total demonstration steps to collect")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    args = parser.parse_args()
    collect_and_pretrain(steps=args.steps, device_name=args.device)


if __name__ == "__main__":
    main()
