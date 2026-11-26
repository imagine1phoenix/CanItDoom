#!/usr/bin/env python3
"""Recurrent Behavioral Cloning (BC) pretrainer from ExpertSpeedrunTeacher.

Collects clean, non-circling, goal-directed speedrun demonstrations across
Curriculum Stages 1-4 and two_rooms.wad using a training-only teacher bot,
and pre-trains the DoomActorCritic neural cluster:
- Blended cluster logits vs teacher actions (BCEWithLogitsLoss)
- Domain-masked specialist expert heads (Nav, Enemy, Door, Progression)
- Action-wise ClusterRouter gating weights vs training-time situation labels
- Full V1 EgoSpatialMemory topological map and landmark odometry unrolling
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
        self.last_enemy_dir = 0
        self.enemy_pursuit_timer = 0
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
        self.last_enemy_dir = 0
        self.enemy_pursuit_timer = 0
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

        # 1. Combat & Enemy Tracking: Find nearest visible hostile enemy
        closest_enemy_idx = -1
        min_enemy_dist = 999.0
        for i in range(len(v_ents["mask"])):
            if v_ents["mask"][i] == 1:
                etype = int(v_ents["entity_type"][i])
                if etype in (1, 2, 3, 4):
                    d = float(v_ents["features"][i][0])
                    if d < min_enemy_dist:
                        min_enemy_dist = d
                        closest_enemy_idx = i
                elif etype in (5, 6, 7, 8, 9) and v_ents["features"][i][0] < 0.45:
                    prog_active = True

        if closest_enemy_idx >= 0:
            enemy_active = True
            feat = v_ents["features"][closest_enemy_idx]
            dist_norm, sin_r, cos_r = feat[0], feat[1], feat[2]

            # Aim: Turn actively to face the enemy
            if sin_r > 0.05:
                action[4] = 1  # TURN_LEFT
                self.last_enemy_dir = 1
            elif sin_r < -0.05:
                action[5] = 1  # TURN_RIGHT
                self.last_enemy_dir = -1

            # Fire: When enemy is aligned in front cone
            if cos_r > 0.0 and abs(sin_r) < 0.28:
                action[6] = 1  # ATTACK

            # Pursue / Combat Movement:
            if dist_norm > 0.16 and cos_r > 0.0:
                action[0] = 1  # MOVE_FORWARD
                action[8] = 1  # SPEED
            elif dist_norm < 0.08:
                action[1] = 1  # Backpedal from close hostiles
            else:
                if abs(sin_r) < 0.20:
                    action[2 if self.last_enemy_dir < 0 else 3] = 1  # Strafe

            self.enemy_pursuit_timer = 20  # Keep pursuit active if line-of-sight breaks

        elif self.enemy_pursuit_timer > 0 and self.last_enemy_dir != 0:
            # Occlusion tracking: Enemy ducked behind a pillar/corner, round the corner to re-acquire!
            self.enemy_pursuit_timer -= 1
            enemy_active = True
            action[0] = 1  # MOVE_FORWARD
            action[8] = 1  # SPEED
            if self.last_enemy_dir > 0:
                action[4] = 1  # TURN_LEFT around corner
            else:
                action[5] = 1  # TURN_RIGHT around corner

        # In combat mode: prioritize eliminating enemies before advancing to the exit pad!
        if enemy_active:
            return action, EXPERT_ENEMY

        ceil = priv.get("sector_ceiling")
        has_door = 2 in self.env.mapinfo.door_sectors

        if has_door:
            lines = self.env.mapinfo.sector_lines().get(2, np.zeros((0, 4)))
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
            if self.wp_idx == self.door_idx and has_door and ceil is not None and len(ceil) > 2 and ceil[2] < 100.0:
                break
            if dist < 42.0:
                self.wp_idx += 1
            else:
                break

        if self.wp_idx < len(self.wps):
            tx, ty = self.wps[self.wp_idx]
            dx, dy = tx - px, ty - py
            t_ang = (math.degrees(math.atan2(dy, dx))) % 360.0
            ang_diff = (t_ang - pang + 180.0) % 360.0 - 180.0

            action[0] = 1  # MOVE_FORWARD
            action[8] = 1  # SPEED (sprint)

            if ang_diff > 12.0:
                action[4] = 1  # TURN_LEFT
            elif ang_diff < -12.0:
                action[5] = 1  # TURN_RIGHT

            # Sharp turn braking to prevent overshoot
            if abs(ang_diff) > 75.0:
                action[0] = 0
        else:
            action[0] = 1
            action[8] = 1

        # Determine dominant specialist engine for supervised router label
        if enemy_active:
            exp_id = EXPERT_ENEMY
        elif door_active:
            exp_id = EXPERT_DOOR
        elif prog_active:
            exp_id = EXPERT_PROG
        else:
            exp_id = EXPERT_NAV

        return action, exp_id


def _chunk_episode(
    ep_obs: list[dict],
    ep_act: list[np.ndarray],
    ep_done: list[float],
    ep_exp: list[int],
    seq_len: int,
) -> list[tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Chunks an episode trajectory into fixed seq_len windows with valid masks."""
    chunks = []
    T = len(ep_obs)
    for start in range(0, T, seq_len):
        end = min(start + seq_len, T)
        length = end - start
        obs_chunk = ep_obs[start:end]
        act_chunk = np.zeros((seq_len, 9), dtype=np.float32)
        done_chunk = np.zeros(seq_len, dtype=np.float32)
        exp_chunk = np.zeros(seq_len, dtype=np.int64)
        valid_chunk = np.zeros(seq_len, dtype=np.float32)

        act_chunk[:length] = np.array(ep_act[start:end], dtype=np.float32)
        done_chunk[:length] = np.array(ep_done[start:end], dtype=np.float32)
        exp_chunk[:length] = np.array(ep_exp[start:end], dtype=np.int64)
        valid_chunk[:length] = 1.0

        if length < seq_len:
            obs_chunk = obs_chunk + [obs_chunk[-1]] * (seq_len - length)

        chunks.append((obs_chunk, act_chunk, done_chunk, exp_chunk, valid_chunk))
    return chunks


def collect_and_pretrain(
    steps: int = 40960,
    seq_len: int = 32,
    batch_size: int = 16,
    lr: float = 3e-4,
    device_name: str = "auto",
    bc_epochs: int = 2,
) -> None:
    """Collects expert speedrun demonstrations and pre-trains the neural cluster policy."""
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("=======================================================")
    print(f"  Neural Cluster BC Warm-Start Pretrainer")
    print(f"  Target Steps: {steps:,} | SeqLen: {seq_len} | Device: {device}")
    print("=======================================================")

    catalog = load_curriculum_catalog()
    all_wads = []
    stages = [catalog.get(s, []) for s in (1, 2, 3, 4)]
    max_len = max(len(st) for st in stages if len(st) > 0)
    for i in range(max_len):
        for st in stages:
            if i < len(st):
                all_wads.append(st[i])

    two_rooms = REPO_ROOT / "env" / "doom_env" / "maps" / "two_rooms.wad"
    if two_rooms.exists():
        all_wads.append((str(two_rooms), "MAP01"))

    if not all_wads:
        all_wads = [str(two_rooms)]

    model = DoomActorCritic(num_actions=9).to(device)
    optimizer = Adam(model.parameters(), lr=lr, eps=1e-5)
    bce_none = nn.BCEWithLogitsLoss(reduction="none")
    ce_none = nn.CrossEntropyLoss(reduction="none")

    # Domain Action Ownership Masks (MultiBinary 9):
    # [0: FWD, 1: BCK, 2: LFT, 3: RGT, 4: TL, 5: TR, 6: ATK, 7: USE, 8: SPD]
    # Nav: Locomotion, turning, speed (does NOT own ATK or USE)
    # Enemy: Combat strafing, turning, attack, speed
    # Door: Approach, turning, USE, speed
    # Progression: Search, turning, pickup, USE, speed
    specialist_masks = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],  # Nav (0)
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],  # Enemy (1)
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],  # Door (2)
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],  # Progression (3)
        ],
        device=device,
        dtype=torch.float32,
    )

    collected_steps = 0
    exits_seen = 0
    kills_seen = 0
    t0 = time.time()
    seq_buffer = []

    wad_cycle = 0
    while collected_steps < steps:
        wad_cycle += 1
        item = all_wads[wad_cycle % len(all_wads)]
        if isinstance(item, tuple):
            wad_path, map_name = str(item[0]), str(item[1])
        else:
            wad_path, map_name = str(item), "MAP01"
        try:
            env = make_env(wad=wad_path, map_name=map_name, render_buffers=False, render_mode=None)
        except Exception as e:
            print(f"[Warning] Failed to load {wad_path}: {e}")
            continue

        obs, _ = env.reset()
        teacher = ExpertSpeedrunTeacher(env)

        ep_obs: list[dict] = []
        ep_act: list[np.ndarray] = []
        ep_done: list[float] = []
        ep_exp: list[int] = []

        done = False
        ep_steps = 0
        ep_kills = 0
        while not done and ep_steps < 750:
            ep_steps += 1
            collected_steps += 1

            action, exp_id = teacher.act_with_expert(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            if reward > 5.0 or info.get("exit_reached", False):
                exits_seen += 1
            cur_kills = int(env.privileged_state().get("kills", 0))
            if cur_kills > ep_kills:
                kills_seen += (cur_kills - ep_kills)
                ep_kills = cur_kills

            ep_obs.append(obs)
            ep_act.append(action)
            ep_done.append(1.0 if done else 0.0)
            ep_exp.append(exp_id)
            obs = next_obs

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

                # 2. Domain-masked active specialist expert loss
                gather_idx = b_exp_ids.view(batch_size, seq_len, 1, 1).expand(-1, -1, 1, 9)
                active_expert_logits = aux["expert_logits"].gather(dim=2, index=gather_idx).squeeze(2)
                domain_mask = specialist_masks[b_exp_ids]  # (B, T, 9)
                spec_bce = bce_none(active_expert_logits, b_actions) * domain_mask
                spec_loss = (spec_bce.sum(dim=-1) * b_valid).sum() / (domain_mask.sum(dim=-1) * b_valid).sum().clamp(min=1.0)

                # 3. Neural router situation supervision + entropy floor to keep gates soft
                gate_summary_logits = aux["gate_logits"].mean(dim=-1)  # (B, T, 4)
                gate_ce = ce_none(gate_summary_logits.view(-1, 4), b_exp_ids.view(-1)).view(batch_size, seq_len)
                gate_loss = (gate_ce * b_valid).sum() / valid_denom
                gate_ent = (aux["gate_entropy"] * b_valid).sum() / valid_denom

                # 4. Anti-circling spin penalty (penalize simultaneous turn left + turn right logit inflation)
                spin_penalty = (
                    (torch.sigmoid(aux["logits"][..., 4]) * torch.sigmoid(aux["logits"][..., 5])) * b_valid
                ).sum() / valid_denom

                loss = blend_loss + 0.5 * spec_loss + 0.2 * gate_loss - 0.05 * gate_ent + 0.1 * spin_penalty
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
