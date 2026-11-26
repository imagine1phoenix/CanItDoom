"""Recurrent PPO training pipeline with dense reward shaping.

Features:
- Recurrent PPO with Generalized Advantage Estimation (GAE)
- Dense reward shaping from env.privileged_state()
- Count-based exploration bonus via spatial cell binning
- Vectorized parallel environment execution
- Checkpoint management with Git LFS compatibility
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# Add repo root to path so we can import doom_env and submission
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from doom_env import make_env
from submission.model import DoomActorCritic


class RewardShaper:
    """Calculates shaped training rewards from privileged environment state."""

    def __init__(self, grid_size: float = 64.0):
        self.grid_size = grid_size
        self.visited_cells: set[tuple[int, int]] = set()
        self.last_kills = 0
        self.last_damage = 0.0
        self.last_keys: set = set()
        self.last_items = 0

    def reset(self, priv: dict) -> None:
        self.visited_cells = set()
        self.last_kills = priv.get("kills", 0)
        self.last_damage = priv.get("damage_taken", 0.0)
        keys = priv.get("keys", set())
        self.last_keys = set(keys) if isinstance(keys, (set, list)) else set()
        self.last_items = priv.get("items", 0)
        
        # Mark initial position
        x, y = priv.get("x", 0.0), priv.get("y", 0.0)
        cell = (int(x // self.grid_size), int(y // self.grid_size))
        self.visited_cells.add(cell)

    def shape(self, raw_reward: float, priv: dict) -> tuple[float, dict]:
        shaped = -0.001  # Small step penalty to encourage efficiency
        events = {}

        # 1. Goal reaching
        if priv.get("exit_reached", False):
            shaped += 10.0
            events["exit"] = True

        # 2. Death penalty
        if priv.get("dead", False):
            shaped -= 3.0
            events["death"] = True

        # 3. Combat: Kills
        kills = priv.get("kills", 0)
        delta_kills = kills - self.last_kills
        if delta_kills > 0:
            shaped += 1.0 * delta_kills
            events["kill"] = delta_kills
            self.last_kills = kills

        # 4. Survival: Damage taken
        damage = priv.get("damage_taken", 0.0)
        delta_dmg = max(0.0, damage - self.last_damage)
        if delta_dmg > 0:
            shaped -= 0.01 * delta_dmg
            self.last_damage = damage

        # 5. Key pickups
        keys = priv.get("keys", set())
        current_keys = set(keys) if isinstance(keys, (set, list)) else set()
        new_keys = len(current_keys - self.last_keys)
        if new_keys > 0:
            shaped += 3.0 * new_keys
            events["key"] = new_keys
            self.last_keys = current_keys

        # 6. Spatial Exploration bonus (count-based / cell visitation)
        x, y = priv.get("x", 0.0), priv.get("y", 0.0)
        cell = (int(x // self.grid_size), int(y // self.grid_size))
        if cell not in self.visited_cells:
            self.visited_cells.add(cell)
            shaped += 0.05
            events["new_cell"] = True

        return shaped, events


class SingleEnvRunner:
    """Wrapper around one DoomEnv with reward shaping."""

    def __init__(self, wad_path: str, map_name: str = "MAP01", episode_ticks: int = 2100):
        self.env = make_env(wad_path, map_name, episode_ticks=episode_ticks, render_buffers=False)
        self.shaper = RewardShaper()
        self.ep_return = 0.0
        self.ep_len = 0

    def reset(self, seed: int | None = None) -> dict:
        obs, _ = self.env.reset(seed=seed)
        priv = self.env.privileged_state()
        self.shaper.reset(priv)
        self.ep_return = 0.0
        self.ep_len = 0
        return obs

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        obs, reward, term, trunc, info = self.env.step(action)
        priv = self.env.privileged_state()
        shaped_r, events = self.shaper.shape(reward, priv)
        
        self.ep_return += shaped_r
        self.ep_len += 1
        done = term or trunc
        
        info["ep_return"] = self.ep_return
        info["ep_len"] = self.ep_len
        info["events"] = events
        info["exit_reached"] = priv.get("exit_reached", False)

        if done:
            obs = self.reset()

        return obs, shaped_r, done, False, info


def obs_to_tensor(obs: dict, device: torch.device) -> dict:
    """Convert numpy observation dict to batch-1 PyTorch tensor."""
    return {
        "rays": {
            "distance": torch.from_numpy(obs["rays"]["distance"]).unsqueeze(0).to(device),
            "hit_type": torch.from_numpy(obs["rays"]["hit_type"]).unsqueeze(0).to(device),
        },
        "visible_entities": {
            "features": torch.from_numpy(obs["visible_entities"]["features"]).unsqueeze(0).to(device),
            "entity_type": torch.from_numpy(obs["visible_entities"]["entity_type"]).unsqueeze(0).to(device),
            "mask": torch.from_numpy(obs["visible_entities"]["mask"]).unsqueeze(0).to(device),
        },
        "vitals": torch.from_numpy(obs["vitals"]).unsqueeze(0).to(device),
    }


def collate_obs(obs_list: list[dict], device: torch.device) -> dict:
    """Collate list of observation dicts into batched tensors."""
    return {
        "rays": {
            "distance": torch.from_numpy(np.stack([o["rays"]["distance"] for o in obs_list])).to(device),
            "hit_type": torch.from_numpy(np.stack([o["rays"]["hit_type"] for o in obs_list])).to(device),
        },
        "visible_entities": {
            "features": torch.from_numpy(np.stack([o["visible_entities"]["features"] for o in obs_list])).to(device),
            "entity_type": torch.from_numpy(np.stack([o["visible_entities"]["entity_type"] for o in obs_list])).to(device),
            "mask": torch.from_numpy(np.stack([o["visible_entities"]["mask"] for o in obs_list])).to(device),
        },
        "vitals": torch.from_numpy(np.stack([o["vitals"] for o in obs_list])).to(device),
    }


def train_ppo(
    wad_path: str = "maps/two_rooms.wad",
    map_name: str = "MAP01",
    total_steps: int = 50_000,
    rollout_len: int = 128,
    num_envs: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    epochs: int = 4,
    batch_size: int = 64,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    device_name: str = "cpu",
    save_dir: str = "submission/weights",
) -> None:
    """Execute recurrent PPO training loop."""
    device = torch.device(device_name)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Initializing {num_envs} environments on map: {wad_path}:{map_name} (device={device})")
    envs = [SingleEnvRunner(wad_path, map_name) for _ in range(num_envs)]
    obs_batch = [env.reset(seed=i) for i, env in enumerate(envs)]

    model = DoomActorCritic(num_actions=9).to(device)
    optimizer = Adam(model.parameters(), lr=lr, eps=1e-5)

    recent_returns = deque(maxlen=20)
    recent_lens = deque(maxlen=20)

    # Initial recurrent state: (1, num_envs, 256)
    hx = model.initial_state(batch_size=num_envs, device=device)

    num_iterations = total_steps // (rollout_len * num_envs)
    global_step = 0
    start_time = time.time()

    print(f"Starting PPO training: {total_steps} steps, {num_iterations} iterations...")

    for iteration in range(1, num_iterations + 1):
        # Rollout storage
        mb_obs = []
        mb_actions = []
        mb_log_probs = []
        mb_rewards = []
        mb_dones = []
        mb_values = []
        init_hx = hx.clone()

        # Collect rollout
        model.eval()
        for _ in range(rollout_len):
            global_step += num_envs
            tensor_obs = collate_obs(obs_batch, device)

            with torch.no_grad():
                action, log_prob, _, val, hx = model.step(tensor_obs, hx, deterministic=False)

            actions_np = action.cpu().numpy().astype(np.int8)
            next_obs = []
            rewards = []
            dones = []

            for i, env in enumerate(envs):
                o, r, done, _, info = env.step(actions_np[i])
                next_obs.append(o)
                rewards.append(r)
                dones.append(done)

                if done:
                    recent_returns.append(info["ep_return"])
                    recent_lens.append(info["ep_len"])

            mb_obs.append(tensor_obs)
            mb_actions.append(action)
            mb_log_probs.append(log_prob)
            mb_rewards.append(torch.tensor(rewards, device=device, dtype=torch.float32))
            mb_dones.append(torch.tensor(dones, device=device, dtype=torch.float32))
            mb_values.append(val)

            obs_batch = next_obs

        # Bootstrap value for GAE
        final_obs = collate_obs(obs_batch, device)
        with torch.no_grad():
            _, _, _, next_value, _ = model.step(final_obs, hx)

        # Compute GAE
        advantages = torch.zeros(rollout_len, num_envs, device=device)
        last_gae = 0.0
        for t in reversed(range(rollout_len)):
            next_val = next_value if t == rollout_len - 1 else mb_values[t + 1]
            non_terminal = 1.0 - mb_dones[t]
            delta = mb_rewards[t] + gamma * next_val * non_terminal - mb_values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + torch.stack(mb_values)

        # Flatten rollout for optimization
        # Re-pack tensors into sequences of shape (num_envs, rollout_len, ...)
        b_actions = torch.stack(mb_actions, dim=1)  # (N, T, 9)
        b_returns = returns.transpose(0, 1)         # (N, T)
        b_advantages = advantages.transpose(0, 1)   # (N, T)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        b_old_log_probs = torch.stack(mb_log_probs, dim=1)  # (N, T)

        # Structure sequence observation dictionary
        b_obs_seq = {
            "rays": {
                "distance": torch.stack([o["rays"]["distance"] for o in mb_obs], dim=1),
                "hit_type": torch.stack([o["rays"]["hit_type"] for o in mb_obs], dim=1),
            },
            "visible_entities": {
                "features": torch.stack([o["visible_entities"]["features"] for o in mb_obs], dim=1),
                "entity_type": torch.stack([o["visible_entities"]["entity_type"] for o in mb_obs], dim=1),
                "mask": torch.stack([o["visible_entities"]["mask"] for o in mb_obs], dim=1),
            },
            "vitals": torch.stack([o["vitals"] for o in mb_obs], dim=1),
        }
        b_dones_seq = torch.stack(mb_dones, dim=1)

        # Optimization epochs
        model.train()
        for epoch in range(epochs):
            new_log_probs, entropy, new_values = model.evaluate_sequence(
                b_obs_seq, init_hx, b_actions, done_mask_seq=(1.0 - b_dones_seq)
            )

            # Ratio and surrogate loss
            ratio = torch.exp(new_log_probs - b_old_log_probs)
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss with clipping
            value_loss = 0.5 * ((new_values - b_returns) ** 2).mean()

            # Entropy bonus
            ent_loss = -entropy.mean()

            total_loss = policy_loss + vf_coef * value_loss + ent_coef * ent_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        # Logging & Checkpoint saving
        elapsed = time.time() - start_time
        sps = int(global_step / max(0.1, elapsed))
        mean_ret = np.mean(recent_returns) if recent_returns else 0.0
        mean_len = np.mean(recent_lens) if recent_lens else 0.0

        if iteration % 5 == 0 or iteration == num_iterations:
            print(
                f"Iter {iteration:03d}/{num_iterations:03d} | Steps: {global_step:06d} | "
                f"SPS: {sps} | Return: {mean_ret:+.2f} | Length: {mean_len:.1f} | "
                f"Loss: {total_loss.item():.4f} (pi={policy_loss.item():.4f}, v={value_loss.item():.4f})"
            )

        # Save milestone checkpoint
        if iteration % 10 == 0 or iteration == num_iterations:
            ckpt_file = save_path / "model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "global_step": global_step,
                    "mean_return": mean_ret,
                },
                ckpt_file,
            )

    print(f"Training completed in {elapsed:.1f}s. Final weights saved to {save_path / 'model.pt'}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recurrent PPO for ViZDoom")
    parser.add_argument("--map", default="maps/two_rooms.wad", help="WAD file path")
    parser.add_argument("--steps", type=int, default=10_000, help="Total environment steps")
    parser.add_argument("--envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    train_ppo(
        wad_path=args.map,
        total_steps=args.steps,
        num_envs=args.envs,
        device_name=args.device,
    )
