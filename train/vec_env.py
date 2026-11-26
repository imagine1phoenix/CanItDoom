"""Multiprocessing Vectorized Environment Wrapper for ViZDoom.

Steps multiple DoomEnv instances in parallel OS processes so Python's GIL
does not bottleneck ViZDoom simulation throughput during PPO rollouts.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _worker_loop(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    map_pool: list[tuple[str, str]],
    episode_ticks: int,
    seed: int,
) -> None:
    """Worker process loop managing one MultiMapEnvRunner."""
    parent_remote.close()
    # Restrict each worker to 1 thread so workers don't oversubscribe CPU cores
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    from doom_env import make_env
    from train.ppo import RewardShaper

    current_idx = int(np.random.default_rng(seed).integers(0, len(map_pool))) if map_pool else 0
    wad, map_name = map_pool[current_idx]
    env = make_env(wad, map_name, episode_ticks=episode_ticks, render_buffers=False)
    shaper = RewardShaper()
    ep_return = 0.0
    ep_len = 0
    episodes_on_map = 0
    rng = np.random.default_rng(seed)

    def _do_reset(reset_seed: int | None = None) -> dict:
        nonlocal env, current_idx, ep_return, ep_len, episodes_on_map
        if len(map_pool) > 1 and episodes_on_map >= 5:
            env.close()
            current_idx = int(rng.integers(0, len(map_pool)))
            w, m = map_pool[current_idx]
            env = make_env(w, m, episode_ticks=episode_ticks, render_buffers=False)
            episodes_on_map = 0

        obs, _ = env.reset(seed=reset_seed)
        priv = env.privileged_state()
        shaper.reset(priv)
        ep_return = 0.0
        ep_len = 0
        episodes_on_map += 1
        return obs

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs = _do_reset(data)
                remote.send(obs)
            elif cmd == "step":
                obs, reward, term, trunc, info = env.step(data)
                priv = env.privileged_state()
                shaped_r, events = shaper.shape(reward, priv)
                ep_return += shaped_r
                ep_len += 1
                done = term or trunc

                info["ep_return"] = ep_return
                info["ep_len"] = ep_len
                info["events"] = events
                info["exit_reached"] = priv.get("exit_reached", False)
                info["kills"] = priv.get("kills", 0)

                if done:
                    obs = _do_reset()

                remote.send((obs, shaped_r, done, info))
            elif cmd == "update_pool":
                map_pool = data
                episodes_on_map = 5  # Trigger map rotation on next reset
                remote.send(True)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        env.close()
    except EOFError:
        env.close()


class SubprocVecEnv:
    """Runs N ViZDoom environments in parallel worker processes."""

    def __init__(
        self,
        map_pool: list[tuple[str, str]],
        num_envs: int,
        episode_ticks: int = 2100,
        base_seed: int = 0,
    ):
        self.num_envs = num_envs
        self.closed = False
        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.processes = []

        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            proc = ctx.Process(
                target=_worker_loop,
                args=(work_remote, remote, map_pool, episode_ticks, base_seed + i),
                daemon=True,
            )
            proc.start()
            work_remote.close()
            self.processes.append(proc)

    def reset(self, base_seed: int = 0) -> list[dict]:
        for i, remote in enumerate(self.remotes):
            remote.send(("reset", base_seed + i))
        return [remote.recv() for remote in self.remotes]

    def step(self, actions: np.ndarray) -> tuple[list[dict], list[float], list[bool], list[dict]]:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs_list, rewards, dones, infos = zip(*results)
        return list(obs_list), list(rewards), list(dones), list(infos)

    def update_pool(self, new_pool: list[tuple[str, str]]) -> None:
        for remote in self.remotes:
            remote.send(("update_pool", new_pool))
        for remote in self.remotes:
            remote.recv()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for proc in self.processes:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
