"""Tests for V1 relative odometry, ego memory map, landmark slots, and Task P.5 trajectory match."""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from doom_env import make_env
from submission.agent import DoomAgent
from submission.memory import MAP_CHANNELS, MAP_GRID_SIZE, NUM_LANDMARK_SLOTS
from submission.model import DoomActorCritic


def _obs_to_tensor(obs: dict) -> dict[str, torch.Tensor]:
    return {
        "rays": {
            "distance": torch.from_numpy(obs["rays"]["distance"]).unsqueeze(0),
            "hit_type": torch.from_numpy(obs["rays"]["hit_type"]).unsqueeze(0),
        },
        "visible_entities": {
            "features": torch.from_numpy(obs["visible_entities"]["features"]).unsqueeze(0),
            "entity_type": torch.from_numpy(obs["visible_entities"]["entity_type"]).unsqueeze(0),
            "mask": torch.from_numpy(obs["visible_entities"]["mask"]).unsqueeze(0),
        },
        "vitals": torch.from_numpy(obs["vitals"]).unsqueeze(0),
    }


def test_model_and_memory_shapes_and_reset():
    torch.set_num_threads(1)
    env = make_env("maps/two_rooms.wad", "MAP01", render_buffers=False)
    try:
        agent = DoomAgent(env.observation_space, env.action_space, "cpu")
        obs, _ = env.reset(seed=0)
        agent.reset()
        assert agent.hx is None and agent.mem_state is None and agent.prev_action is None

        latencies_ms = []
        for _ in range(20):
            t0 = time.perf_counter()
            act = agent.act(obs)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            assert act.shape == (9,) and set(np.unique(act)) <= {0, 1}
            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break

        assert agent.mem_state is not None
        assert agent.mem_state.grid.shape == (1, MAP_CHANNELS, MAP_GRID_SIZE, MAP_GRID_SIZE)
        assert agent.mem_state.landmarks.shape == (1, NUM_LANDMARK_SLOTS, 6)
        assert np.mean(latencies_ms[1:]) < 40.0, f"Too slow on 1 thread: {np.mean(latencies_ms[1:]):.2f} ms"

        # Reset must wipe all memory buffers completely
        agent.reset()
        assert agent.hx is None and agent.mem_state is None and agent.prev_action is None
    finally:
        env.close()


def test_sequence_evaluation_matches_step_rollout():
    model = DoomActorCritic(num_actions=9)
    model.eval()
    B, T = 2, 6
    obs_seq = {
        "rays": {
            "distance": torch.rand(B, T, 64),
            "hit_type": torch.randint(0, 6, (B, T, 64)),
        },
        "visible_entities": {
            "features": torch.randn(B, T, 32, 5),
            "entity_type": torch.randint(0, 12, (B, T, 32)),
            "mask": torch.randint(0, 2, (B, T, 32)),
        },
        "vitals": torch.randn(B, T, 17),
    }
    hx = model.initial_state(B)
    mem = model.initial_memory(B)
    prev_act = torch.zeros(B, 9)
    init_hx = hx.clone()

    step_lps, step_vals, actions = [], [], []
    grids, lms, odoms = [], [], []
    with torch.no_grad():
        for t in range(T):
            step_obs = {
                "rays": {"distance": obs_seq["rays"]["distance"][:, t], "hit_type": obs_seq["rays"]["hit_type"][:, t]},
                "visible_entities": {
                    "features": obs_seq["visible_entities"]["features"][:, t],
                    "entity_type": obs_seq["visible_entities"]["entity_type"][:, t],
                    "mask": obs_seq["visible_entities"]["mask"][:, t],
                },
                "vitals": obs_seq["vitals"][:, t],
            }
            act, lp, _, val, hx, mem = model.step_with_memory(step_obs, hx, mem, prev_action=prev_act)
            step_lps.append(lp)
            step_vals.append(val)
            actions.append(act)
            grids.append(mem.grid)
            lms.append(mem.landmarks)
            odoms.append(mem.odom_pose)
            prev_act = act

        seq_lp, _, seq_val = model.evaluate_sequence(
            obs_seq,
            init_hx,
            torch.stack(actions, dim=1),
            mem_grids_seq=torch.stack(grids, dim=1),
            mem_landmarks_seq=torch.stack(lms, dim=1),
            mem_odom_seq=torch.stack(odoms, dim=1),
        )

    torch.testing.assert_close(seq_lp, torch.stack(step_lps, dim=1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(seq_val, torch.stack(step_vals, dim=1), atol=1e-5, rtol=1e-5)


def test_odometry_trajectory_and_landmarks_match_replay():
    """Task P.5 Done-When check: replay trajectories match internal odometry & landmark map."""
    env = make_env("maps/stage4/s4_seed000.wad", "MAP01", episode_ticks=800, render_buffers=False)
    try:
        model = DoomActorCritic(num_actions=9)
        model.eval()
        obs, _ = env.reset(seed=0)
        p0 = env.privileged_state()
        x0, y0, ang0 = p0["x"], p0["y"], math.radians(p0["angle"])
        c0, s0 = math.cos(ang0), math.sin(ang0)

        hx = model.initial_state(1)
        mem = model.initial_memory(1)
        prev_action = torch.zeros(1, 9)

        # Scripted exploration trajectory: walk forward, strafe, turn left/right, walk back
        fwd = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        turn_l = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
        turn_r = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.int8)
        script = [fwd] * 12 + [turn_l] * 6 + [fwd] * 10 + [turn_r] * 6 + [fwd] * 10

        pos_errors = []
        heading_cos_errors = []
        for act in script:
            obs_t = _obs_to_tensor(obs)
            with torch.no_grad():
                _, _, _, _, hx, mem = model.step_with_memory(
                    obs_t, hx, mem, prev_action=prev_action, deterministic=True
                )

            p = env.privileged_state()
            # Transform ground-truth world displacement into spawn frame
            dx_w, dy_w = p["x"] - x0, p["y"] - y0
            gt_rel_x = dx_w * c0 + dy_w * s0
            gt_rel_y = -dx_w * s0 + dy_w * c0
            gt_dtheta = math.radians(p["angle"]) - ang0

            est_x, est_y, est_c, est_s = mem.odom_pose[0].tolist()
            pos_err = math.hypot(est_x - gt_rel_x, est_y - gt_rel_y)
            # Cosine similarity between estimated and ground-truth heading unit vectors
            cos_sim = est_c * math.cos(gt_dtheta) + est_s * math.sin(gt_dtheta)
            pos_errors.append(pos_err)
            heading_cos_errors.append(cos_sim)

            prev_action = torch.from_numpy(act).float().unsqueeze(0)
            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break

        # Also check last step after final env.step()
        obs_t = _obs_to_tensor(obs)
        with torch.no_grad():
            _, _, _, _, hx, mem = model.step_with_memory(
                obs_t, hx, mem, prev_action=prev_action, deterministic=True
            )
        p = env.privileged_state()
        dx_w, dy_w = p["x"] - x0, p["y"] - y0
        gt_rel_x = dx_w * c0 + dy_w * s0
        gt_rel_y = -dx_w * s0 + dy_w * c0
        est_x, est_y, est_c, est_s = mem.odom_pose[0].tolist()
        pos_errors.append(math.hypot(est_x - gt_rel_x, est_y - gt_rel_y))

        mean_pos_err = float(np.mean(pos_errors))
        min_cos_sim = float(np.min(heading_cos_errors))
        total_travel = math.hypot(gt_rel_x, gt_rel_y)
        print(
            f"[odometry] traveled={total_travel:.1f} units | mean_pos_err={mean_pos_err:.1f} units | "
            f"min_heading_cos_sim={min_cos_sim:.4f} | valid_landmarks={int((mem.landmarks[0,:,2] > 0.5).sum())} | "
            f"explored_cells={int((mem.grid[0,0] > 0.5).sum())}"
        )

        assert min_cos_sim > 0.99, f"Heading drift too high: min cos_sim={min_cos_sim:.4f}"
        assert mean_pos_err < 32.0, f"Position drift exceeded half a grid cell: {mean_pos_err:.1f}"
        assert (mem.grid[0, 0] > 0.5).sum() > 10, "Ego memory grid did not carve explored space"
        assert (mem.landmarks[0, :, 2] > 0.5).sum() >= 1, "Stage 4 map should record at least 1 landmark"
    finally:
        env.close()
