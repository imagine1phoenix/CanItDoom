"""Ego-centric relative odometry map and persistent landmark memory (V1, Task P.5).

Uses ONLY observations passed to DoomAgent.act() and the agent's own previous
action. Position updates, landmark slot rotations, and map grid transforms
use only sine and cosine (never atan2 / arctan2), per RULES.md §2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MAP_GRID_SIZE = 32
MAP_HALF_EXTENT = 1024.0  # covers [-1024, +1024] map units around player (64 units/cell)
MAP_CHANNELS = 5          # 0: free/explored, 1: wall, 2: door, 3: locked_door, 4: item/key
NUM_LANDMARK_SLOTS = 12
LANDMARK_FEAT_DIM = 6     # [fwd/1024, side/1024, valid, is_door, is_locked, is_item]

FRAME_SKIP = 4.0
TURN_TAP_RAD = math.radians(7.03125)    # measured single-step turn (1024/65536 * 360 / 2)
TURN_HOLD_RAD = math.radians(14.0625)   # measured steady-state 4-tic turn (2560/65536 * 360)
DUPLICATE_RADIUS = 96.0


@dataclass
class MemoryState:
    """Fixed-size per-episode spatial memory state (cleared on reset / done)."""

    grid: torch.Tensor       # (B, 5, 32, 32) ego-centric local semantic grid
    landmarks: torch.Tensor  # (B, 12, 6) persistent ego-frame landmark slots
    odom_pose: torch.Tensor  # (B, 4) [rel_x, rel_y, cos_h, sin_h] in episode spawn frame
    prev_vel: torch.Tensor   # (B, 2) [vel_forward, vel_right] from previous step
    turn_hold: torch.Tensor  # (B,) long consecutive turn steps
    step_idx: torch.Tensor   # (B,) long decision step index within episode

    def clone(self) -> "MemoryState":
        return MemoryState(
            grid=self.grid.clone(),
            landmarks=self.landmarks.clone(),
            odom_pose=self.odom_pose.clone(),
            prev_vel=self.prev_vel.clone(),
            turn_hold=self.turn_hold.clone(),
            step_idx=self.step_idx.clone(),
        )


def mask_memory_state(state: MemoryState, keep_mask: torch.Tensor) -> MemoryState:
    """Zero out memory state for batch elements where episode ended (keep_mask == 0)."""
    m4 = keep_mask.view(-1, 1, 1, 1)
    m3 = keep_mask.view(-1, 1, 1)
    m2 = keep_mask.view(-1, 1)
    m1 = keep_mask.view(-1)
    odom = state.odom_pose * m2
    # Reset cos_h to 1.0, sin_h to 0.0 when episode resets
    reset_flag = 1.0 - m1
    odom = odom.clone()
    odom[:, 2] = odom[:, 2] + reset_flag
    return MemoryState(
        grid=state.grid * m4,
        landmarks=state.landmarks * m3,
        odom_pose=odom,
        prev_vel=state.prev_vel * m2,
        turn_hold=(state.turn_hold * m1.long()),
        step_idx=(state.step_idx * m1.long()),
    )


class EgoSpatialMemory(nn.Module):
    """Parameter-free ego-centric odometry grid + landmark slot updater."""

    def __init__(self, num_rays: int = 64, ray_range: float = 1024.0):
        super().__init__()
        self.num_rays = num_rays
        self.ray_range = ray_range

        angles = torch.arange(num_rays, dtype=torch.float32) * (2.0 * math.pi / num_rays)
        self.register_buffer("ray_cos", torch.cos(angles))
        self.register_buffer("ray_sin", torch.sin(angles))

        # Radial samples along each ray for free-space carving (16 steps)
        radii = torch.linspace(0.0625, 1.0, 16, dtype=torch.float32)
        self.register_buffer("radii", radii)

    def initial_state(self, batch_size: int = 1, device: torch.device | str = "cpu") -> MemoryState:
        grid = torch.zeros(batch_size, MAP_CHANNELS, MAP_GRID_SIZE, MAP_GRID_SIZE, device=device)
        landmarks = torch.zeros(batch_size, NUM_LANDMARK_SLOTS, LANDMARK_FEAT_DIM, device=device)
        odom_pose = torch.zeros(batch_size, 4, device=device)
        odom_pose[:, 2] = 1.0  # cos(0) = 1.0, sin(0) = 0.0
        prev_vel = torch.zeros(batch_size, 2, device=device)
        turn_hold = torch.zeros(batch_size, dtype=torch.long, device=device)
        step_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        return MemoryState(
            grid=grid,
            landmarks=landmarks,
            odom_pose=odom_pose,
            prev_vel=prev_vel,
            turn_hold=turn_hold,
            step_idx=step_idx,
        )

    @staticmethod
    def _to_grid_coords(fwd: torch.Tensor, side: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map ego (fwd, side_left) offsets in map units to integer grid (row, col, in_bounds)."""
        # Row 0 is +MAP_HALF_EXTENT ahead, Row 31 is -MAP_HALF_EXTENT behind (UP = ahead)
        # Col 0 is +MAP_HALF_EXTENT left, Col 31 is -MAP_HALF_EXTENT right
        u = (MAP_HALF_EXTENT - side) / (2.0 * MAP_HALF_EXTENT) * (MAP_GRID_SIZE - 1)
        v = (MAP_HALF_EXTENT - fwd) / (2.0 * MAP_HALF_EXTENT) * (MAP_GRID_SIZE - 1)
        col = torch.round(u).long()
        row = torch.round(v).long()
        valid = (col >= 0) & (col < MAP_GRID_SIZE) & (row >= 0) & (row < MAP_GRID_SIZE)
        col = torch.clamp(col, 0, MAP_GRID_SIZE - 1)
        row = torch.clamp(row, 0, MAP_GRID_SIZE - 1)
        return row, col, valid

    @torch.no_grad()
    def step_update(
        self,
        obs: dict[str, torch.Tensor],
        state: MemoryState,
        prev_action: torch.Tensor | None = None,
    ) -> MemoryState:
        """Advance relative odometry, rotate/shift ego grid & landmarks, and splat new observations."""
        rays_dist = obs["rays"]["distance"]
        rays_type = obs["rays"]["hit_type"]
        vitals = obs["vitals"]
        ent_feat = obs["visible_entities"]["features"]
        ent_type = obs["visible_entities"]["entity_type"]
        ent_mask = obs["visible_entities"]["mask"]

        B = vitals.shape[0]
        device = vitals.device

        new_step_idx = state.step_idx + 1
        active_step = state.step_idx > 0  # step 0 is initial observation before any action

        # 1. Estimate incremental turn dtheta from prev_action (TURN_LEFT=4, TURN_RIGHT=5)
        if prev_action is not None:
            turn_cmd = prev_action[:, 4].float() - prev_action[:, 5].float()
        else:
            turn_cmd = torch.zeros(B, device=device)

        is_turning = (turn_cmd != 0) & active_step
        new_turn_hold = torch.where(is_turning, state.turn_hold + 1, torch.zeros_like(state.turn_hold))
        # On the very first env.step() after reset (step_idx == 1), Doom spawn freeze gives ~0 turn
        can_turn = active_step & (state.step_idx >= 2)
        mag = torch.where(
            new_turn_hold <= 1,
            torch.full((B,), TURN_TAP_RAD, device=device),
            torch.full((B,), TURN_HOLD_RAD, device=device),
        )
        dtheta = torch.where(can_turn, turn_cmd * mag, torch.zeros(B, device=device))

        cos_dt = torch.cos(dtheta)
        sin_dt = torch.sin(dtheta)

        # 2. Estimate incremental ego translation from vitals[14:16] = (vel_forward, vel_right)
        curr_vf = vitals[:, 14]
        curr_vr = vitals[:, 15]
        dfwd = torch.where(active_step, curr_vf * FRAME_SKIP, torch.zeros(B, device=device))
        dright = torch.where(active_step, curr_vr * FRAME_SKIP, torch.zeros(B, device=device))
        dside = -dright  # left-positive

        # Suppress wall-pushing odometry drift using up to two orthogonal contact normals (< 18.5 units)
        dist_units = rays_dist * self.ray_range  # (B, 64)
        blocking = (rays_type == 1) | (rays_type == 2) | (rays_type == 3)
        masked_d = torch.where(blocking, dist_units, torch.full_like(dist_units, 1e6))
        min_d1, idx1 = torch.min(masked_d, dim=1)
        in_c1 = min_d1 < 18.5
        if in_c1.any():
            n1x = self.ray_cos[idx1]
            n1y = self.ray_sin[idx1]
            proj1 = dfwd * n1x + dside * n1y
            pen1 = torch.where(in_c1 & (proj1 > 0.0), proj1, torch.zeros_like(proj1))
            dfwd = dfwd - pen1 * n1x
            dside = dside - pen1 * n1y

            # Second wall normal (orthogonal to first) for 90-degree corners
            align = self.ray_cos.view(1, -1) * n1x.unsqueeze(1) + self.ray_sin.view(1, -1) * n1y.unsqueeze(1)
            masked_d2 = torch.where(blocking & (align <= 0.5), dist_units, torch.full_like(dist_units, 1e6))
            min_d2, idx2 = torch.min(masked_d2, dim=1)
            in_c2 = in_c1 & (min_d2 < 18.5)
            if in_c2.any():
                n2x = self.ray_cos[idx2]
                n2y = self.ray_sin[idx2]
                proj2 = dfwd * n2x + dside * n2y
                pen2 = torch.where(in_c2 & (proj2 > 0.0), proj2, torch.zeros_like(proj2))
                dfwd = dfwd - pen2 * n2x
                dside = dside - pen2 * n2y

            dright = -dside

        new_prev_vel = torch.stack([curr_vf, curr_vr], dim=-1)

        # 3. Update dead-reckoned relative pose [rel_x, rel_y, cos_h, sin_h] in episode spawn frame
        old_x = state.odom_pose[:, 0]
        old_y = state.odom_pose[:, 1]
        old_c = state.odom_pose[:, 2]
        old_s = state.odom_pose[:, 3]

        new_c = old_c * cos_dt - old_s * sin_dt
        new_s = old_s * cos_dt + old_c * sin_dt
        # Re-normalize unit circle to avoid drift over long episodes
        norm = torch.sqrt(new_c * new_c + new_s * new_s + 1e-12)
        new_c = new_c / norm
        new_s = new_s / norm

        # Inverse of ego_rotate: vx = fwd*cos(h) + right*sin(h), vy = fwd*sin(h) - right*cos(h)
        new_x = old_x + dfwd * new_c + dright * new_s
        new_y = old_y + dfwd * new_s - dright * new_c
        new_odom = torch.stack([new_x, new_y, new_c, new_s], dim=-1)

        # 4. Rotate and translate previous ego grid into current ego frame
        # Affine matrix maps (u_new, v_new) -> (u_old, v_old) where u=-side/L, v=-fwd/L
        theta_mat = torch.zeros(B, 2, 3, device=device)
        theta_mat[:, 0, 0] = cos_dt
        theta_mat[:, 0, 1] = sin_dt
        theta_mat[:, 0, 2] = -dside / MAP_HALF_EXTENT
        theta_mat[:, 1, 0] = -sin_dt
        theta_mat[:, 1, 1] = cos_dt
        theta_mat[:, 1, 2] = -dfwd / MAP_HALF_EXTENT

        warp_grid = F.affine_grid(theta_mat, state.grid.shape, align_corners=False)
        warped = F.grid_sample(
            state.grid,
            warp_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        # Slight decay keeps recent observations crisp and prevents saturation
        new_grid = warped * 0.98

        # 5. Splat current rays into ego-centric grid
        # 5a. Carve free space along each ray up to hit distance
        sample_d = self.radii.view(1, 1, -1) * dist_units.unsqueeze(-1)  # (B, 64, 16)
        free_valid = sample_d < (dist_units.unsqueeze(-1) - 32.0)
        sample_fwd = sample_d * self.ray_cos.view(1, self.num_rays, 1)
        sample_side = sample_d * self.ray_sin.view(1, self.num_rays, 1)
        f_row, f_col, f_in = self._to_grid_coords(sample_fwd, sample_side)
        f_mask = free_valid & f_in

        b_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(f_row)
        if f_mask.any():
            new_grid[b_idx[f_mask], 0, f_row[f_mask], f_col[f_mask]] = 1.0

        # Mark player's current cell (center of ego map) as visited/free
        center = MAP_GRID_SIZE // 2
        new_grid[:, 0, center, center] = 1.0

        # 5b. Splat hit points (wall=1->ch1, door=2->ch2, locked=3->ch3, item=5->ch4)
        hit_fwd = dist_units * self.ray_cos.view(1, self.num_rays)
        hit_side = dist_units * self.ray_sin.view(1, self.num_rays)
        h_row, h_col, h_in = self._to_grid_coords(hit_fwd, hit_side)
        h_range_ok = (rays_dist < 0.999) & h_in
        b_ray = torch.arange(B, device=device).view(B, 1).expand_as(h_row)

        for rtype, ch in ((1, 1), (2, 2), (3, 3), (5, 4)):
            rmask = (rays_type == rtype) & h_range_ok
            if rmask.any():
                new_grid[b_ray[rmask], ch, h_row[rmask], h_col[rmask]] = 1.0

        # 5c. Also splat visible item/key entities (types 5..9) into channel 4
        is_item_ent = (ent_mask > 0) & (ent_type >= 5) & (ent_type <= 9)
        if is_item_ent.any():
            edist = ent_feat[:, :, 0] * self.ray_range
            esin = ent_feat[:, :, 1]
            ecos = ent_feat[:, :, 2]
            e_row, e_col, e_in = self._to_grid_coords(edist * ecos, edist * esin)
            emask = is_item_ent & e_in
            if emask.any():
                b_ent = torch.arange(B, device=device).view(B, 1).expand_as(e_row)
                new_grid[b_ent[emask], 4, e_row[emask], e_col[emask]] = 1.0

        # 6. Transform & update persistent landmark slots (B, 12, 6)
        lm = state.landmarks.clone()
        lm_valid = lm[:, :, 2] > 0.5
        old_lm_fwd = lm[:, :, 0] * self.ray_range - dfwd.unsqueeze(1)
        old_lm_side = lm[:, :, 1] * self.ray_range - dside.unsqueeze(1)
        new_lm_fwd = old_lm_fwd * cos_dt.unsqueeze(1) + old_lm_side * sin_dt.unsqueeze(1)
        new_lm_side = -old_lm_fwd * sin_dt.unsqueeze(1) + old_lm_side * cos_dt.unsqueeze(1)
        lm[:, :, 0] = torch.where(lm_valid, new_lm_fwd / self.ray_range, torch.zeros_like(new_lm_fwd))
        lm[:, :, 1] = torch.where(lm_valid, new_lm_side / self.ray_range, torch.zeros_like(new_lm_side))

        # Ingest newly observed landmarks per batch item (nearest per type: locked=3, door=2, item=5)
        for b in range(B):
            candidates: list[tuple[float, float, int]] = []
            for rtype, cls_idx in ((3, 4), (2, 3), (5, 5)):
                rmask = (rays_type[b] == rtype) & (rays_dist[b] < 0.999)
                if rmask.any():
                    idx = torch.argmin(torch.where(rmask, rays_dist[b], torch.full_like(rays_dist[b], 2.0)))
                    d = float(dist_units[b, idx].item())
                    f_off = d * float(self.ray_cos[idx].item())
                    s_off = d * float(self.ray_sin[idx].item())
                    candidates.append((f_off, s_off, cls_idx))

            # Visible key/weapon/health entities (entity types 5..9 -> cls_idx=5)
            for k in range(ent_mask.shape[1]):
                if ent_mask[b, k] > 0 and 5 <= int(ent_type[b, k].item()) <= 9:
                    d = float(ent_feat[b, k, 0].item()) * self.ray_range
                    f_off = d * float(ent_feat[b, k, 2].item())
                    s_off = d * float(ent_feat[b, k, 1].item())
                    candidates.append((f_off, s_off, 5))

            for f_off, s_off, cls_idx in candidates:
                fn = f_off / self.ray_range
                sn = s_off / self.ray_range
                # Check for existing nearby slot of same landmark class
                same_cls = (lm[b, :, 2] > 0.5) & (lm[b, :, cls_idx] > 0.5)
                if same_cls.any():
                    df = lm[b, :, 0] * self.ray_range - f_off
                    ds = lm[b, :, 1] * self.ray_range - s_off
                    dist2 = df * df + ds * ds
                    dist2 = torch.where(same_cls, dist2, torch.full_like(dist2, 1e12))
                    best_slot = int(torch.argmin(dist2).item())
                    if float(dist2[best_slot].item()) <= DUPLICATE_RADIUS * DUPLICATE_RADIUS:
                        lm[b, best_slot, 0] = fn
                        lm[b, best_slot, 1] = sn
                        continue

                # Insert into first empty slot, or replace farthest existing slot
                empty = torch.nonzero(lm[b, :, 2] < 0.5, as_tuple=False)
                if len(empty) > 0:
                    slot = int(empty[0, 0].item())
                else:
                    r2 = lm[b, :, 0] ** 2 + lm[b, :, 1] ** 2
                    slot = int(torch.argmax(r2).item())

                lm[b, slot].zero_()
                lm[b, slot, 0] = fn
                lm[b, slot, 1] = sn
                lm[b, slot, 2] = 1.0
                lm[b, slot, cls_idx] = 1.0

        return MemoryState(
            grid=new_grid,
            landmarks=lm,
            odom_pose=new_odom,
            prev_vel=new_prev_vel,
            turn_hold=new_turn_hold,
            step_idx=new_step_idx,
        )

    @torch.no_grad()
    def build_sequence_inputs(
        self,
        obs_seq: dict[str, torch.Tensor],
        actions_seq: torch.Tensor,
        init_mem: MemoryState | None = None,
        done_mask_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unroll parameter-free odometry & landmark memory over a (B, T, ...) batch sequence."""
        B, T = obs_seq["vitals"].shape[:2]
        device = obs_seq["vitals"].device
        mem = self.initial_state(B, device=device) if init_mem is None else init_mem.clone()

        grids = []
        landmarks = []
        odoms = []

        for t in range(T):
            if t > 0 and done_mask_seq is not None:
                mem = mask_memory_state(mem, done_mask_seq[:, t - 1])
            step_obs = {
                "rays": {
                    "distance": obs_seq["rays"]["distance"][:, t],
                    "hit_type": obs_seq["rays"]["hit_type"][:, t],
                },
                "visible_entities": {
                    "features": obs_seq["visible_entities"]["features"][:, t],
                    "entity_type": obs_seq["visible_entities"]["entity_type"][:, t],
                    "mask": obs_seq["visible_entities"]["mask"][:, t],
                },
                "vitals": obs_seq["vitals"][:, t],
            }
            if t == 0:
                prev_act = None
            else:
                prev_act = actions_seq[:, t - 1]
                if done_mask_seq is not None:
                    prev_act = prev_act * done_mask_seq[:, t - 1].unsqueeze(-1)

            mem = self.step_update(step_obs, mem, prev_action=prev_act)
            grids.append(mem.grid)
            landmarks.append(mem.landmarks)
            odoms.append(mem.odom_pose)

        return torch.stack(grids, dim=1), torch.stack(landmarks, dim=1), torch.stack(odoms, dim=1)


class SpatialMemoryEncoder(nn.Module):
    """Learnable Conv2d + masked-slot encoder for the V1 ego map and landmark slots."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.map_conv = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, 16, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),            # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),            # 8 -> 4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 48),
            nn.ReLU(),
        )
        self.slot_mlp = nn.Sequential(
            nn.Linear(LANDMARK_FEAT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        # 48 (ego map) + 32 (landmark slots) + 4 (odom pose [x/1024, y/1024, cos, sin]) = 84
        self.proj = nn.Sequential(
            nn.Linear(48 + 32 + 4, out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        grid: torch.Tensor,
        landmarks: torch.Tensor,
        odom_pose: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape = odom_pose.shape[:-1]
        g = grid.view(-1, MAP_CHANNELS, MAP_GRID_SIZE, MAP_GRID_SIZE)
        lm = landmarks.view(-1, NUM_LANDMARK_SLOTS, LANDMARK_FEAT_DIM)
        odom = odom_pose.view(-1, 4).clone()
        odom[:, 0:2] = odom[:, 0:2] / MAP_HALF_EXTENT

        map_feat = self.map_conv(g)

        valid = lm[:, :, 2:3]
        slot_h = self.slot_mlp(lm) * valid
        counts = torch.clamp(valid.sum(dim=1), min=1.0)
        slot_pool = slot_h.sum(dim=1) / counts

        out = self.proj(torch.cat([map_feat, slot_pool, odom], dim=-1))
        return out.view(*orig_shape, -1)
