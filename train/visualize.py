#!/usr/bin/env python3
"""Interactive Visualizer & Replay HUD for CanItDoom.

Recreates and extends the 3-panel real-time HUD seen on competitor setups:
  - Panel 1 (Left): First-person rendered DOOM camera with telemetry HUD overlay
  - Panel 2 (Center): 90-degree FOV rays + sector map wireframe
  - Panel 3 (Right): Privileged telemetry data + Ego-Centric Polar Radar (UP = ahead)

Supports 3 agent modes:
  - 'model': Uses the trained PPO DoomAgent policy (submission/agent.py)
  - 'teacher': Heuristic obstacle-avoidance & combat bot for level debugging
  - 'manual': Keyboard control (WASD / Arrows to steer, Space to shoot, E to use)

Controls:
  [P] Pause / Resume
  [SPACE] Step single frame (when paused)
  [N] Next map / episode
  [M] Cycle agent mode (model -> teacher -> manual)
  [ESC / Q] Quit
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pygame

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SUBMISSION_DIR = REPO_ROOT / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))
ENV_DIR = REPO_ROOT / "env"
if str(ENV_DIR) not in sys.path:
    sys.path.append(str(ENV_DIR))

from doom_env import make_env
import doom_env.spaces as S
from train.bc_pretrain import ExpertSpeedrunTeacher

# UI Layout Dimensions
CANVAS_W = 1260
CANVAS_H = 430
PANEL_W = 410
PANEL_H = 410
FPS = 15

# Color Palette (Dark Theme)
COLOR_BG = (18, 18, 22)
COLOR_PANEL_BG = (26, 26, 32)
COLOR_BORDER = (45, 45, 55)
COLOR_TEXT_MAIN = (235, 235, 240)
COLOR_TEXT_DIM = (140, 140, 155)
COLOR_TEXT_HIGHLIGHT = (255, 215, 0)
COLOR_GREEN = (60, 220, 100)
COLOR_RED = (240, 70, 70)
COLOR_CYAN = (60, 190, 240)
COLOR_MAGENTA = (220, 80, 230)
COLOR_GRAY = (120, 120, 135)
COLOR_WHITE = (255, 255, 255)

# Ray hit type color mapping
HIT_COLORS = {
    0: (60, 60, 70),       # None
    1: COLOR_GRAY,         # Wall
    2: COLOR_CYAN,         # Door
    3: COLOR_MAGENTA,      # Locked Door
    4: COLOR_RED,          # Enemy
    5: COLOR_GREEN,        # Item
}

BUTTON_NAMES = ("FWD", "BCK", "STL", "STR", "TRL", "TRR", "ATK", "USE", "WPN")


class TeacherBot:
    """Heuristic rule-based teacher bot used for map debugging and imitation."""

    def __init__(self):
        self.use_cooldown = 0
        self.stuck_counter = 0
        self.last_fwd_vel = 0.0

    def reset(self):
        self.use_cooldown = 0
        self.stuck_counter = 0
        self.last_fwd_vel = 0.0

    def act(self, obs: dict) -> np.ndarray:
        action = np.zeros(9, dtype=np.int8)
        rays_dist = obs["rays"]["distance"]
        rays_type = obs["rays"]["hit_type"]
        vitals = obs["vitals"]
        fwd_vel = vitals[14]

        # Check front rays (index 0 is ahead, 1-7 left, 57-63 right)
        front_dist = min(rays_dist[0], rays_dist[1], rays_dist[63])
        left_dist = np.mean(rays_dist[2:16])
        right_dist = np.mean(rays_dist[48:62])

        # Enemy in forward 90 deg FOV?
        has_enemy_front = False
        ents = obs["visible_entities"]
        if np.any(ents["mask"] == 1):
            for i in range(len(ents["mask"])):
                if ents["mask"][i] == 1 and ents["entity_type"][i] == 1:  # Enemy
                    feat = ents["features"][i]
                    if abs(float(feat[1])) < 0.35 and float(feat[2]) > 0.0 and float(feat[0]) < 0.6:
                        has_enemy_front = True
                        break

        # Check doors in front
        front_type = rays_type[0]
        if front_type in (2, 3) and front_dist < 0.15:
            if self.use_cooldown == 0:
                action[7] = 1  # USE
                self.use_cooldown = 12
        if self.use_cooldown > 0:
            self.use_cooldown -= 1

        # Combat action
        if has_enemy_front:
            action[6] = 1  # ATTACK

        # Navigation steering
        if front_dist < 0.18:
            # Wall ahead, turn towards more open side
            if left_dist > right_dist:
                action[4] = 1  # TURN_LEFT
            else:
                action[5] = 1  # TURN_RIGHT
        else:
            action[0] = 1  # MOVE_FORWARD
            # Gentle steering to center in corridors
            if left_dist < 0.15:
                action[5] = 1  # Steer right
            elif right_dist < 0.15:
                action[4] = 1  # Steer left

        self.last_fwd_vel = fwd_vel
        return action


class VisualizerHUD:
    """Renders the 3-panel interactive inspection screen."""

    def __init__(self, headless: bool = False):
        self.headless = headless
        pygame.init()
        pygame.font.init()
        self.font_small = pygame.font.SysFont("Menlo, Consolas, Courier New, monospace", 11)
        self.font_bold = pygame.font.SysFont("Menlo, Consolas, Courier New, monospace", 12, bold=True)
        self.font_title = pygame.font.SysFont("Menlo, Consolas, Courier New, monospace", 13, bold=True)

        if not headless and (os.environ.get("DISPLAY") or sys.platform in ("darwin", "win32")):
            try:
                self.screen = pygame.display.set_mode((CANVAS_W, CANVAS_H))
                pygame.display.set_caption("CanItDoom Neural Cluster HUD (P pause, N next level, Esc quit)")
            except Exception:
                self.headless = True
                self.screen = pygame.Surface((CANVAS_W, CANVAS_H))
        else:
            self.headless = True
            self.screen = pygame.Surface((CANVAS_W, CANVAS_H))

    def render_frame(
        self,
        obs: dict,
        priv: dict,
        step: int,
        episode_return: float,
        action: np.ndarray,
        mode: str,
        map_label: str,
        last_result: str,
        sector_polys: list,
    ) -> pygame.Surface:
        canvas = self.screen
        canvas.fill(COLOR_BG)

        # Panel Rects
        r1 = pygame.Rect(10, 10, PANEL_W, PANEL_H)
        r2 = pygame.Rect(20 + PANEL_W, 10, PANEL_W, PANEL_H)
        r3 = pygame.Rect(30 + PANEL_W * 2, 10, PANEL_W, PANEL_H)

        for r in (r1, r2, r3):
            pygame.draw.rect(canvas, COLOR_PANEL_BG, r, border_radius=6)
            pygame.draw.rect(canvas, COLOR_BORDER, r, width=1, border_radius=6)

        # -------------------------------------------------------------
        # PANEL 1 (LEFT): DOOM Camera View + Game Telemetry
        # -------------------------------------------------------------
        rgb = obs.get("rgb")
        if rgb is not None:
            # ViZDoom RGB is (120, 160, 3) -> transpose to (160, 120, 3) for pygame
            surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
            scaled = pygame.transform.scale(surf, (PANEL_W - 12, 280))
            canvas.blit(scaled, (r1.x + 6, r1.y + 6))
            pygame.draw.rect(canvas, COLOR_BORDER, (r1.x + 6, r1.y + 6, PANEL_W - 12, 280), width=1)

        # Telemetry overlay at bottom of panel 1
        hp = priv.get("health", 0.0)
        ammo = obs["vitals"][3]
        armor = obs["vitals"][1]
        speed = obs["vitals"][14]
        n_ents = int(np.sum(obs["visible_entities"]["mask"]))
        keys_held = priv.get("keys", np.zeros(6))
        key_labels = ["Bc", "Yc", "Rc", "Bs", "Ys", "Rs"]
        active_keys = [lbl for lbl, flag in zip(key_labels, keys_held) if flag > 0]
        keys_str = ",".join(active_keys) if active_keys else "-"

        hud_y = r1.y + 292
        line1 = f"step {step:<4d} hp {hp:<3.0f} arm {armor:<3.0f} ammo {ammo:<3.0f} spd {speed:<+4.1f} ret {episode_return:<+5.2f}"
        t1 = self.font_small.render(line1, True, COLOR_TEXT_MAIN)
        canvas.blit(t1, (r1.x + 8, hud_y))

        # Pressed buttons
        pressed_btns = [name for name, val in zip(BUTTON_NAMES, action) if int(val) == 1]
        btn_str = " ".join(pressed_btns) if pressed_btns else "NONE"
        line2 = f"{mode.upper()} | {map_label} | act: {btn_str}"
        t2 = self.font_bold.render(line2[:54], True, COLOR_GREEN if mode == "model" else COLOR_TEXT_HIGHLIGHT)
        canvas.blit(t2, (r1.x + 8, hud_y + 18))

        line3 = f"keys {keys_str} | ents {n_ents} | last: {last_result}"
        t3 = self.font_small.render(line3[:56], True, COLOR_TEXT_DIM)
        canvas.blit(t3, (r1.x + 8, hud_y + 36))

        line4 = "[P]ause  [N]ext  [M]ode(model/teacher/manual)  [Esc]uit"
        t4 = self.font_small.render(line4, True, COLOR_TEXT_DIM)
        canvas.blit(t4, (r1.x + 8, hud_y + 58))

        # -------------------------------------------------------------
        # PANEL 2 (CENTER): 3D POV from 90 deg FOV rays + map wireframe
        # -------------------------------------------------------------
        t_p2 = self.font_title.render("90 deg FOV from rays + map wireframe", True, COLOR_TEXT_MAIN)
        canvas.blit(t_p2, (r2.x + 12, r2.y + 10))

        pov_rect = pygame.Rect(r2.x + 8, r2.y + 32, PANEL_W - 16, PANEL_H - 42)
        pygame.draw.rect(canvas, (14, 14, 18), pov_rect, border_radius=4)
        pygame.draw.rect(canvas, COLOR_BORDER, pov_rect, width=1, border_radius=4)

        px, py, pz = priv.get("x", 0.0), priv.get("y", 0.0), priv.get("z", 0.0)
        p_ang = priv.get("angle", 0.0)  # degrees
        rad = math.radians(p_ang)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        eye_z = pz + 41.0  # Doom player eye height

        fov_w = pov_rect.width
        fov_h = pov_rect.height
        fov_cx = pov_rect.centerx
        fov_cy = pov_rect.centery
        # 90 deg horizontal FOV -> focal length = fov_w / 2
        focal_x = fov_w * 0.5
        focal_y = fov_w * 0.5

        # Clip surface to pov_rect so 3D lines don't spill outside Panel 2
        old_clip = canvas.get_clip()
        canvas.set_clip(pov_rect)

        # 1. Draw 3D Map Wireframe from sector_polygons()
        sec_floors = priv.get("sector_floor")
        sec_ceils = priv.get("sector_ceiling")

        def world_to_cam(wx: float, wy: float) -> tuple[float, float]:
            dx = wx - px
            dy = wy - py
            # Forward = depth (cam_z), Left = -cam_x (right is positive screen X)
            cam_z = dx * cos_a + dy * sin_a
            cam_x = dx * sin_a - dy * cos_a
            return cam_x, cam_z

        def project_pt(cam_x: float, cam_z: float, wz: float) -> tuple[int, int]:
            sx = int(fov_cx + (cam_x / cam_z) * focal_x)
            sy = int(fov_cy - ((wz - eye_z) / cam_z) * focal_y)
            return sx, sy

        for sec_idx, (sec_id, lines) in enumerate(sector_polys):
            fl_h = float(sec_floors[sec_idx]) if (sec_floors is not None and sec_idx < len(sec_floors)) else 0.0
            cl_h = float(sec_ceils[sec_idx]) if (sec_ceils is not None and sec_idx < len(sec_ceils)) else 128.0
            for ln in lines:
                x1, y1, x2, y2 = float(ln[0]), float(ln[1]), float(ln[2]), float(ln[3])
                cx1, cz1 = world_to_cam(x1, y1)
                cx2, cz2 = world_to_cam(x2, y2)

                # Near-plane clip at z = 8.0
                if cz1 < 8.0 and cz2 < 8.0:
                    continue
                if cz1 < 8.0:
                    t = (8.0 - cz1) / (cz2 - cz1)
                    cx1 = cx1 + t * (cx2 - cx1)
                    cz1 = 8.0
                elif cz2 < 8.0:
                    t = (8.0 - cz2) / (cz1 - cz2)
                    cx2 = cx2 + t * (cx1 - cx2)
                    cz2 = 8.0

                p1_bot = project_pt(cx1, cz1, fl_h)
                p2_bot = project_pt(cx2, cz2, fl_h)
                p1_top = project_pt(cx1, cz1, cl_h)
                p2_top = project_pt(cx2, cz2, cl_h)

                wire_col = (95, 100, 115)
                pygame.draw.line(canvas, wire_col, p1_bot, p2_bot, 1)
                pygame.draw.line(canvas, wire_col, p1_top, p2_top, 1)
                pygame.draw.line(canvas, (70, 75, 90), p1_bot, p1_top, 1)
                pygame.draw.line(canvas, (70, 75, 90), p2_bot, p2_top, 1)

        # 2. Draw 3D Raycast Columns across 90-degree FOV (-45 deg to +45 deg)
        # Ray 0 is straight ahead, counter-clockwise (1..8 = left, 56..63 = right)
        rays_dist = obs["rays"]["distance"] * 1024.0
        rays_type = obs["rays"]["hit_type"]
        num_rays = len(rays_dist)

        fov_ray_indices = list(range(8, -1, -1)) + list(range(63, 55, -1))  # Left (+45) to Right (-45)
        ray_fill_colors = {
            0: (40, 42, 52),
            1: (175, 170, 190),   # Wall: lavender-gray
            2: (50, 175, 245),    # Door: cyan-blue
            3: (210, 70, 230),    # Locked door: magenta
            4: (245, 55, 55),     # Enemy: bright red
            5: (65, 225, 95),     # Item: bright green
        }
        col_w = max(4, int(fov_w / len(fov_ray_indices)) + 1)

        for idx in fov_ray_indices:
            # Signed angle in radians: positive = left, negative = right
            ang_deg = (idx if idx <= 32 else idx - num_rays) * (360.0 / num_rays)
            rel_rad = math.radians(ang_deg)
            # Project onto screen X (left is -X, right is +X)
            screen_x = fov_cx - math.tan(rel_rad) * focal_x
            dist = max(16.0, float(rays_dist[idx]) * math.cos(rel_rad))
            ht = int(rays_type[idx])
            if ht == 0 and rays_dist[idx] >= 1023.0:
                continue

            bar_h = min(fov_h - 12, int((56.0 / dist) * focal_y))
            bx = int(screen_x - col_w / 2)
            by = int(fov_cy - bar_h / 2)
            col = ray_fill_colors.get(ht, (150, 150, 160))
            pygame.draw.rect(canvas, col, (bx, by, col_w, bar_h))
            if ht in (2, 3, 4, 5):
                pygame.draw.rect(canvas, COLOR_WHITE, (bx, by, col_w, bar_h), width=1)

        # 3. Project visible_entities markers + center crosshair in FOV
        ents = obs["visible_entities"]
        for i in range(len(ents["mask"])):
            if ents["mask"][i] == 1:
                feat = ents["features"][i]
                edist = max(16.0, float(feat[0]) * 1024.0)
                sin_e, cos_e = float(feat[1]), float(feat[2])
                if cos_e > 0.1:
                    ex = int(fov_cx - (sin_e / cos_e) * focal_x)
                    ey = int(fov_cy + (12.0 / (edist * cos_e)) * focal_y)
                    etype = int(ents["entity_type"][i])
                    ecol = COLOR_RED if etype == 1 else COLOR_GREEN
                    pygame.draw.circle(canvas, ecol, (ex, ey), 6, width=2)

        # Center crosshair (+)
        pygame.draw.line(canvas, COLOR_WHITE, (fov_cx - 6, fov_cy), (fov_cx + 6, fov_cy), 1)
        pygame.draw.line(canvas, COLOR_WHITE, (fov_cx, fov_cy - 6), (fov_cx, fov_cy + 6), 1)
        canvas.set_clip(old_clip)

        # -------------------------------------------------------------
        # PANEL 3 (RIGHT): DATA (privileged) + Polar Radar (UP = ahead)
        # -------------------------------------------------------------
        t_p3 = self.font_title.render("DATA (privileged is training-only)", True, COLOR_TEXT_HIGHLIGHT)
        canvas.blit(t_p3, (r3.x + 12, r3.y + 10))

        # Text Stats
        tic = priv.get("tic", 0)
        max_tic = priv.get("episode_ticks", 2100)
        vel_fwd = obs["vitals"][14]
        vel_rt = obs["vitals"][15]
        kills = priv.get("kills", 0)
        items = priv.get("items", 0)
        dmg = priv.get("damage_taken", 0.0)
        wpn_id = int(obs["vitals"][2])
        wpn_names = {1: "fist", 2: "pistol", 3: "shotgun", 4: "chaingun", 5: "rocket", 6: "plasma", 7: "bfg"}
        wpn_str = wpn_names.get(wpn_id, f"wpn{wpn_id}")

        # Count ray hit categories
        w_cnt = int(np.sum(rays_type == 1))
        e_cnt = int(np.sum(rays_type == 4))
        d_cnt = int(np.sum(rays_type == 2))
        l_cnt = int(np.sum(rays_type == 3))
        i_cnt = int(np.sum(rays_type == 5))

        stats = [
            f"pos {px:5.0f},{py:<5.0f} ang {p_ang:<4.0f}",
            f"tic {tic:<4d}/{max_tic:<4d} step {step:<4d} ret {episode_return:<+5.2f}",
            f"hp {hp:<3.0f} arm {armor:<3.0f} {wpn_str} ammo {ammo:<3.0f}",
            f"vel {vel_fwd:<+4.1f} {vel_rt:<+4.1f} keys {keys_str}",
            f"kills {kills:<2d} items {items:<2d} dmg {dmg:<3.0f}",
            f"rays w={w_cnt:<2d}, e={e_cnt:<2d}, d={d_cnt + l_cnt:<2d}, i={i_cnt:<2d}",
            f"visible_entities {n_ents}/32 (90 deg + LOS)",
        ]
        stat_y = r3.y + 32
        for s in stats:
            ts = self.font_small.render(s, True, COLOR_TEXT_MAIN if ("rays" in s or "visible" in s) else COLOR_TEXT_DIM)
            canvas.blit(ts, (r3.x + 12, stat_y))
            stat_y += 16

        # Polar Radar Section
        radar_h1 = self.font_small.render("UP = ahead", True, COLOR_TEXT_HIGHLIGHT)
        radar_h2 = self.font_small.render(f"visible_entities {n_ents} (90 deg FOV)", True, COLOR_TEXT_MAIN)
        canvas.blit(radar_h1, (r3.x + 12, stat_y + 10))
        canvas.blit(radar_h2, (r3.x + 12, stat_y + 26))

        # Legend on the left side below radar title (matching screenshot)
        legend_y = stat_y + 52
        items_legend = [
            ("none", (90, 90, 105)),
            ("wall", COLOR_GRAY),
            ("door", COLOR_CYAN),
            ("locked door", COLOR_MAGENTA),
            ("enemy", COLOR_RED),
            ("item", COLOR_GREEN),
        ]
        for name, col in items_legend:
            pygame.draw.circle(canvas, col, (r3.x + 20, legend_y + 5), 4)
            tl = self.font_small.render(name, True, COLOR_TEXT_DIM)
            canvas.blit(tl, (r3.x + 30, legend_y - 2))
            legend_y += 18

        # Radar Circle on the right side of Panel 3 bottom
        radar_cx = r3.x + PANEL_W - 115
        radar_cy = stat_y + 125
        max_radar_r = 85

        # Draw concentric range circles
        for r_step in (28, 56, 85):
            pygame.draw.circle(canvas, (38, 40, 52), (radar_cx, radar_cy), r_step, 1)

        # Draw 90 deg FOV cone lines (-45 deg to +45 deg from UP)
        fov_rad1 = math.radians(90 - 45)
        fov_rad2 = math.radians(90 + 45)
        x_fov1 = int(radar_cx + max_radar_r * math.cos(fov_rad1))
        y_fov1 = int(radar_cy - max_radar_r * math.sin(fov_rad1))
        x_fov2 = int(radar_cx + max_radar_r * math.cos(fov_rad2))
        y_fov2 = int(radar_cy - max_radar_r * math.sin(fov_rad2))
        pygame.draw.line(canvas, (65, 70, 88), (radar_cx, radar_cy), (x_fov1, y_fov1), 1)
        pygame.draw.line(canvas, (65, 70, 88), (radar_cx, radar_cy), (x_fov2, y_fov2), 1)

        # Plot 64 Rays on Polar Radar (index 0 is UP, index 16 is LEFT)
        for i in range(num_rays):
            theta = math.radians(90) + i * (2.0 * math.pi / num_rays)
            dist_norm = obs["rays"]["distance"][i]
            rr = dist_norm * max_radar_r
            rx = int(radar_cx + rr * math.cos(theta))
            ry = int(radar_cy - rr * math.sin(theta))
            ht = int(rays_type[i])
            c = HIT_COLORS.get(ht, COLOR_GRAY)
            if ht > 0:
                pygame.draw.line(canvas, (c[0] // 3, c[1] // 3, c[2] // 3), (radar_cx, radar_cy), (rx, ry), 1)
                pygame.draw.circle(canvas, c, (rx, ry), 2)

        # Plot visible entities + velocity vectors inside 90 deg FOV
        for i in range(len(ents["mask"])):
            if ents["mask"][i] == 1:
                feat = ents["features"][i]  # [dist/1024, sin(rel), cos(rel), vel_fwd, vel_rt]
                edist = min(1.0, float(feat[0])) * max_radar_r
                sin_ang, cos_ang = float(feat[1]), float(feat[2])
                ex = int(radar_cx - edist * sin_ang)
                ey = int(radar_cy - edist * cos_ang)
                etype = int(ents["entity_type"][i])
                ecolor = COLOR_RED if etype == 1 else COLOR_GREEN
                pygame.draw.circle(canvas, ecolor, (ex, ey), 4)
                # Velocity vector
                vx = int(ex + float(feat[4]) * 1.5)
                vy = int(ey - float(feat[3]) * 1.5)
                pygame.draw.line(canvas, ecolor, (ex, ey), (vx, vy), 1)

        # Player center dot on radar
        pygame.draw.circle(canvas, COLOR_WHITE, (radar_cx, radar_cy), 3)

        if not self.headless:
            pygame.display.flip()

        return canvas


def main():
    parser = argparse.ArgumentParser(description="CanItDoom Interactive Visualizer HUD")
    parser.add_argument("--map", default="maps/stage1/s1_seed000.wad:MAP01", help="WAD path and map lump (wad:MAP)")
    parser.add_argument("--checkpoint", default="submission/weights/model.pt", help="Path to trained model checkpoint")
    parser.add_argument("--mode", choices=["model", "teacher", "manual"], default="model", help="Agent mode")
    parser.add_argument("--headless", action="store_true", help="Run without opening desktop window")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--record", default=None, help="Output MP4 file path for recording")
    args = parser.parse_args()

    # Resolve map
    if ":" in args.map:
        wad_str, map_name = args.map.split(":", 1)
    else:
        wad_str, map_name = args.map, "MAP01"
    wad_path = Path(wad_str).resolve()
    if not wad_path.is_file():
        # Fallback to two_rooms if stage map not found
        wad_path = REPO_ROOT / "maps" / "two_rooms.wad"
        map_name = "MAP01"

    print(f"Loading environment with map: {wad_path.name}:{map_name}")
    env = make_env(str(wad_path), map_name=map_name, render_buffers=True)

    # Initialize agent
    mode = args.mode
    model_agent = None
    try:
        from submission.agent import DoomAgent
        model_agent = DoomAgent(env.observation_space, env.action_space, device="cpu")
        print(f"Loaded DoomAgent policy (weights: {args.checkpoint})")
    except Exception as exc:
        print(f"Note: DoomAgent load fallback ({exc}); switching to teacher bot mode.")
        mode = "teacher"

    teacher_bot = ExpertSpeedrunTeacher(env)
    hud = VisualizerHUD(headless=args.headless)

    # Optional video writer
    writer = None
    if args.record:
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.record, fourcc, 15.0, (CANVAS_W, CANVAS_H))
            print(f"Recording video to: {args.record}")
        except ImportError:
            print("Notice: OpenCV not installed; skipping MP4 output.")

    obs, _ = env.reset()
    if model_agent:
        model_agent.reset()
    teacher_bot.reset()

    paused = False
    step = 0
    ep_return = 0.0
    ep_num = 1
    last_result = "N/A"
    clock = pygame.time.Clock()

    print("\nVisualizer active. Controls:")
    print("  [P] Pause / Resume")
    print("  [SPACE] Step 1 frame (when paused)")
    print("  [N] Next map / episode")
    print("  [M] Cycle agent mode (model / teacher / manual)")
    print("  [ESC / Q] Quit\n")

    running = True
    manual_action = np.zeros(9, dtype=np.int8)

    while running and step < args.max_steps:
        # Handle Pygame Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                    print("PAUSED" if paused else "RESUMED")
                elif event.key == pygame.K_SPACE and paused:
                    # Single step advance
                    pass
                elif event.key == pygame.K_n:
                    # Reset to new episode
                    last_result = f"ep {ep_num}: SKIP, ret {ep_return:+.2f}, {step} steps"
                    obs, _ = env.reset()
                    if model_agent:
                        model_agent.reset()
                    teacher_bot.reset()
                    step = 0
                    ep_return = 0.0
                    ep_num += 1
                elif event.key == pygame.K_m:
                    modes = ["model", "teacher", "manual"]
                    idx = (modes.index(mode) + 1) % len(modes)
                    mode = modes[idx]
                    print(f"Switched agent mode to: {mode.upper()}")

        if paused:
            clock.tick(10)
            continue

        # Choose action based on mode
        if mode == "model" and model_agent is not None:
            action = np.asarray(model_agent.act(obs), dtype=np.int8)
        elif mode == "teacher":
            action = teacher_bot.act(obs)
        else:
            # Manual keyboard controls
            keys = pygame.key.get_pressed()
            manual_action = np.zeros(9, dtype=np.int8)
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                manual_action[0] = 1
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                manual_action[1] = 1
            if keys[pygame.K_a]:
                manual_action[2] = 1
            if keys[pygame.K_d]:
                manual_action[3] = 1
            if keys[pygame.K_LEFT]:
                manual_action[4] = 1
            if keys[pygame.K_RIGHT]:
                manual_action[5] = 1
            if keys[pygame.K_SPACE]:
                manual_action[6] = 1  # ATTACK
            if keys[pygame.K_e]:
                manual_action[7] = 1  # USE
            action = manual_action

        # Render HUD frame
        priv = env.privileged_state()
        polys = env.sector_polygons()
        frame_surface = hud.render_frame(
            obs=obs,
            priv=priv,
            step=step,
            episode_return=ep_return,
            action=action,
            mode=mode,
            map_label=f"{wad_path.stem}:{map_name}",
            last_result=last_result,
            sector_polys=polys,
        )

        # Write frame to video if recording
        if writer is not None:
            frame_arr = pygame.surfarray.array3d(frame_surface)
            frame_bgr = np.transpose(frame_arr, (1, 0, 2))[:, :, ::-1]
            writer.write(frame_bgr)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        ep_return += reward

        if terminated or truncated:
            reason = "EXIT" if info.get("exit_reached") else ("DEAD" if priv.get("dead") else "TIMEOUT")
            last_result = f"ep {ep_num}: {reason}, kills {priv.get('kills', 0)}, {step} steps"
            print(f"Episode {ep_num} finished: {reason} | return: {ep_return:+.2f} | steps: {step}")
            obs, _ = env.reset()
            if model_agent:
                model_agent.reset()
            teacher_bot.reset()
            step = 0
            ep_return = 0.0
            ep_num += 1

        clock.tick(FPS)

    if writer is not None:
        writer.release()
        print(f"Video saved to {args.record}")

    env.close()
    pygame.quit()
    print("Visualizer closed cleanly.")


if __name__ == "__main__":
    main()
