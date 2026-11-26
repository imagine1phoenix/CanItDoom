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
                    rel_angle = math.atan2(feat[1], feat[2]) if hasattr(math, "atan2") else 0.0
                    if abs(rel_angle) < 0.35 and feat[0] < 0.6:
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

        if not headless and os.environ.get("DISPLAY") or sys.platform == "darwin":
            try:
                self.screen = pygame.display.set_mode((CANVAS_W, CANVAS_H))
                pygame.display.set_caption("teacher bot on generated levels (P pause, N next level, Esc quit)")
            except Exception:
                self.headless = True
                self.screen = pygame.Surface((CANVAS_W, CANVAS_H))
        else:
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
        speed = math.sqrt(obs["vitals"][14] ** 2 + obs["vitals"][15] ** 2)
        n_ents = int(np.sum(obs["visible_entities"]["mask"]))

        hud_y = r1.y + 292
        line1 = f"step {step:<4d} health {hp:<3.0f} armor {armor:<3.0f} ammo {ammo:<3.0f} ents {n_ents:<2d} speed {speed:<+4.1f} ret {episode_return:<+5.2f}"
        t1 = self.font_small.render(line1, True, COLOR_TEXT_MAIN)
        canvas.blit(t1, (r1.x + 8, hud_y))

        # Pressed buttons
        pressed_btns = [name for name, val in zip(BUTTON_NAMES, action) if int(val) == 1]
        btn_str = " ".join(pressed_btns) if pressed_btns else "NONE"
        line2 = f"mode: {mode.upper():<7s} | map: {map_label:<12s} | act: {btn_str}"
        t2 = self.font_bold.render(line2, True, COLOR_GREEN if mode == "model" else COLOR_TEXT_HIGHLIGHT)
        canvas.blit(t2, (r1.x + 8, hud_y + 18))

        line3 = f"{mode} bot | generated level | last: {last_result}"
        t3 = self.font_small.render(line3, True, COLOR_TEXT_DIM)
        canvas.blit(t3, (r1.x + 8, hud_y + 36))

        # -------------------------------------------------------------
        # PANEL 2 (CENTER): 90 deg FOV from rays + map wireframe
        # -------------------------------------------------------------
        t_p2 = self.font_title.render("90 deg FOV from rays + map wireframe", True, COLOR_TEXT_MAIN)
        canvas.blit(t_p2, (r2.x + 12, r2.y + 10))

        # Center area for map wireframe
        map_rect = pygame.Rect(r2.x + 10, r2.y + 32, PANEL_W - 20, PANEL_H - 44)
        pygame.draw.rect(canvas, (14, 14, 18), map_rect, border_radius=4)
        pygame.draw.rect(canvas, COLOR_BORDER, map_rect, width=1, border_radius=4)

        px, py = priv.get("x", 0.0), priv.get("y", 0.0)
        p_ang = priv.get("angle", 0.0)  # degrees
        rad = math.radians(p_ang)

        # Auto-scale map around player
        cx = map_rect.centerx
        cy = map_rect.centery
        scale_map = 0.22  # map units to screen pixels

        # Draw sector polygons
        for _, lines in sector_polys:
            for ln in lines:
                x1, y1, x2, y2 = ln
                sx1 = int(cx + (x1 - px) * scale_map)
                sy1 = int(cy - (y1 - py) * scale_map)
                sx2 = int(cx + (x2 - px) * scale_map)
                sy2 = int(cy - (y2 - py) * scale_map)
                if map_rect.collidepoint(sx1, sy1) or map_rect.collidepoint(sx2, sy2):
                    pygame.draw.line(canvas, (75, 85, 110), (sx1, sy1), (sx2, sy2), 1)

        # Draw Ray Traces in Wireframe View
        rays_dist = obs["rays"]["distance"] * 1024.0
        rays_type = obs["rays"]["hit_type"]
        num_rays = len(rays_dist)
        for i in range(num_rays):
            # Ray 0 is ahead, counter-clockwise
            ray_angle = rad + i * (2.0 * math.pi / num_rays)
            rd = rays_dist[i] * scale_map
            rx = int(cx + rd * math.cos(ray_angle))
            ry = int(cy - rd * math.sin(ray_angle))
            ht = int(rays_type[i])
            c = HIT_COLORS.get(ht, COLOR_GRAY)

            # Highlight 90 deg forward cone
            is_forward_cone = (i <= 4) or (i >= 60)
            if is_forward_cone:
                pygame.draw.line(canvas, (c[0] // 2, c[1] // 2, c[2] // 2), (cx, cy), (rx, ry), 1)
            if ht > 0:
                pygame.draw.circle(canvas, c, (rx, ry), 2)

        # Draw Player in Wireframe View
        pygame.draw.circle(canvas, COLOR_GREEN, (cx, cy), 5)
        dir_x = int(cx + 18 * math.cos(rad))
        dir_y = int(cy - 18 * math.sin(rad))
        pygame.draw.line(canvas, COLOR_WHITE, (cx, cy), (dir_x, dir_y), 2)

        # -------------------------------------------------------------
        # PANEL 3 (RIGHT): DATA (privileged) + Polar Radar (UP = ahead)
        # -------------------------------------------------------------
        t_p3 = self.font_title.render("DATA (privileged is training-only)", True, COLOR_TEXT_MAIN)
        canvas.blit(t_p3, (r3.x + 12, r3.y + 10))

        # Text Stats
        tic = priv.get("tic", 0)
        max_tic = priv.get("episode_ticks", 2100)
        vel_fwd = obs["vitals"][14]
        vel_rt = obs["vitals"][15]
        kills = priv.get("kills", 0)
        items = priv.get("items", 0)
        dmg = priv.get("damage_taken", 0.0)

        # Count ray hit categories
        w_cnt = int(np.sum(rays_type == 1))
        e_cnt = int(np.sum(rays_type == 4))
        d_cnt = int(np.sum(rays_type == 2))
        i_cnt = int(np.sum(rays_type == 5))

        stats = [
            f"pos {px:6.0f},{py:<6.0f} ang {p_ang:<4.0f}",
            f"tic {tic:<4d}/{max_tic:<4d}  step {step:<4d}  rew {episode_return:<+5.2f}",
            f"hp {hp:<3.0f} arm {armor:<3.0f} pistol ammo {ammo:<3.0f}",
            f"vel {vel_fwd:<+4.1f} {vel_rt:<+4.1f} keys -",
            f"kills {kills:<2d} items {items:<2d} dmg {dmg:<3.0f}",
            f"rays: w={w_cnt:<2d}, e={e_cnt:<2d}, d={d_cnt:<2d}, i={i_cnt:<2d}",
            f"visible_entities {n_ents}/32 (90 deg + LOS)",
        ]
        stat_y = r3.y + 30
        for s in stats:
            ts = self.font_small.render(s, True, COLOR_TEXT_DIM if "rays" not in s else COLOR_TEXT_MAIN)
            canvas.blit(ts, (r3.x + 12, stat_y))
            stat_y += 15

        # Polar Radar Section
        radar_title = self.font_small.render(f"UP = ahead | visible_entities {n_ents} (90 deg FOV)", True, COLOR_TEXT_HIGHLIGHT)
        canvas.blit(radar_title, (r3.x + 12, stat_y + 8))

        radar_cx = r3.x + PANEL_W // 2
        radar_cy = stat_y + 115
        max_radar_r = 75

        # Draw concentric range circles (256, 512, 768, 1024)
        for r_step in (25, 50, 75):
            pygame.draw.circle(canvas, (38, 40, 52), (radar_cx, radar_cy), r_step, 1)

        # Draw 90 deg FOV cone lines (-45 deg to +45 deg from UP)
        fov_rad1 = math.radians(90 - 45)
        fov_rad2 = math.radians(90 + 45)
        x_fov1 = int(radar_cx + max_radar_r * math.cos(fov_rad1))
        y_fov1 = int(radar_cy - max_radar_r * math.sin(fov_rad1))
        x_fov2 = int(radar_cx + max_radar_r * math.cos(fov_rad2))
        y_fov2 = int(radar_cy - max_radar_r * math.sin(fov_rad2))
        pygame.draw.line(canvas, (55, 60, 75), (radar_cx, radar_cy), (x_fov1, y_fov1), 1)
        pygame.draw.line(canvas, (55, 60, 75), (radar_cx, radar_cy), (x_fov2, y_fov2), 1)

        # Plot 64 Rays on Polar Radar (index 0 is UP, index 16 is LEFT)
        for i in range(num_rays):
            theta = math.radians(90) + i * (2.0 * math.pi / num_rays)
            dist_norm = obs["rays"]["distance"][i]
            rr = dist_norm * max_radar_r
            rx = int(radar_cx + rr * math.cos(theta))
            ry = int(radar_cy - rr * math.sin(theta))
            ht = int(rays_type[i])
            if ht > 0:
                c = HIT_COLORS.get(ht, COLOR_GRAY)
                pygame.draw.circle(canvas, c, (rx, ry), 2)

        # Plot visible entities in 90 deg FOV
        ents = obs["visible_entities"]
        for i in range(len(ents["mask"])):
            if ents["mask"][i] == 1:
                feat = ents["features"][i]  # [dist/1024, sin(rel), cos(rel), vel_fwd, vel_rt]
                edist = feat[0] * max_radar_r
                # rel_angle: 0 is ahead (UP), positive is left
                sin_ang = feat[1]
                cos_ang = feat[2]
                ex = int(radar_cx - edist * sin_ang)
                ey = int(radar_cy - edist * cos_ang)
                etype = int(ents["entity_type"][i])
                ecolor = COLOR_RED if etype == 1 else COLOR_GREEN
                pygame.draw.circle(canvas, ecolor, (ex, ey), 4)

        # Player center dot on radar
        pygame.draw.circle(canvas, COLOR_WHITE, (radar_cx, radar_cy), 3)

        # Legend at bottom
        legend_y = r3.y + PANEL_H - 24
        items_legend = [
            ("none", (80, 80, 90)),
            ("wall", COLOR_GRAY),
            ("door", COLOR_CYAN),
            ("locked", COLOR_MAGENTA),
            ("enemy", COLOR_RED),
            ("item", COLOR_GREEN),
        ]
        leg_x = r3.x + 12
        for name, col in items_legend:
            pygame.draw.circle(canvas, col, (leg_x, legend_y + 4), 3)
            tl = self.font_small.render(name, True, COLOR_TEXT_DIM)
            canvas.blit(tl, (leg_x + 6, legend_y - 2))
            leg_x += 62

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

    teacher_bot = TeacherBot()
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
