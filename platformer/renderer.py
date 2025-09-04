from __future__ import annotations

import math
from typing import Optional, List

import numpy as np
import os
import re

try:
    import pygame
except Exception:  # pragma: no cover
    pygame = None  # type: ignore

from .env import PlatformerEnv, GameConfig
from .policy import MLPPolicy, MLPPolicyConfig


class Renderer:
    def __init__(self, window_width: int = 800, window_height: int = 600) -> None:
        if pygame is None:
            raise RuntimeError("pygame not installed. Run `pip install pygame`.")
        self.clock = pygame.time.Clock()
        self.screen = None
        self.font = None
        self.font_big = None
        self._size = (window_width, window_height)
        self.scale_x = 1.0
        self.scale_y = 1.0

    def _world_to_screen(self, x: float, y: float, height: float) -> tuple[int, int]:
        sx = int(x * self.scale_x)
        sy = int((height - y) * self.scale_y)
        return sx, sy

    def _px_scale(self) -> float:
        return float(min(self.scale_x, self.scale_y))

    def _ensure_window(self, cfg: GameConfig) -> None:
        if not pygame.get_init():
            pygame.init()
        width_px, height_px = self._size
        # derive scales from desired window size
        self.scale_x = width_px / float(max(1e-6, cfg.width))
        self.scale_y = height_px / float(max(1e-6, cfg.height))
        if self.screen is None:
            self.screen = pygame.display.set_mode((width_px, height_px))
            pygame.display.set_caption("Platformer")
            self.font = pygame.font.SysFont(None, 20)
            self.font_big = pygame.font.SysFont(None, 28)

    def close(self) -> None:
        if pygame.get_init():
            pygame.quit()
        self.screen = None
        self.font = None
        self.font_big = None

    def _parse_ckpt_meta(self, label: str) -> tuple[Optional[int], Optional[float]]:
        base = os.path.basename(label)
        gen = None
        fit = None
        m1 = re.search(r"gen(\d+)", base)
        if m1:
            try:
                gen = int(m1.group(1))
            except Exception:
                gen = None
        m2 = re.search(r"fit([0-9]+(?:\.[0-9]+)?)", base)
        if m2:
            try:
                fit = float(m2.group(1))
            except Exception:
                fit = None
        return gen, fit

    def render_episode(self, env: PlatformerEnv, policy: Optional[MLPPolicy] = None, fps: int = 30, speed: float = 1.0, label: Optional[str] = None, progress: Optional[tuple[int, int]] = None) -> None:
        cfg = env.config
        self._ensure_window(cfg)
        screen = self.screen
        font = self.font
        font_big = self.font_big
        assert screen is not None and font is not None and font_big is not None

        obs = env.reset()
        done = False
        steps_per_frame = max(1, int(round(float(speed))))
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
                # allow skipping to next generation with space when rendering a policy
                if policy is not None and event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    done = True
                    break

            for _ in range(steps_per_frame):
                if done:
                    break
                if policy is None:
                    keys = pygame.key.get_pressed()
                    action = 0
                    if keys[pygame.K_LEFT]:
                        action = 1
                    if keys[pygame.K_RIGHT]:
                        action = 2
                    if keys[pygame.K_SPACE]:
                        action = 5 if action == 2 else (4 if action == 1 else 3)
                else:
                    action = policy.act(obs)
                obs, reward, done, info = env.step(action)

            # Draw
            screen.fill((30, 30, 40))
            # platforms
            for (px, py, pw, ph) in cfg.platforms:
                rect = pygame.Rect(
                    *self._world_to_screen(px, py + ph, cfg.height),
                    int(pw * self.scale_x),
                    int(ph * self.scale_y),
                )
                rect.top -= rect.height
                pygame.draw.rect(screen, (80, 120, 80), rect)

            # coins (uncollected only)
            if hasattr(env, "coins"):
                for coin in env.coins:
                    if coin.get("collected"):
                        continue
                    cx, cy = self._world_to_screen(coin["x"], coin["y"], cfg.height)
                    pygame.draw.circle(screen, (255, 215, 0), (cx, cy), int(cfg.coin_radius * self._px_scale()))

            # jump powerup
            if hasattr(env, "powerup") and env.powerup and not env.powerup.get("collected"):
                px, py = self._world_to_screen(env.powerup["x"], env.powerup["y"], cfg.height)
                pygame.draw.circle(screen, (80, 180, 255), (px, py), int(cfg.powerup_radius * self._px_scale()))

            # player as circle
            cx, cy = self._world_to_screen(env.x, env.y + 0.4, cfg.height)
            pygame.draw.circle(screen, (220, 200, 80), (cx, cy), int(0.4 * self._px_scale()))

            # goal line
            width_px, height_px = self._size
            gx1 = self._world_to_screen(cfg.x_goal, 0.0, cfg.height)[0]
            pygame.draw.line(screen, (200, 80, 80), (gx1, 0), (gx1, height_px), 2)

            # HUD: coins, time (seconds), powerup
            coins_text = f"Coins: {info.get('coins_collected', 0)}/{info.get('coins_total', 0)}"
            time_left_s = max(0.0, float(info.get('time_remaining', 0)) * float(cfg.dt))
            time_text = f"Time: {time_left_s:.1f}s"
            pwr_text = "Power: x2 jump" if info.get('has_jump_powerup') else "Power: none"
            hud_text = f"{coins_text}  {time_text}  {pwr_text}"
            text_surface = self.font.render(hud_text, True, (230, 230, 230))
            screen.blit(text_surface, (10, 10))

            # Progress top-right (index/total) during playback
            if progress is not None:
                i, n = progress
                prog_surface = self.font_big.render(f"{i}/{n}", True, (255, 255, 255))
                pr = prog_surface.get_rect()
                pr.top = 6
                pr.right = width_px - 10
                screen.blit(prog_surface, pr)

            # Label + parsed meta
            if label:
                base = os.path.basename(label)
                gen, fit = self._parse_ckpt_meta(base)
                if gen is not None or fit is not None:
                    meta = []
                    if gen is not None:
                        meta.append(f"Gen {gen}")
                    if fit is not None:
                        meta.append(f"Fit {fit:.3f}")
                    meta_text = "  ".join(meta)
                    meta_surface = self.font_big.render(meta_text, True, (255, 255, 0))
                    screen.blit(meta_surface, (10, 30))
                    label_surface = self.font.render(base, True, (200, 200, 255))
                    screen.blit(label_surface, (10, 30 + meta_surface.get_height() + 2))
                else:
                    label_surface = self.font.render(base, True, (200, 200, 255))
                    screen.blit(label_surface, (10, 30))

            pygame.display.flip()
            self.clock.tick(fps)


def render(weights_paths: Optional[List[str]] = None, speed: float = 1.0) -> None:
    env = PlatformerEnv(GameConfig())
    renderer = Renderer()

    if not weights_paths:
        renderer.render_episode(env, None, speed=speed)
        renderer.close()
        return

    # Sort by generation (then fitness) parsed from filename for chronological playback
    def sort_key(p: str) -> tuple:
        gen, fit = renderer._parse_ckpt_meta(os.path.basename(p))
        gen_key = gen if gen is not None else float('inf')
        fit_key = fit if fit is not None else float('inf')
        return (gen_key, fit_key)

    weights_paths = sorted(weights_paths, key=sort_key)

    cfg = MLPPolicyConfig(input_size=env.observation_size, output_size=env.action_size)
    total = len(weights_paths)
    for idx, path in enumerate(weights_paths, start=1):
        policy = MLPPolicy(cfg)
        flat = np.load(path)
        policy.set_flat(flat)
        renderer.render_episode(env, policy, speed=speed, label=path, progress=(idx, total))

    renderer.close()
