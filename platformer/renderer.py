from __future__ import annotations

import math
from typing import Optional, List, Dict, Tuple, Any

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
        # sprite caches
        self._sprite_base: Dict[str, Any] = {}
        self._sprite_scaled: Dict[Tuple[str, int, int], Any] = {}
        # character facing state
        self._face_left = False

    def _world_to_screen(self, x: float, y: float, height: float) -> Tuple[int, int]:
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
            # Load static bitmap assets
            assets_dir = os.path.join(os.path.dirname(__file__), 'static')
            self._load_bitmaps(assets_dir)

    

    def _load_bitmaps(self, assets_dir: str) -> None:
        # Strictly load required PNGs from static assets directory
        dino_path = os.path.join(assets_dir, 'dino.png')
        coin_path = os.path.join(assets_dir, 'coin.png')
        boots_path = os.path.join(assets_dir, 'boots.png')
        rac_path = os.path.join(assets_dir, 'raccoon.png')
        missing: List[str] = []
        if not os.path.exists(dino_path):
            missing.append(dino_path)
        if not os.path.exists(coin_path):
            missing.append(coin_path)
        if not os.path.exists(boots_path):
            missing.append(boots_path)
        if not os.path.exists(rac_path):
            missing.append(rac_path)
        if missing:
            raise FileNotFoundError(
                "Missing static assets: " + ", ".join(missing) +
                ". Place PNGs in the 'platformer/static/' directory."
            )
        self._sprite_base['dino'] = pygame.image.load(dino_path).convert_alpha()
        self._sprite_base['coin'] = pygame.image.load(coin_path).convert_alpha()
        self._sprite_base['boots'] = pygame.image.load(boots_path).convert_alpha()
        self._sprite_base['raccoon'] = pygame.image.load(rac_path).convert_alpha()

    def _get_sprite_scaled(self, name: str, target_w: int, target_h: int) -> Any:
        key = (name, max(1, target_w), max(1, target_h))
        if key in self._sprite_scaled:
            return self._sprite_scaled[key]
        base = self._sprite_base.get(name)
        if base is None:
            raise KeyError(f"sprite not found: {name}")
        scaled = pygame.transform.scale(base, (key[1], key[2]))
        self._sprite_scaled[key] = scaled
        return scaled

    def _get_sprite_scaled_by_world_h(self, name: str, world_h: float) -> Any:
        base = self._sprite_base.get(name)
        if base is None:
            raise KeyError(f"sprite not found: {name}")
        bw, bh = base.get_width(), base.get_height()
        if bh <= 0:
            bh = 1
        px_h = max(1, int(round(world_h * self.scale_y)))
        px_w = max(1, int(round(px_h * (bw / float(bh)))))
        return self._get_sprite_scaled(name, px_w, px_h)

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

    def render_episode(self, env: PlatformerEnv, policy: Optional[MLPPolicy] = None, fps: int = 30, speed: float = 1.0, label: Optional[str] = None, progress: Optional[tuple[int, int]] = None, show_hitboxes: bool = False) -> None:
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

            # Update facing based on horizontal velocity
            if env.vx < -0.05:
                self._face_left = False
            elif env.vx > 0.05:
                self._face_left = True

            # Draw
            screen.fill((30, 30, 40))
            # platforms
            for (px, py, pw, ph) in cfg.platforms:
                rect = pygame.Rect(
                    *self._world_to_screen(px, py + ph, cfg.height),
                    int(pw * self.scale_x),
                    int(ph * self.scale_y),
                )
                pygame.draw.rect(screen, (80, 120, 80), rect)
                if show_hitboxes:
                    pygame.draw.rect(screen, (60, 200, 220), rect, 1)

            # coins (uncollected only) - draw sprite
            if hasattr(env, "coins"):
                for coin in env.coins:
                    if coin.get("collected"):
                        continue
                    # desired world size is diameter = 2*coin_radius; use vertical scaling to preserve proportion to platforms
                    world_h = 2.0 * cfg.coin_radius
                    spr = self._get_sprite_scaled_by_world_h('coin', world_h)
                    # position centered at coin (x,y)
                    cx, cy = self._world_to_screen(coin["x"], coin["y"], cfg.height)
                    rect = spr.get_rect()
                    rect.center = (cx, cy)
                    screen.blit(spr, rect)
                    if show_hitboxes:
                        pygame.draw.circle(screen, (255, 255, 0), (cx, cy), int(4 * cfg.coin_radius * self._px_scale()), 1)

            # boots powerup sprite
            if hasattr(env, "powerup") and env.powerup and not env.powerup.get("collected"):
                world_h = 1.2 * cfg.powerup_radius
                spr = self._get_sprite_scaled_by_world_h('boots', world_h)
                cx, cy = self._world_to_screen(env.powerup["x"], env.powerup["y"], cfg.height)
                rect = spr.get_rect()
                rect.center = (cx, cy)
                screen.blit(spr, rect)
                if show_hitboxes:
                    pygame.draw.circle(screen, (0, 200, 255), (cx, cy), int(4 * cfg.powerup_radius * self._px_scale()), 1)

            # player as 8-bit dinosaur sprite (approx 0.8 world units tall)
            world_h = 1.0
            dino = self._get_sprite_scaled_by_world_h('dino', world_h)
            if self._face_left:
                dino = pygame.transform.flip(dino, True, False)
            # anchor bottom-center at (x, y)
            bx, by = self._world_to_screen(env.x, env.y, cfg.height)
            rect = dino.get_rect()
            rect.midbottom = (bx, by)
            screen.blit(dino, rect)

            # Overlay boots on dinosaur if powerup active
            if info.get('has_jump_powerup'):
                boots = self._get_sprite_scaled_by_world_h('boots', world_h * 0.33)
                if self._face_left:
                    boots = pygame.transform.flip(boots, True, False)
                brect = boots.get_rect()
                brect.midbottom = (bx, by - 1)
                screen.blit(boots, brect)
            if show_hitboxes:
                pradius = 0.4
                pcx, pcy = self._world_to_screen(env.x, env.y + pradius, cfg.height)
                pygame.draw.circle(screen, (255, 60, 60), (pcx, pcy), int(4 * pradius * self._px_scale()), 1)

            # raccoon goal
            rx, ry = getattr(env, 'raccoon', {"x": cfg.x_goal, "y": 0.6}).values()
            rac_world_h = 1.0
            rac = self._get_sprite_scaled_by_world_h('raccoon', rac_world_h)
            rcx, rcy = self._world_to_screen(rx, ry, cfg.height)
            rrect = rac.get_rect()
            # Center the raccoon sprite on (rx, ry) to match collision center in env
            rrect.center = (rcx, rcy)
            screen.blit(rac, rrect)
            if show_hitboxes:
                pygame.draw.circle(screen, (255, 255, 255), (rcx, rcy), int(4 * cfg.raccoon_radius * self._px_scale()), 1)

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
                pr.right = self._size[0] - 10
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


def render(weights_paths: Optional[List[str]] = None, speed: float = 1.0, show_hitboxes: bool = False) -> None:
    env = PlatformerEnv(GameConfig())
    renderer = Renderer()

    if not weights_paths:
        renderer.render_episode(env, None, speed=speed, show_hitboxes=show_hitboxes)
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
        renderer.render_episode(env, policy, speed=speed, label=path, progress=(idx, total), show_hitboxes=show_hitboxes)

    renderer.close()
