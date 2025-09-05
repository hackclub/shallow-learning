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
    def __init__(self, window_width: int = 800, window_height: int = 600, fullscreen: bool = False) -> None:
        if pygame is None:
            raise RuntimeError("pygame not installed. Run `pip install pygame`.")
        self.clock = pygame.time.Clock()
        self.screen = None
        self.font = None
        self.font_big = None
        self._size = (window_width, window_height)
        self._fullscreen = fullscreen
        self.scale_x = 1.0
        self.scale_y = 1.0
        self._offset_x_px = 0
        self._offset_y_px = 0
        # sprite caches
        self._sprite_base: Dict[str, Any] = {}
        self._sprite_scaled: Dict[Tuple[str, int, int], Any] = {}
        # character facing state (default facing right: base sprite faces left, so flip)
        self._face_left = True
        # camera state (world x of left edge)
        self._camera_x = 0.0
        # content ratio cache (visible alpha bounds / surface size)
        self._content_ratio: Dict[str, Tuple[float, float]] = {}

    def show_intro(self, env: PlatformerEnv, message: str = "COLLECT COINS FOR HEIDI", flashes: int = 4, on_ms: int = 450, off_ms: int = 250) -> None:
        cfg = env.config
        self._ensure_window(cfg)
        screen = self.screen
        font_big = self.font_big
        assert screen is not None and font_big is not None
        width_px, height_px = self._size
        text_surface = font_big.render(message, True, (255, 255, 255))
        rect = text_surface.get_rect()
        rect.center = (width_px // 2, height_px // 2)

        for i in range(max(1, flashes * 2)):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    return
            screen.fill((0, 0, 0))
            if i % 2 == 0:
                screen.blit(text_surface, rect)
                pygame.display.flip()
                pygame.time.delay(on_ms)
            else:
                pygame.display.flip()
                pygame.time.delay(off_ms)

    def _world_to_screen(self, x: float, y: float, height: float) -> Tuple[int, int]:
        sx = int(self._offset_x_px + (x - self._camera_x) * self.scale_x)
        sy = int(self._offset_y_px + (height - y) * self.scale_y)
        return sx, sy

    def _world_rect_to_screen(self, x: float, y: float, w: float, h: float, height: float) -> Any:
        """Pixel-perfect mapping from world rect to screen rect using edge rounding.
        x,y are world bottom-left; y axis points up. Returns a pygame.Rect.
        """
        s = self.scale_x
        left = int(round(self._offset_x_px + (x - self._camera_x) * s))
        right = int(round(self._offset_x_px + (x + w - self._camera_x) * s))
        bottom = int(round(self._offset_y_px + (height - y) * s))
        top = int(round(self._offset_y_px + (height - (y + h)) * s))
        return pygame.Rect(left, top, max(0, right - left), max(0, bottom - top))

    def _px_scale(self) -> float:
        return float(min(self.scale_x, self.scale_y))

    def _ensure_window(self, cfg: GameConfig) -> None:
        if not pygame.get_init():
            pygame.init()
        width_px, height_px = self._size
        # derive UNIFORM scale from width (zoomed view).
        s = width_px / float(max(1e-6, cfg.width))
        self.scale_x = s
        self.scale_y = s
        world_px_w = int(round(cfg.width * s))
        self._offset_x_px = (width_px - world_px_w) // 2
        # Align bottom row of ASCII map (if any) with y=0. Place floor exactly
        # 1 tile above the bottom of the window when a map is present.
        # Align bottom row to exact window bottom (no padding) when level_map is set
        if getattr(cfg, 'level_map', None):
            self._offset_y_px = int(round(height_px - (cfg.height) * s))
        else:
            # default 1m padding when no map is used
            self._offset_y_px = int(round(height_px - (cfg.height + 1.0) * s))
        if self.screen is None:
            flags = pygame.FULLSCREEN if self._fullscreen else 0
            self.screen = pygame.display.set_mode((width_px, height_px), flags)
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

    def _get_content_ratio(self, name: str) -> Tuple[float, float]:
        """Return (height_ratio, width_ratio) of visible (alpha>0) bounds to total surface size."""
        if name in self._content_ratio:
            return self._content_ratio[name]
        surf = self._sprite_base.get(name)
        if surf is None:
            raise KeyError(f"sprite not found: {name}")
        mask = pygame.mask.from_surface(surf)
        rects = mask.get_bounding_rects()
        if rects:
            # Union all rects
            min_l = min(r.left for r in rects)
            min_t = min(r.top for r in rects)
            max_r = max(r.right for r in rects)
            max_b = max(r.bottom for r in rects)
            width = max(1, max_r - min_l)
            height = max(1, max_b - min_t)
        else:
            width = surf.get_width()
            height = surf.get_height()
        h_ratio = height / float(max(1, surf.get_height()))
        w_ratio = width / float(max(1, surf.get_width()))
        # Clamp to sane range
        h_ratio = float(max(0.05, min(1.0, h_ratio)))
        w_ratio = float(max(0.05, min(1.0, w_ratio)))
        self._content_ratio[name] = (h_ratio, w_ratio)
        return self._content_ratio[name]

    def _get_sprite_scaled_to_rect(self, name: str, world_w: float, world_h: float) -> Any:
        """Scale sprite so its visible content bounds fit exactly in (world_w, world_h)."""
        px_w = max(1, int(round(world_w * self.scale_x)))
        px_h = max(1, int(round(world_h * self.scale_y)))
        # Correct for transparent padding so the visible content matches exactly
        h_ratio, w_ratio = self._get_content_ratio(name)
        target_w = max(1, int(round(px_w / max(1e-6, w_ratio))))
        target_h = max(1, int(round(px_h / max(1e-6, h_ratio))))
        return self._get_sprite_scaled(name, target_w, target_h)

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
        play_speed = float(speed)
        steps_per_frame = max(1, int(round(play_speed)))
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                    # Toggle fullscreen
                    self._fullscreen = not self._fullscreen
                    flags = pygame.FULLSCREEN if self._fullscreen else 0
                    width_px, height_px = self._size
                    self.screen = pygame.display.set_mode((width_px, height_px), flags)
                    # Recompute scaling offsets
                    self._ensure_window(cfg)
                # macOS: Command + F to toggle fullscreen (also works on others with GUI modifier)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                    mods = event.mod if hasattr(event, 'mod') else 0
                    KMOD_GUI = getattr(pygame, 'KMOD_GUI', 0)
                    KMOD_META = getattr(pygame, 'KMOD_META', 0)
                    if mods & (KMOD_GUI or KMOD_META):
                        self._fullscreen = not self._fullscreen
                        flags = pygame.FULLSCREEN if self._fullscreen else 0
                        width_px, height_px = self._size
                        self.screen = pygame.display.set_mode((width_px, height_px), flags)
                        self._ensure_window(cfg)
                # allow skipping to next generation with space when rendering a policy
                if policy is not None and event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    done = True
                    break
                # Speed controls: '=' to increase, '-' to decrease
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_EQUALS, getattr(pygame, 'K_PLUS', pygame.K_EQUALS), getattr(pygame, 'K_KP_PLUS', pygame.K_EQUALS)):
                        play_speed = min(64.0, play_speed * 1.5)
                        steps_per_frame = max(1, int(round(play_speed)))
                    elif event.key in (pygame.K_MINUS, getattr(pygame, 'K_KP_MINUS', pygame.K_MINUS)):
                        play_speed = max(0.25, play_speed / 1.5)
                        steps_per_frame = max(1, int(round(play_speed)))

            for _ in range(steps_per_frame):
                if done:
                    break
                if policy is None:
                    keys = pygame.key.get_pressed()
                    left = keys[pygame.K_LEFT]
                    right = keys[pygame.K_RIGHT]
                    jump = keys[pygame.K_SPACE]
                    run = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

                    base = 0
                    if left and not right:
                        base = 1
                    elif right and not left:
                        base = 2
                    if jump:
                        base = 5 if base == 2 else (4 if base == 1 else 3)
                    if run:
                        base += 6
                    action = base
                else:
                    action = policy.act(obs)
                obs, reward, done, info = env.step(action)

            # Update facing based on horizontal velocity
            if env.vx < -0.05:
                self._face_left = False
            elif env.vx > 0.05:
                self._face_left = True

            # Camera follow with 20% margins
            # Do not allow camera to scroll beyond the map. When an ASCII map
            # defines the level, the level length equals cfg.width exactly.
            if getattr(cfg, 'level_map', None):
                level_length = float(cfg.width)
            else:
                level_length = float(cfg.width * max(1, int(getattr(cfg, 'level_screens', 1))))
            viewport_w = float(cfg.width)
            left_margin = self._camera_x + 0.2 * viewport_w
            right_margin = self._camera_x + 0.8 * viewport_w
            # Adjust camera to keep player within margins
            if env.x < left_margin:
                self._camera_x = env.x - 0.2 * viewport_w
            elif env.x > right_margin:
                self._camera_x = env.x - 0.8 * viewport_w
            # Clamp camera within level bounds
            max_cam = max(0.0, level_length - viewport_w)
            self._camera_x = float(max(0.0, min(self._camera_x, max_cam)))
            # Snap camera to pixel grid to keep tile edges aligned
            s = self.scale_x
            if s > 0:
                self._camera_x = round(self._camera_x * s) / s

            # Draw
            screen.fill((30, 30, 40))
            # platforms (use pixel-perfect rect mapping)
            for (px, py, pw, ph) in cfg.platforms:
                rect = self._world_rect_to_screen(px, py, pw, ph, cfg.height)
                pygame.draw.rect(screen, (140, 140, 140), rect)
                if show_hitboxes:
                    pygame.draw.rect(screen, (60, 200, 220), rect, 1)

            # coins (uncollected only) - draw sprite
            if hasattr(env, "coins"):
                for coin in env.coins:
                    if not getattr(coin, 'active', True):
                        continue
                    cw = float(getattr(coin, 'w', getattr(cfg, 'coin_w', 1.0)))
                    ch = float(getattr(coin, 'h', getattr(cfg, 'coin_h', 1.0)))
                    spr = self._get_sprite_scaled_to_rect('coin', cw, ch)
                    cx, cy = self._world_to_screen(coin.x, coin.y, cfg.height)
                    rect = spr.get_rect()
                    rect.center = (cx, cy)
                    screen.blit(spr, rect)
                    if show_hitboxes:
                        hr = pygame.Rect(cx - int((cw * self.scale_x) * 0.5), cy - int((ch * self.scale_y) * 0.5), int(cw * self.scale_x), int(ch * self.scale_y))
                        pygame.draw.rect(screen, (255, 255, 0), hr, 1)

            # boots powerup sprite
            if hasattr(env, "powerup") and env.powerup and getattr(env.powerup, 'active', False):
                pw = float(getattr(env.powerup, 'w', getattr(cfg, 'powerup_w', 0.2)))
                ph = float(getattr(env.powerup, 'h', getattr(cfg, 'powerup_h', 0.2)))
                spr = self._get_sprite_scaled_to_rect('boots', pw, ph)
                cx, cy = self._world_to_screen(env.powerup.x, env.powerup.y, cfg.height)
                rect = spr.get_rect()
                rect.center = (cx, cy)
                screen.blit(spr, rect)
                if show_hitboxes:
                    hr = pygame.Rect(cx - int((pw * self.scale_x) * 0.5), cy - int((ph * self.scale_y) * 0.5), int(pw * self.scale_x), int(ph * self.scale_y))
                    pygame.draw.rect(screen, (0, 200, 255), hr, 1)

            # player as 8-bit dinosaur sprite scaled to collider rect
            dino = self._get_sprite_scaled_to_rect('dino', float(getattr(cfg, 'player_w', 0.8)), float(getattr(cfg, 'player_h', 0.8)))
            if self._face_left:
                dino = pygame.transform.flip(dino, True, False)
            # anchor bottom-center at (x, y)
            bx, by = self._world_to_screen(env.x, env.y, cfg.height)
            rect = dino.get_rect()
            rect.midbottom = (bx, by)
            screen.blit(dino, rect)

            # Overlay boots on dinosaur if powerup active (smaller and at feet)
            if info.get('has_jump_powerup'):
                boot_w = float(getattr(cfg, 'player_w', 0.8)) * 0.6
                boot_h = float(getattr(cfg, 'player_h', 0.8)) * 0.38
                boots = self._get_sprite_scaled_to_rect('boots', boot_w, boot_h)
                if self._face_left:
                    boots = pygame.transform.flip(boots, True, False)
                brect = boots.get_rect()
                # Place boots centered at player feet, with a tiny lift so they don't clip ground
                brect.midbottom = (bx, by - int(0.02 * self.scale_y))
                screen.blit(boots, brect)
            if show_hitboxes:
                pw = float(getattr(cfg, 'player_w', 0.8))
                ph = float(getattr(cfg, 'player_h', 0.8))
                pcx, pcy = self._world_to_screen(env.x, env.y + 0.5*ph, cfg.height)
                hr = pygame.Rect(pcx - int((pw * self.scale_x) * 0.5), pcy - int((ph * self.scale_y) * 0.5), int(pw * self.scale_x), int(ph * self.scale_y))
                pygame.draw.rect(screen, (255, 60, 60), hr, 1)

            # raccoon goal
            rac = getattr(env, 'raccoon', None)
            rx = rac.x if rac is not None else cfg.x_goal
            ry = rac.y if rac is not None else 0.6
            rw = float(getattr(rac, 'w', getattr(cfg, 'raccoon_w', 1.0))) if rac is not None else float(getattr(cfg, 'raccoon_w', 1.0))
            rh = float(getattr(rac, 'h', getattr(cfg, 'raccoon_h', 1.0))) if rac is not None else float(getattr(cfg, 'raccoon_h', 1.0))
            rac_img = self._get_sprite_scaled_to_rect('raccoon', rw, rh)
            rcx, rcy = self._world_to_screen(rx, ry, cfg.height)
            rrect = rac_img.get_rect()
            # Center the raccoon sprite on (rx, ry) to match collision center in env
            rrect.center = (rcx, rcy)
            screen.blit(rac_img, rrect)
            if show_hitboxes:
                hr = pygame.Rect(rcx - int((rw * self.scale_x) * 0.5), rcy - int((rh * self.scale_y) * 0.5), int(rw * self.scale_x), int(rh * self.scale_y))
                pygame.draw.rect(screen, (255, 255, 255), hr, 1)

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


def render(weights_paths: Optional[List[str]] = None, speed: float = 1.0, show_hitboxes: bool = False, show_intro: bool = False, fullscreen: bool = False) -> None:
    env = PlatformerEnv(GameConfig())
    renderer = Renderer(fullscreen=fullscreen)

    if not weights_paths:
        if show_intro:
            renderer.show_intro(env)
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
