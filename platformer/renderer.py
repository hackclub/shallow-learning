from __future__ import annotations

import math
from typing import Optional, List, Dict, Tuple, Any
import sys

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
        # navigation edge-trigger (accept LEFT/RIGHT only on key-up -> key-down transition)
        self._nav_ready = True

    def _debounce_navigation(self, max_ms: int = 200) -> None:
        """Briefly wait for LEFT/RIGHT keys to be released to avoid rapid re-triggers.
        Limits to max_ms to avoid long stalls.
        """
        if pygame is None:
            return
        try:
            print("[nav] debounce: waiting for key release")
        except Exception:
            pass
        start = pygame.time.get_ticks()
        while True:
            # Pump events so KEYUP is registered
            for _ in pygame.event.get():
                pass
            pressed = pygame.key.get_pressed()
            left_down = pressed[pygame.K_LEFT] if len(pressed) > pygame.K_LEFT else False
            right_down = pressed[pygame.K_RIGHT] if len(pressed) > pygame.K_RIGHT else False
            if not left_down and not right_down:
                break
            if pygame.time.get_ticks() - start >= max_ms:
                break
            self.clock.tick(120)

    def wait_for_space(self) -> None:
        """Hold the final frame until SPACE (or window close)."""
        if pygame is None:
            return
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    # Cmd/Ctrl-Q, Cmd/Ctrl-W, or 'q' should terminate immediately
                    mods = event.mod if hasattr(event, 'mod') else 0
                    KMOD_GUI = getattr(pygame, 'KMOD_GUI', 0)
                    KMOD_META = getattr(pygame, 'KMOD_META', 0)
                    if event.key in (pygame.K_q, pygame.K_w) and (mods & (KMOD_GUI | KMOD_META)):
                        pygame.quit()
                        sys.exit(0)
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit(0)
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    running = False
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                    self._fullscreen = not self._fullscreen
                    flags = pygame.FULLSCREEN if self._fullscreen else 0
                    width_px, height_px = self._size
                    self.screen = pygame.display.set_mode((width_px, height_px), flags)
            pygame.display.flip()
            self.clock.tick(30)

    def hold_or_advance(self, seconds: float = 3.0) -> bool:
        """Hold the final frame for up to 'seconds'.
        Returns True if the user pressed SPACE/ENTER or closed the window (advance),
        False if time elapsed (loop same playback again).
        """
        if pygame is None:
            return True
        remaining = float(max(0.0, seconds))
        while remaining > 0.0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    mods = event.mod if hasattr(event, 'mod') else 0
                    KMOD_GUI = getattr(pygame, 'KMOD_GUI', 0)
                    KMOD_META = getattr(pygame, 'KMOD_META', 0)
                    if event.key in (pygame.K_q, pygame.K_w) and (mods & (KMOD_GUI | KMOD_META)):
                        pygame.quit()
                        sys.exit(0)
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit(0)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                    self._fullscreen = not self._fullscreen
                    flags = pygame.FULLSCREEN if self._fullscreen else 0
                    width_px, height_px = self._size
                    self.screen = pygame.display.set_mode((width_px, height_px), flags)
            pygame.display.flip()
            self.clock.tick(30)
            remaining -= (1.0 / 30.0)
        return False

    def hold_or_navigate(self, seconds: float = 3.0) -> Optional[str]:
        """Hold final frame and listen for Left/Right navigation.
        Returns 'next' for Right arrow, 'prev' for Left arrow, or None if time elapsed.
        """
        if pygame is None:
            return None
        # Reset edge-trigger so the very next keydown is accepted during this hold.
        # Without this, if a previous frame left _nav_ready False, arrows would be ignored.
        self._nav_ready = True
        remaining = float(max(0.0, seconds))
        while remaining > 0.0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    # Immediate quit shortcuts
                    mods = event.mod if hasattr(event, 'mod') else 0
                    KMOD_GUI = getattr(pygame, 'KMOD_GUI', 0)
                    KMOD_META = getattr(pygame, 'KMOD_META', 0)
                    if event.key in (pygame.K_q, pygame.K_w) and (mods & (KMOD_GUI | KMOD_META)):
                        pygame.quit()
                        sys.exit(0)
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit(0)
                if event.type == pygame.KEYUP and event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    self._nav_ready = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT and self._nav_ready:
                    self._nav_ready = False
                    return 'prev'
                if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT and self._nav_ready:
                    self._nav_ready = False
                    return 'next'
                if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                    self._fullscreen = not self._fullscreen
                    flags = pygame.FULLSCREEN if self._fullscreen else 0
                    width_px, height_px = self._size
                    self.screen = pygame.display.set_mode((width_px, height_px), flags)
            pygame.display.flip()
            self.clock.tick(30)
            remaining -= (1.0 / 30.0)
        return None

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
            # If optimized versions exist, prefer them
            try:
                from .assets import optimize_all
                # Do not auto-optimize here (could be slow), but we can rely on pre-generated files
            except Exception:
                pass
            self._load_bitmaps(assets_dir)

    

    def _load_bitmaps(self, assets_dir: str) -> None:
        # Strictly load required PNGs from static assets directory
        dino_path = os.path.join(assets_dir, 'dino.png')
        coin_path = os.path.join(assets_dir, 'coin.png')
        boots_path = os.path.join(assets_dir, 'boots.png')
        rac_path = os.path.join(assets_dir, 'raccoon.png')
        heart_path = os.path.join(assets_dir, 'heart.png')
        brick_path = os.path.join(assets_dir, 'brick.png')
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
        # Prefer optimized variants if present
        def pick(path: str) -> str:
            base, ext = os.path.splitext(path)
            opt = f"{base}_optimized{ext}"
            return opt if os.path.exists(opt) else path
        self._sprite_base['dino'] = pygame.image.load(pick(dino_path)).convert_alpha()
        self._sprite_base['coin'] = pygame.image.load(pick(coin_path)).convert_alpha()
        self._sprite_base['boots'] = pygame.image.load(pick(boots_path)).convert_alpha()
        self._sprite_base['raccoon'] = pygame.image.load(pick(rac_path)).convert_alpha()
        # Optional heart sprite for celebration. If missing, we will draw a vector heart.
        try:
            heart_file = pick(heart_path)
            if os.path.exists(heart_file):
                self._sprite_base['heart'] = pygame.image.load(heart_file).convert_alpha()
        except Exception:
            pass
        # Optional brick texture for platforms
        try:
            brick_file = pick(brick_path)
            if os.path.exists(brick_file):
                self._sprite_base['brick'] = pygame.image.load(brick_file).convert_alpha()
        except Exception:
            # If brick asset is missing or fails to load, silently fall back to solid rectangles
            pass

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

    def _parse_ckpt_meta(self, label: str) -> tuple[Optional[int], Optional[float], Optional[int]]:
        base = os.path.basename(label)
        gen = None
        fit = None
        seed = None
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
        m3 = re.search(r"seed(\d+)", base)
        if m3:
            try:
                seed = int(m3.group(1))
            except Exception:
                seed = None
        return gen, fit, seed

    def render_episode(self, env: PlatformerEnv, policy: Optional[MLPPolicy] = None, fps: int = 30, speed: float = 1.0, label: Optional[str] = None, progress: Optional[tuple[int, int]] = None, show_hitboxes: bool = False, log_rays: bool = False, contrail: bool = False, contrail_alpha: float = 0.01, contrail_mod: int = 1, reload_event: Optional[object] = None) -> Optional[str]:
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
        celebrate_time_left_s = 0.0
        post_end_time_left_s = 0.0  # ensure a final HUD refresh on failure/time-up
        episode_return = 0.0
        first_frame = True
        paused = False  # SPACE toggles pause/resume (only when policy is not None)

        # Special snapshot mode for contrail: draw static world once, then step as fast as possible
        # without flipping until the very end. Camera is held fixed to avoid smearing.
        if contrail:
            try:
                print(f"[contrail] enabled alpha={contrail_alpha} mod={contrail_mod}")
            except Exception:
                pass
            # Freeze camera for the whole snapshot so trail aligns with platforms
            original_camera_x = self._camera_x
            self._camera_x = 0.0
            # Draw static background/platforms and goal once
            screen.fill((30, 30, 40))
            for (px, py, pw, ph) in cfg.platforms:
                rect = self._world_rect_to_screen(px, py, pw, ph, cfg.height)
                brick_base = self._sprite_base.get('brick')
                if brick_base is not None:
                    tile_px = max(8, int(round(0.5 * self.scale_x)))
                    try:
                        tile = self._get_sprite_scaled('brick', tile_px, tile_px)
                    except Exception:
                        tile = brick_base
                    tw, th = tile.get_width(), tile.get_height()
                    if tw <= 0 or th <= 0:
                        pygame.draw.rect(screen, (140, 140, 140), rect)
                    else:
                        old_clip = screen.get_clip()
                        screen.set_clip(rect)
                        start_x = rect.left - (rect.left % tw)
                        start_y = rect.top - (rect.top % th)
                        y = start_y
                        while y < rect.bottom:
                            x = start_x
                            while x < rect.right:
                                screen.blit(tile, (x, y))
                                x += tw
                            y += th
                        screen.set_clip(old_clip)
                else:
                    pygame.draw.rect(screen, (140, 140, 140), rect)
                if show_hitboxes:
                    pygame.draw.rect(screen, (60, 200, 220), rect, 1)

            # Draw raccoon goal once
            rac = getattr(env, 'raccoon', None)
            rx = rac.x if rac is not None else cfg.x_goal
            ry = rac.y if rac is not None else 0.6
            rw = float(getattr(rac, 'w', getattr(cfg, 'raccoon_w', 1.0))) if rac is not None else float(getattr(cfg, 'raccoon_w', 1.0))
            rh = float(getattr(rac, 'h', getattr(cfg, 'raccoon_h', 1.0))) if rac is not None else float(getattr(cfg, 'raccoon_h', 1.0))
            rac_img = self._get_sprite_scaled_to_rect('raccoon', rw, rh)
            rcx, rcy = self._world_to_screen(rx, ry, cfg.height)
            rrect = rac_img.get_rect()
            rrect.center = (rcx, rcy)
            screen.blit(rac_img, rrect)

            # Tight loop: step env as fast as possible, drawing only the ghost each step
            episode_return = 0.0
            frame_i = 0
            while not done:
                # If an external reload was requested (watch mode), return immediately
                try:
                    if reload_event is not None and callable(getattr(reload_event, 'is_set', None)) and reload_event.is_set():
                        print("[watch] reload requested during snapshot -> exiting episode early")
                        return None
                except Exception:
                    pass
                # Minimal event pump to allow window close or user skip
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit(0)
                    if event.type == pygame.KEYDOWN:
                        # Immediate quit shortcuts
                        mods = event.mod if hasattr(event, 'mod') else 0
                        KMOD_GUI = getattr(pygame, 'KMOD_GUI', 0)
                        KMOD_META = getattr(pygame, 'KMOD_META', 0)
                        if event.key in (pygame.K_q, pygame.K_w) and (mods & (KMOD_GUI | KMOD_META)):
                            pygame.quit()
                            sys.exit(0)
                        if event.key == pygame.K_q:
                            pygame.quit()
                            sys.exit(0)
                    if event.type == pygame.KEYUP and event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        self._nav_ready = True
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT and self._nav_ready:
                        self._nav_ready = False
                        print("[nav] RIGHT detected during snapshot -> prev (edge)")
                        return 'prev'
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT and self._nav_ready:
                        self._nav_ready = False
                        print("[nav] LEFT detected during snapshot -> next (edge)")
                        return 'next'
                # Compute action
                if policy is None:
                    # In snapshot mode without a policy, do nothing (idle)
                    action = 0
                else:
                    action = policy.act(obs)
                obs, reward, done, info = env.step(action)
                try:
                    episode_return += float(reward)
                except Exception:
                    pass

                # Update facing for sprite flip based on horizontal velocity
                if env.vx < -0.05:
                    self._face_left = False
                elif env.vx > 0.05:
                    self._face_left = True

                # Draw ghost at current position every contrail_mod frames (no clears, no flips)
                if max(1, int(contrail_mod)) <= 1 or (frame_i % max(1, int(contrail_mod)) == 0):
                    dino = self._get_sprite_scaled_to_rect('dino', float(getattr(cfg, 'player_w', 0.8)), float(getattr(cfg, 'player_h', 0.8)))
                    if self._face_left:
                        dino = pygame.transform.flip(dino, True, False)
                    bx, by = self._world_to_screen(env.x, env.y, cfg.height)
                    rect = dino.get_rect()
                    rect.midbottom = (bx, by)
                    ghost = dino.copy()
                    ghost.set_alpha(max(1, int(255 * max(0.0, min(1.0, contrail_alpha)))))
                    screen.blit(ghost, rect)
                    # Optional boots overlay ghost when active
                    if info.get('has_jump_powerup'):
                        boot_w = float(getattr(cfg, 'player_w', 0.8)) * 0.6
                        boot_h = float(getattr(cfg, 'player_h', 0.8)) * 0.38
                        boots = self._get_sprite_scaled_to_rect('boots', boot_w, boot_h)
                        if self._face_left:
                            boots = pygame.transform.flip(boots, True, False)
                        brect = boots.get_rect()
                        brect.midbottom = (bx, by - int(0.02 * self.scale_y))
                        ghost_b = boots.copy()
                        ghost_b.set_alpha(max(1, int(255 * max(0.0, min(1.0, contrail_alpha)))))
                        screen.blit(ghost_b, brect)
                frame_i += 1

                # Yield a tiny bit to avoid pegging the CPU
                self.clock.tick(240)

            # Draw final HUD/metadata once before presenting
            try:
                # HUD (coins/time/fitness)
                coins_text = f"Coins: {info.get('coins_collected', 0)}/{info.get('coins_total', 0)}"
                time_left_s = max(0.0, float(info.get('time_remaining', 0)) * float(cfg.dt))
                time_text = f"Time: {time_left_s:.1f}s"
                pwr_text = "Power: x2 jump" if info.get('has_jump_powerup') else "Power: none"
                fit_text = f"Fitness: {episode_return:.3f}"
                hud_text = f"{coins_text}  {time_text}  {fit_text}  {pwr_text}"
                text_surface = self.font.render(hud_text, True, (230, 230, 230))
                screen.blit(text_surface, (10, 10))
                # Progress top-right (if provided)
                if progress is not None:
                    i, n = progress
                    gen_val = None
                    try:
                        import json
                        with open(label + ".json", "r") as f:  # type: ignore[arg-type]
                            meta = json.load(f)
                            gen_val = meta.get("gen")
                    except Exception:
                        base = os.path.basename(label) if label else ""  # type: ignore[arg-type]
                        m = re.search(r"gen(\d+)", base)
                        gen_val = int(m.group(1)) if m else None
                    rank_desc = max(1, int(n) - int(i) + 1)
                    txt = f"Gen {gen_val} ({rank_desc}/{n})" if gen_val is not None else f"{rank_desc}/{n}"
                    prog_surface = self.font_big.render(txt, True, (255, 255, 255))
                    pr = prog_surface.get_rect()
                    pr.top = 6
                    pr.right = self._size[0] - 10
                    screen.blit(prog_surface, pr)
                # Label + parsed meta (Gen/Seed/EpisodeSeed) — disabled (already shown elsewhere)
                if False and label:
                    base = os.path.basename(label)
                    gen, _fit, seed = self._parse_ckpt_meta(base)
                    epseed = None
                    try:
                        import json
                        with open(label + ".json", "r") as f:
                            meta = json.load(f)
                            if gen is None:
                                gen = meta.get("gen")
                            if seed is None:
                                seed = meta.get("seed")
                            epseed = meta.get("epseed")
                    except Exception:
                        m_ep = re.search(r"epseed(\d+)", base)
                        epseed = int(m_ep.group(1)) if m_ep else None
                    if gen is not None or seed is not None or epseed is not None:
                        parts = []
                        if gen is not None:
                            parts.append(f"Gen {gen}")
                        if seed is not None:
                            parts.append(f"Seed {seed}")
                        if epseed is not None and epseed != seed:
                            parts.append(f"EpisodeSeed {epseed}")
                        meta_surface = self.font_big.render("  ".join(parts), True, (255, 255, 0))
                        screen.blit(meta_surface, (10, 30))
                        label_surface = self.font.render(base, True, (200, 200, 255))
                        screen.blit(label_surface, (10, 30 + meta_surface.get_height() + 2))
            except Exception:
                pass
            # Present final composite once, then restore camera
            pygame.display.flip()
            self._camera_x = original_camera_x
            return None
        while not done:
            skip_next = False
            nav: Optional[str] = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
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
                # Immediate quit shortcuts
                if event.type == pygame.KEYDOWN:
                    mods = event.mod if hasattr(event, 'mod') else 0
                    KMOD_GUI = getattr(pygame, 'KMOD_GUI', 0)
                    KMOD_META = getattr(pygame, 'KMOD_META', 0)
                    if event.key in (pygame.K_q, pygame.K_w) and (mods & (KMOD_GUI | KMOD_META)):
                        pygame.quit()
                        sys.exit(0)
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit(0)
                # Left/Right navigate generations ONLY during policy-driven playback
                # In manual play (policy is None), arrow keys must control the character.
                if policy is not None:
                    if event.type == pygame.KEYUP and event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        self._nav_ready = True
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT and self._nav_ready:
                        self._nav_ready = False
                        print("[nav] RIGHT detected during playback -> request prev (edge)")
                        nav = 'prev'
                        skip_next = True
                        done = True
                        celebrate_time_left_s = 0.0
                        post_end_time_left_s = 0.0
                        break
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT and self._nav_ready:
                        self._nav_ready = False
                        print("[nav] LEFT detected during playback -> request next (edge)")
                        nav = 'next'
                        skip_next = True
                        done = True
                        celebrate_time_left_s = 0.0
                        post_end_time_left_s = 0.0
                        break
                # SPACE toggles pause/resume for policy-driven playback
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and policy is not None:
                    paused = not paused
                    # When pausing, ensure celebration timers do not count down via stepping
                    continue
                # Speed controls: '=' to increase, '-' to decrease
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_EQUALS, getattr(pygame, 'K_PLUS', pygame.K_EQUALS), getattr(pygame, 'K_KP_PLUS', pygame.K_EQUALS)):
                        play_speed = min(64.0, play_speed * 1.5)
                        steps_per_frame = max(1, int(round(play_speed)))
                    elif event.key in (pygame.K_MINUS, getattr(pygame, 'K_KP_MINUS', pygame.K_MINUS)):
                        play_speed = max(0.25, play_speed / 1.5)
                        steps_per_frame = max(1, int(round(play_speed)))

            if skip_next:
                # Return early to allow outer loop to navigate
                try:
                    print(f"[nav] render_episode returning navigation='{nav}'")
                except Exception:
                    pass
                return nav

            for _ in range(steps_per_frame):
                # Freeze physics during celebration or post-end HUD hold
                if paused or celebrate_time_left_s > 0.0 or post_end_time_left_s > 0.0:
                    break  # freeze physics during celebration
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
                # Accumulate episode return in realtime so HUD can display recalculated fitness
                try:
                    episode_return += float(reward)
                except Exception:
                    pass
                if log_rays and info.get('ray_updated'):
                    # Pretty print 8 directions with type and distance
                    tmap = {0.0: 'none', 0.25: 'plat', 0.5: 'coin', 0.75: 'boots', 1.0: 'raccoon'}
                    types = info.get('ray_types') or []
                    dists = info.get('ray_dists') or []
                    dirs = ['0°','45°','90°','135°','180°','225°','270°','315°']
                    parts = []
                    for i in range(min(8, len(types), len(dists))):
                        parts.append(f"{dirs[i]}:{tmap.get(round(float(types[i]),2), str(types[i]))}@{float(dists[i]):.2f}")
                    print("rays:" , "  ".join(parts))
                # Start celebration if we just succeeded; otherwise hold final frame briefly to show HUD
                if done and info.get('succeed') and celebrate_time_left_s <= 0.0:
                    celebrate_time_left_s = 3.0
                    # Continue rendering without further stepping
                    done = False
                elif done and not info.get('succeed') and post_end_time_left_s <= 0.0:
                    post_end_time_left_s = 0.75
                    done = False

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
            if (not contrail) or first_frame:
                screen.fill((30, 30, 40))
                # platforms (tile brick texture if available; else solid rectangle)
                for (px, py, pw, ph) in cfg.platforms:
                    rect = self._world_rect_to_screen(px, py, pw, ph, cfg.height)
                    brick_base = self._sprite_base.get('brick')
                    if brick_base is not None:
                        # Choose a tile size relative to world scale (about 0.5 world units)
                        tile_px = max(8, int(round(0.5 * self.scale_x)))
                        try:
                            tile = self._get_sprite_scaled('brick', tile_px, tile_px)
                        except Exception:
                            tile = brick_base
                        tw, th = tile.get_width(), tile.get_height()
                        if tw <= 0 or th <= 0:
                            pygame.draw.rect(screen, (140, 140, 140), rect)
                        else:
                            old_clip = screen.get_clip()
                            screen.set_clip(rect)
                            # Start one tile earlier to ensure full coverage after clipping
                            start_x = rect.left - (rect.left % tw)
                            start_y = rect.top - (rect.top % th)
                            y = start_y
                            while y < rect.bottom:
                                x = start_x
                                while x < rect.right:
                                    screen.blit(tile, (x, y))
                                    x += tw
                                y += th
                            screen.set_clip(old_clip)
                    else:
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
            if ((not contrail) or first_frame) and hasattr(env, "powerup") and env.powerup and getattr(env.powerup, 'active', False):
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

            # When celebrating a success, show raccoon and dino standing together, centered
            # at the raccoon's original center. Dino stands to the LEFT, facing the raccoon.
            pair_drawn = False
            celebrating_success = (celebrate_time_left_s > 0.0) or ((info.get('succeed') if 'info' in locals() else False) and done)
            if celebrating_success:
                try:
                    rac = getattr(env, 'raccoon', None)
                    rx = rac.x if rac is not None else cfg.x_goal
                    ry = rac.y if rac is not None else 0.6
                    rw = float(getattr(rac, 'w', getattr(cfg, 'raccoon_w', 1.0))) if rac is not None else float(getattr(cfg, 'raccoon_w', 1.0))
                    rh = float(getattr(rac, 'h', getattr(cfg, 'raccoon_h', 1.0))) if rac is not None else float(getattr(cfg, 'raccoon_h', 1.0))
                    rac_img2 = self._get_sprite_scaled_to_rect('raccoon', rw, rh)
                    rcx2, rcy2 = self._world_to_screen(rx, ry, cfg.height)
                    rrect2 = rac_img2.get_rect()
                    # Player sprite (force facing RIGHT toward raccoon on the right)
                    dino2 = self._get_sprite_scaled_to_rect('dino', float(getattr(cfg, 'player_w', 0.8)), float(getattr(cfg, 'player_h', 0.8)))
                    dino2 = pygame.transform.flip(dino2, True, False)
                    drect2 = dino2.get_rect()
                    gap_px = 6
                    total_w = drect2.width + gap_px + rrect2.width
                    left_x = rcx2 - total_w // 2
                    base_bottom = rcy2 + rrect2.height // 2
                    # Position dino to the left, raccoon to the right
                    drect2.midbottom = (left_x + drect2.width // 2, base_bottom)
                    rrect2.midbottom = (left_x + drect2.width + gap_px + rrect2.width // 2, base_bottom)
                    screen.blit(dino2, drect2)
                    screen.blit(rac_img2, rrect2)
                    # Overlay boots on dino if powerup is active; align to dino's feet
                    if info.get('has_jump_powerup'):
                        boot_w = float(getattr(cfg, 'player_w', 0.8)) * 0.6
                        boot_h = float(getattr(cfg, 'player_h', 0.8)) * 0.38
                        boots2 = self._get_sprite_scaled_to_rect('boots', boot_w, boot_h)
                        # dino2 is flipped to face right; boots should match that orientation
                        boots2 = pygame.transform.flip(boots2, True, False)
                        brect2 = boots2.get_rect()
                        brect2.midbottom = (drect2.midbottom[0], drect2.midbottom[1] - int(0.02 * self.scale_y))
                        screen.blit(boots2, brect2)
                    pair_drawn = True
                except Exception:
                    pair_drawn = False

            if not pair_drawn:
                # player as 8-bit dinosaur sprite scaled to collider rect
                dino = self._get_sprite_scaled_to_rect('dino', float(getattr(cfg, 'player_w', 0.8)), float(getattr(cfg, 'player_h', 0.8)))
                if self._face_left:
                    dino = pygame.transform.flip(dino, True, False)
                # anchor bottom-center at (x, y)
                bx, by = self._world_to_screen(env.x, env.y, cfg.height)
                rect = dino.get_rect()
                rect.midbottom = (bx, by)
                if contrail:
                    ghost = dino.copy()
                    ghost.set_alpha(max(1, int(255 * max(0.0, min(1.0, contrail_alpha)))))
                    screen.blit(ghost, rect)
                else:
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
                if contrail:
                    ghost_b = boots.copy()
                    ghost_b.set_alpha(max(1, int(255 * max(0.0, min(1.0, contrail_alpha)))))
                    screen.blit(ghost_b, brect)
                else:
                    screen.blit(boots, brect)
            if show_hitboxes:
                pw = float(getattr(cfg, 'player_w', 0.8))
                ph = float(getattr(cfg, 'player_h', 0.8))
                pcx, pcy = self._world_to_screen(env.x, env.y + 0.5*ph, cfg.height)
                hr = pygame.Rect(pcx - int((pw * self.scale_x) * 0.5), pcy - int((ph * self.scale_y) * 0.5), int(pw * self.scale_x), int(ph * self.scale_y))
                pygame.draw.rect(screen, (255, 60, 60), hr, 1)

            # raccoon goal
            if ((not contrail) or first_frame) and not pair_drawn:
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

            # Heart celebration above raccoon
            if celebrate_time_left_s > 0.0 or (info.get('succeed') if 'info' in locals() else False):
                heart_offset = rh * 0.9
                hx, hy = self._world_to_screen(rx, ry + heart_offset, cfg.height)
                # If a heart sprite is available, render it; otherwise draw vector heart
                heart_surface = self._sprite_base.get('heart')
                if heart_surface is not None:
                    # Scale by raccoon height for a pleasing size
                    heart_img = self._get_sprite_scaled_by_world_h('heart', rh * 0.9)
                    rect_h = heart_img.get_rect()
                    rect_h.center = (hx, hy)
                    screen.blit(heart_img, rect_h)
                else:
                    size_px = int(max(16, rh * self._px_scale() * 0.8))
                    col = (255, 0, 64)
                    pygame.draw.circle(screen, col, (hx - size_px // 3, hy), size_px // 3)
                    pygame.draw.circle(screen, col, (hx + size_px // 3, hy), size_px // 3)
                    points = [(hx - size_px // 1.5, hy), (hx + size_px // 1.5, hy), (hx, hy + size_px)]
                    pygame.draw.polygon(screen, col, points)
                    pygame.draw.circle(screen, (255, 255, 255), (hx - size_px // 3, hy), size_px // 3, 1)
                    pygame.draw.circle(screen, (255, 255, 255), (hx + size_px // 3, hy), size_px // 3, 1)

            # HUD: coins, time (seconds), powerup
            coins_text = f"Coins: {info.get('coins_collected', 0)}/{info.get('coins_total', 0)}"
            time_left_s = max(0.0, float(info.get('time_remaining', 0)) * float(cfg.dt))
            time_text = f"Time: {time_left_s:.1f}s"
            pwr_text = "Power: x2 jump" if info.get('has_jump_powerup') else "Power: none"
            fit_text = f"Fitness: {episode_return:.3f}"
            hud_text = f"{coins_text}  {time_text}  {fit_text}  {pwr_text}"
            text_surface = self.font.render(hud_text, True, (230, 230, 230))
            screen.blit(text_surface, (10, 10))

            # Pause overlay
            if paused and policy is not None:
                pause_surface = self.font_big.render("Paused (SPACE to resume)", True, (255, 255, 0))
                pr = pause_surface.get_rect()
                pr.center = (self._size[0] // 2, 28)
                screen.blit(pause_surface, pr)

            # Visualize rays when requested
            if log_rays and info.get('ray_types') is not None and info.get('ray_dists') is not None:
                max_dist = float(getattr(cfg, 'raycast_max_dist', 30.0))
                origin_x = env.x
                origin_y = env.y + float(getattr(cfg, 'player_h', 0.8)) * 0.5
                types = list(info.get('ray_types'))
                dists = list(info.get('ray_dists'))
                colors = {
                    0.0: (120, 120, 120),   # none
                    0.25: (160, 160, 160),  # platform
                    0.5: (255, 220, 0),     # coin
                    0.75: (0, 220, 255),    # boots
                    1.0: (255, 255, 255),   # raccoon
                }
                for i in range(min(8, len(types), len(dists))):
                    ang = math.radians(45.0 * i)
                    dist = float(dists[i]) * max_dist
                    tx = origin_x + math.cos(ang) * dist
                    ty = origin_y + math.sin(ang) * dist
                    sx1, sy1 = self._world_to_screen(origin_x, origin_y, cfg.height)
                    sx2, sy2 = self._world_to_screen(tx, ty, cfg.height)
                    col = colors.get(float(round(types[i], 2)), (180, 180, 180))
                    pygame.draw.line(screen, col, (sx1, sy1), (sx2, sy2), 2)

            # Progress top-right: show generation number and rank
            if progress is not None:
                i, n = progress
                # Derive current gen for this label (prefer JSON)
                gen_val = None
                try:
                    import json
                    with open(label + ".json", "r") as f:  # type: ignore[arg-type]
                        meta = json.load(f)
                        gen_val = meta.get("gen")
                except Exception:
                    base = os.path.basename(label) if label else ""  # type: ignore[arg-type]
                    m = re.search(r"gen(\d+)", base)
                    gen_val = int(m.group(1)) if m else None
                rank_desc = max(1, int(n) - int(i) + 1)
                txt = f"Gen {gen_val} ({rank_desc}/{n})" if gen_val is not None else f"{rank_desc}/{n}"
                prog_surface = self.font_big.render(txt, True, (255, 255, 255))
                pr = prog_surface.get_rect()
                pr.top = 6
                pr.right = self._size[0] - 10
                screen.blit(prog_surface, pr)

            # Label + parsed meta — disabled (already shown top-right)
            if False and label:
                base = os.path.basename(label)
                gen, fit, seed = self._parse_ckpt_meta(base)
                # Prefer JSON sidecar for meta; we will not show recorded fitness/best to avoid confusion
                epseed = None
                try:
                    import json
                    with open(label + ".json", "r") as f:
                        meta = json.load(f)
                        if gen is None:
                            gen = meta.get("gen")
                        if seed is None:
                            seed = meta.get("seed")
                        epseed = meta.get("epseed")
                except Exception:
                    m_ep = re.search(r"epseed(\d+)", base)
                    epseed = int(m_ep.group(1)) if m_ep else None

                if gen is not None or seed is not None or epseed is not None:
                    meta = []
                    if gen is not None:
                        meta.append(f"Gen {gen}")
                    if seed is not None:
                        meta.append(f"Seed {seed}")
                    if epseed is not None and epseed != seed:
                        meta.append(f"EpisodeSeed {epseed}")
                    meta_text = "  ".join(meta)
                    meta_surface = self.font_big.render(meta_text, True, (255, 255, 0))
                    screen.blit(meta_surface, (10, 30))
                    label_surface = self.font.render(base, True, (200, 200, 255))
                    screen.blit(label_surface, (10, 30 + meta_surface.get_height() + 2))
                else:
                    label_surface = self.font.render(base, True, (200, 200, 255))
                    screen.blit(label_surface, (10, 30))

            # Early-out if a reload is requested between frames
            try:
                if reload_event is not None and callable(getattr(reload_event, 'is_set', None)) and reload_event.is_set():
                    print("[watch] reload requested during episode -> exiting early")
                    return None
            except Exception:
                pass
            pygame.display.flip()
            first_frame = False
            self.clock.tick(fps)

            # Count down celebration time and exit after delay
            if celebrate_time_left_s > 0.0:
                celebrate_time_left_s = max(0.0, celebrate_time_left_s - (1.0 / max(1, fps)))
                if celebrate_time_left_s <= 0.0:
                    # End loop after showing heart
                    break
            # Briefly hold on failure/time-up to allow HUD to show final fitness
            if post_end_time_left_s > 0.0:
                post_end_time_left_s = max(0.0, post_end_time_left_s - (1.0 / max(1, fps)))
                if post_end_time_left_s <= 0.0:
                    break


def render(weights_paths: Optional[List[str]] = None, speed: float = 1.0, show_hitboxes: bool = False, show_intro: bool = False, fullscreen: bool = False, log_rays: bool = False, seed: Optional[int] = None, oneshot: bool = False, contrail: bool = False, contrail_alpha: float = 0.01, contrail_mod: int = 1, start_from_latest: bool = False, allow_navigation: bool = True, reload_event: Optional[object] = None) -> None:
    try:
        print(f"[renderer] start: contrail={contrail} alpha={contrail_alpha} mod={contrail_mod} allow_navigation={allow_navigation}")
    except Exception:
        pass
    # If one checkpoint is provided, prefer JSON sidecar epseed/seed; else parse filename
    seed_for_env: Optional[int] = seed
    if seed_for_env is None and weights_paths and len(weights_paths) == 1:
        json_seed = None
        try:
            import json
            with open(weights_paths[0] + ".json", "r") as f:
                meta = json.load(f)
                json_seed = meta.get("epseed") or meta.get("seed")
        except Exception:
            json_seed = None
        if json_seed is not None:
            seed_for_env = int(json_seed)
        else:
            base = os.path.basename(weights_paths[0])
            m = re.search(r"seed(\d+)", base)
            if m:
                try:
                    seed_for_env = int(m.group(1))
                except Exception:
                    seed_for_env = None
    env = PlatformerEnv(GameConfig(), seed=seed_for_env)
    renderer = Renderer(fullscreen=fullscreen)

    if not weights_paths:
        if show_intro:
            renderer.show_intro(env)
        renderer.render_episode(env, None, speed=speed, show_hitboxes=show_hitboxes, log_rays=log_rays, contrail=contrail, contrail_alpha=contrail_alpha, contrail_mod=contrail_mod)
        renderer.close()
        return

    # Sort best-to-worst (highest fitness first). Prefer JSON sidecar 'fit'; fallback to filename.
    def get_fit_and_gen(p: str) -> tuple:
        fit_val = None
        gen_val = None
        try:
            import json
            with open(p + ".json", "r") as f:
                meta = json.load(f)
                fit_val = meta.get("fit")
                gen_val = meta.get("gen")
        except Exception:
            pass
        if fit_val is None or gen_val is None:
            gen, fit, _seed = renderer._parse_ckpt_meta(os.path.basename(p))
            if fit_val is None:
                fit_val = fit
            if gen_val is None:
                gen_val = gen
        # Defaults push unknowns to the end
        if fit_val is None:
            fit_val = float('-inf')
        if gen_val is None:
            gen_val = -1
        return float(fit_val), int(gen_val)

    # Sort by generation DESC (best-to-worst chronology). Tie-breaker: higher fit first.
    weights_paths = sorted(
        weights_paths,
        key=lambda p: (-get_fit_and_gen(p)[1], -get_fit_and_gen(p)[0])
    )

    # Debug: show planned render order
    try:
        order_lines = []
        for p in weights_paths:
            fit_v, gen_v = get_fit_and_gen(p)
            order_lines.append(f"gen={gen_v} fit={fit_v:.3f} file={os.path.basename(p)}")
        if order_lines:
            print("[renderer] Render order (gen desc, fit desc):")
            for line in order_lines:
                print("[renderer]  ", line)
    except Exception:
        pass

    total = len(weights_paths)
    # After sorting gen DESC, the newest is first. Start from index 1.
    idx = 1
    loop_iter = 0
    while idx <= len(weights_paths):
        path = weights_paths[idx - 1]
        try:
            print(f"[render-loop] iter={loop_iter} idx={idx}/{total} file={os.path.basename(path)}")
            sys.stdout.flush()
        except Exception:
            pass
        try:
            fit_v, gen_v = get_fit_and_gen(path)
            print(f"[renderer] Playing {idx}/{total}: gen={gen_v} fit={fit_v:.3f} file={os.path.basename(path)}")
        except Exception:
            pass
        # If a seed tag exists per file, reseed env to reproduce that episode deterministically
        base = os.path.basename(path)
        # CLI seed overrides epseed/seed; otherwise prefer JSON sidecar epseed over seed when present
        if seed is not None:
            print(f"[seed] Using CLI-provided seed={seed}")
            env = PlatformerEnv(GameConfig(), seed=seed)
        else:
            chosen_seed = None
            try:
                import json
                with open(path + ".json", "r") as f:
                    meta = json.load(f)
                    chosen_seed = meta.get("epseed") or meta.get("seed")
                    print(f"[seed] Loaded JSON sidecar for {os.path.basename(path)} -> epseed={meta.get('epseed')} seed={meta.get('seed')}")
            except Exception as e:
                print(f"[seed] No/invalid JSON for {os.path.basename(path)}: {e}")
                chosen_seed = None
            if chosen_seed is None:
                m_ep = re.search(r"epseed(\d+)", base)
                m = m_ep if m_ep else re.search(r"seed(\d+)", base)
                if m is not None:
                    try:
                        chosen_seed = int(m.group(1))
                    except Exception:
                        chosen_seed = None
            if chosen_seed is not None:
                try:
                    print(f"[seed] Reseeding env with chosen_seed={chosen_seed}")
                    env = PlatformerEnv(GameConfig(), seed=int(chosen_seed))
                except Exception:
                    env = PlatformerEnv(GameConfig())
            else:
                print("[seed] No seed found; using default env seed")

        cfg = MLPPolicyConfig(input_size=env.observation_size, output_size=env.action_size)
        print(f"[policy] cfg: input_size={cfg.input_size} output_size={cfg.output_size}")
        policy = MLPPolicy(cfg)
        print(f"[weights] loading: {os.path.basename(path)}")
        flat = np.load(path)
        print(f"[weights] shape={flat.shape} dtype={flat.dtype}")
        policy.set_flat(flat)
        print("[policy] set_flat completed")
        print(f"[nav] calling render_episode(idx={idx}/{total}, oneshot={oneshot}, allow_navigation={allow_navigation})")
        nav = renderer.render_episode(env, policy, speed=speed, label=path, progress=(idx, total), show_hitboxes=show_hitboxes, log_rays=log_rays, contrail=contrail, contrail_alpha=contrail_alpha, contrail_mod=contrail_mod, reload_event=reload_event)
        print(f"[nav] returned from render_episode with nav={nav}")
        # If a watcher requested reload, exit render immediately so caller can restart
        try:
            if reload_event is not None and callable(getattr(reload_event, 'is_set', None)) and reload_event.is_set():
                print("[watch] reload requested -> returning from render()")
                return None
        except Exception:
            pass
        if oneshot:
            print("[nav] oneshot=True -> returning from render() immediately")
            return None
        # After playback, allow navigation: Right advances, Left goes back; otherwise in contrail do not auto-advance
        print(f"[nav] post-episode: received nav={nav} allow_navigation={allow_navigation} idx={idx}")
        if allow_navigation and nav == 'next':
            next_idx = 1 if idx >= total else idx + 1
            wrap_note = " (wrap)" if next_idx == 1 and idx == total else ""
            print(f"[nav] applying 'next': {idx} -> {next_idx}{wrap_note} (before debounce)")
            idx = next_idx
            renderer._debounce_navigation()
            print(f"[nav] applied 'next': new idx={idx}; continuing loop")
            loop_iter += 1
            continue
        if allow_navigation and nav == 'prev':
            prev_idx = total if idx <= 1 else idx - 1
            wrap_note = " (wrap)" if prev_idx == total and idx == 1 else ""
            print(f"[nav] applying 'prev': {idx} -> {prev_idx}{wrap_note} (before debounce)")
            idx = prev_idx
            renderer._debounce_navigation()
            print(f"[nav] applied 'prev': new idx={idx}; continuing loop")
            loop_iter += 1
            continue
        if contrail:
            # In contrail snapshot, do not auto-progress. Wait briefly for navigation keys only.
            print("[nav] contrail hold: waiting for nav keys")
            hold_nav = renderer.hold_or_navigate(seconds=3.0) if allow_navigation else None
            if allow_navigation and hold_nav == 'next':
                next_idx = 1 if idx >= total else idx + 1
                wrap_note = " (wrap)" if next_idx == 1 and idx == total else ""
                print(f"[nav] hold next -> {idx} -> {next_idx}{wrap_note}")
                idx = next_idx
                renderer._debounce_navigation()
                loop_iter += 1
            elif allow_navigation and hold_nav == 'prev':
                prev_idx = total if idx <= 1 else idx - 1
                wrap_note = " (wrap)" if prev_idx == total and idx == 1 else ""
                print(f"[nav] hold prev -> {idx} -> {prev_idx}{wrap_note}")
                idx = prev_idx
                renderer._debounce_navigation()
                loop_iter += 1
            else:
                # Stay on the same generation until user navigates
                if not allow_navigation:
                    print("[nav] navigation disabled (watch mode) -> staying on same generation")
                else:
                    print("[nav] no nav key -> staying on same generation")
                continue
        else:
            print("[nav] normal hold: waiting for nav keys or timeout")
            hold_nav = renderer.hold_or_navigate(seconds=3.0) if allow_navigation else None
            if allow_navigation and hold_nav == 'next':
                next_idx = 1 if idx >= total else idx + 1
                wrap_note = " (wrap)" if next_idx == 1 and idx == total else ""
                print(f"[nav] hold next -> {idx} -> {next_idx}{wrap_note}")
                idx = next_idx
                renderer._debounce_navigation()
                loop_iter += 1
            elif allow_navigation and hold_nav == 'prev':
                prev_idx = total if idx <= 1 else idx - 1
                wrap_note = " (wrap)" if prev_idx == total and idx == 1 else ""
                print(f"[nav] hold prev -> {idx} -> {prev_idx}{wrap_note}")
                idx = prev_idx
                renderer._debounce_navigation()
                loop_iter += 1
            else:
                # Loop same checkpoint again
                if not allow_navigation:
                    print("[nav] navigation disabled (watch mode) -> loop same")
                else:
                    print("[nav] no nav key -> loop same")
                loop_iter += 1
                continue

    renderer.close()
