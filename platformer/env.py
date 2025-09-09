from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import math
import numpy as np


Platform = Tuple[float, float, float, float]  # (x, y, w, h) in world units, y upwards


# Common entity and collider helpers
@dataclass
class GameEntity:
    kind: str
    x: float
    y: float
    # Axis-aligned rectangle size in world units, centered at (x, y)
    w: float
    h: float
    active: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)


def entities_collide(a: GameEntity, b: GameEntity) -> bool:
    """Axis-aligned rectangle overlap test using entity centers and sizes."""
    a_left = a.x - a.w * 0.5
    a_right = a.x + a.w * 0.5
    a_bottom = a.y - a.h * 0.5
    a_top = a.y + a.h * 0.5
    b_left = b.x - b.w * 0.5
    b_right = b.x + b.w * 0.5
    b_bottom = b.y - b.h * 0.5
    b_top = b.y + b.h * 0.5
    return (a_right > b_left) and (a_left < b_right) and (a_top > b_bottom) and (a_bottom < b_top)

@dataclass
class GameConfig:
    width: float = 50.0  # viewport width in world units (one screen)
    height: float = 10.0
    level_screens: int = 2  # number of screens the level spans horizontally

    # Simulation timing
    tickrate_hz: float = 30.0  # simulation updates per second
    episode_time_s: float = 60.0  # episode duration in seconds
    dt: float = 0.1
    gravity: float = -25.0  # y-axis points upward
    move_accel: float = 120.0
    max_speed: float = 8.0
    jump_velocity: float = 15.0
    friction: float = 8.0

    episode_length: int = 250  # derived from episode_time_s * tickrate_hz

    # Platforms: specify as list of (x, y, w, h), y upwards
    platforms: List[Platform] = field(default_factory=list)
    # ASCII level map used as the primary level definition. Each character is
    # one tile of size tile_size. Legend: '#': platform, 'P': player spawn,
    # 'C': coin, 'B': boots, 'R': raccoon, '.' or ' ' empty.
    level_map: List[str] = field(default_factory=lambda: [
        "###################################################",
        "#....C......C.......C........................######",
        "#.............................................#####",
        "#.C......C.......C......C......................####",
        "#################################...............###",
        "#..............................C#................##",
        "#..............................C#.................#",
        "#............................####.................#",
        "#..............................................C..#",
        "#.................................................#",
        "#.................................................#",
        "#.......................C.........................#",
        "#.......###########################################",
        "#................................................C#",
        "#..............................C.C.C.C.C.C.......C#",
        "#C....................###.......C.C.C.C.C........C#",
        "#................................................C#",
        "#...............................................###",
        "#.................................................#",
        "#.....###.........................................#",
        "#.................................................#",
        "#.................................................#",
        "#..............................C..................#",
        "#................############################.....#", 
        "#................#................................#",
        "#C............C..#................................#",
        "#C...............#...................#............#",
        "###..............#...................#.CCCCC......#",
        "#.......................C............#.CCCCC......#",
        "#....................................#.CCCCC......#",
        "#....................................##############",
        "#.................................................#",
        "#...........################......................#",
        "#..........................#......................#",
        "#..........................#......................#",
        "#..........................################.......#",
        "#.........................................#.......#",
        "#........P................................#.......#",
        "#....#########............................#...R.. #",
        "#....#########..................B.........#.......#",
        "###################################################"
    ])
    tile_size: float = 1.0

    # Coins and rewards
    coin_w: float = 1.0
    coin_h: float = 1.0
    coin_reward: float = 1000.0
    finish_speed_bonus: float = 50.0  # bonus scaled by remaining time ratio
    finish_base_bonus: float = 5.0  # flat bonus on finish
    failure_penalty: float = 5000.0  # applied on any non-successful termination
    jump_penalty: float = 0.2  # small penalty when a jump is initiated

    # Movement control
    air_control_scale: float = 0.333  # fraction of horizontal accel allowed while airborne
    # Running (Mario-style): increases max speed while on ground
    run_speed_multiplier: float = 1.7
    # Player collider size (rect)
    player_w: float = 1.5
    player_h: float = 3

    # Powerup: jump multiplier near goal
    powerup_w: float = 2
    powerup_h: float = 1
    powerup_jump_multiplier: float = 2.0
    powerup_offset_x: float = 1.5  # place this far before goal line
    powerup_height: float = 0     # y above ground
    # Raccoon goal entity
    raccoon_w: float = 1.5
    raccoon_h: float = 3.0

    # Variable jump (pressure-sensitive): max time the jump can be held
    variable_jump_max_hold_s: float = 0.333
    # Cooldown after landing before another jump is allowed
    jump_cooldown_s: float = 0.10

    # Raycasting for observations
    raycast_enabled: bool = True
    raycast_max_dist: float = 30.0
    raycast_interval_s: float = 0.25

    # Exploration bitfield observation (8x8 grid across the map)
    exploration_grid_size: int = 8
    exploration_enabled: bool = True

    def __post_init__(self) -> None:
        # Derive dt and episode_length from tickrate and episode time
        if self.tickrate_hz <= 0.0:
            self.tickrate_hz = 50.0
        self.dt = 1.0 / float(self.tickrate_hz)
        self.episode_length = int(max(1, round(self.episode_time_s * self.tickrate_hz)))
        # If ASCII map provided, ensure world dimensions accommodate it
        if self.level_map:
            rows = self.level_map
            nrows = len(rows)
            ncols = max((len(r) for r in rows), default=0)
            # Use the ASCII map dimensions as the world dimensions exactly so
            # the bottom row corresponds to y=0 and aligns with the window bottom.
            self.width = float(ncols) * float(self.tile_size)
            self.height = float(nrows) * float(self.tile_size)
        if not self.platforms:
            # Start with ground only; rest will be procedurally generated in env.reset
            self.platforms = [
                (-1e6, -0.5, 2e6, 0.5),  # infinite ground strip at y=0 top
            ]
        # Ensure positive sizes
        self.player_w = max(1e-6, float(self.player_w))
        self.player_h = max(1e-6, float(self.player_h))
        self.coin_w = max(1e-6, float(self.coin_w))
        self.coin_h = max(1e-6, float(self.coin_h))
        self.powerup_w = max(1e-6, float(self.powerup_w))
        self.powerup_h = max(1e-6, float(self.powerup_h))
        self.raccoon_w = max(1e-6, float(self.raccoon_w))
        self.raccoon_h = max(1e-6, float(self.raccoon_h))


class PlatformerEnv:
    """Headless 2D platformer environment.

    Coordinate system: x rightward, y upward. Gravity is negative.
    Collisions: AABB with platform tops; simple ground and platforms.
    Actions (discrete):
        0: idle
        1: left
        2: right
        3: jump
        4: left + jump
        5: right + jump
        6: run
        7: run + left
        8: run + right
        9: run + jump
        10: run + left + jump
        11: run + right + jump
    Observation (np.float32):
        [x/width, y/height, vx/max_speed, vy/jump_velocity, on_ground(0/1),
         coins_collected/num_coins, time_remaining_ratio]
    """

    def __init__(self, config: Optional[GameConfig] = None, seed: Optional[int] = None) -> None:
        self.config = config or GameConfig()
        self.rng = np.random.default_rng(seed)
        self.reset()


    @property
    def action_size(self) -> int:
        return 12

    @property
    def observation_size(self) -> int:
        # Base 7 + (8 directions * [type, distance]) + 1 staleness indicator when raycasting enabled
        base = 7 + (17 if getattr(self.config, 'raycast_enabled', True) else 0)
        if getattr(self.config, 'exploration_enabled', True):
            base += 64
        return base

    def reset(self) -> np.ndarray:
        self.timestep = 0
        self.x = 2.0
        self.y = 1.0
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = True
        self.prev_x = self.x
        self.done = False

        # Raycast cache/state
        if self.config.raycast_enabled:
            self._ray_obs_types: List[float] = [0.0] * 8
            self._ray_obs_dists: List[float] = [1.0] * 8
            # Trigger an immediate raycast on first frame after reset
            self._raycast_time_left_s: float = 0.0

        # Exploration bitfield state
        if self.config.exploration_enabled:
            self._explore_bits: int = 0  # 64-bit bitfield (use Python int)

        # Deterministically (re)generate platforms across the full level, or from ASCII map
        cfg = self.config
        level_length = cfg.width * max(1, int(cfg.level_screens))
        # If using ASCII map, do NOT add the infinite ground; the bottom row of
        # tiles defines the floor. Otherwise, retain the infinite ground strip.
        plats: List[Platform] = [] if cfg.level_map else [(-1e6, -0.5, 2e6, 0.5)]
        coins_from_map: List[GameEntity] = []
        powerup_from_map: Optional[GameEntity] = None
        raccoon_from_map: Optional[GameEntity] = None
        player_spawn: Optional[Tuple[float, float]] = None
        if cfg.level_map:
            rows = cfg.level_map
            nrows = len(rows)
            ncols = max((len(r) for r in rows), default=0)
            tile = float(cfg.tile_size)
            def ch_at(r: int, c: int) -> str:
                if r < 0 or r >= nrows or c < 0 or c >= len(rows[r]):
                    return ' '
                return rows[r][c]
            for r in range(nrows):
                y_bottom = cfg.height - (r + 1) * tile
                c = 0
                while c < ncols:
                    ch = ch_at(r, c)
                    if ch == '#':
                        start = c
                        while c + 1 < ncols and ch_at(r, c + 1) == '#':
                            c += 1
                        end = c
                        px = start * tile
                        pw = (end - start + 1) * tile
                        py = y_bottom
                        ph = tile
                        plats.append((px, py, pw, ph))
                    elif ch == 'C':
                        cx = (c + 0.5) * tile
                        cy = y_bottom + 0.5 * tile
                        coins_from_map.append(GameEntity(kind="coin", x=cx, y=cy, w=cfg.coin_w, h=cfg.coin_h))
                    elif ch == 'B':
                        bx = (c + 0.5) * tile
                        by = y_bottom + 0.5 * tile
                        powerup_from_map = GameEntity(kind="boots", x=bx, y=by, w=cfg.powerup_w, h=cfg.powerup_h, active=True)
                    elif ch == 'R':
                        rx = (c + 0.5) * tile
                        ry = y_bottom + 0.5 * tile
                        raccoon_from_map = GameEntity(kind="raccoon", x=rx, y=ry, w=cfg.raccoon_w, h=cfg.raccoon_h)
                    elif ch == 'P':
                        px = (c + 0.5) * tile
                        py_spawn = y_bottom
                        player_spawn = (px, py_spawn)
                    c += 1
            self.config.platforms = plats
        else:
            widths = [2.0, 4.0, 8.0]  # 1x2, 1x4, 1x8 (height fixed at 1)
            ph = 1.0
            top_levels = [1, 1.25, 1.75, 5]
            spacing = 10.0
            i = 0
            x = 2.0
            while x < (level_length - 2.0):
                if x < 1:
                    x += spacing
                    continue
                pw = widths[i % len(widths)]
                top_y = top_levels[i % len(top_levels)]
                px = min(max(2.0, x), level_length - pw - 2.0)
                py = top_y - ph
                plats.append((px, py, pw, ph))
                i += 1
                x += spacing
            self.config.platforms = plats

        # Player entity (center anchored at collider center)
        if cfg.level_map and player_spawn is not None:
            self.x, self.y = float(player_spawn[0]), float(player_spawn[1])
        self.player_entity = GameEntity(
            kind="player",
            x=self.x,
            y=self.y + self.config.player_h * 0.5,
            w=self.config.player_w,
            h=self.config.player_h,
        )

        # Coins: from ASCII map if available; otherwise place on platforms
        self.coins: List[GameEntity] = []
        if cfg.level_map and coins_from_map:
            self.coins = coins_from_map
        else:
            for (px, py, pw, ph) in self.config.platforms[1:]:  # skip ground
                cx = px + pw * 0.5
                cy = py + ph + (self.config.coin_h * 0.5) + 0.12
                self.coins.append(GameEntity(
                    kind="coin",
                    x=cx,
                    y=cy,
                    w=self.config.coin_w,
                    h=self.config.coin_h,
                ))

        # Jump powerup
        self.has_jump_powerup: bool = False
        mid_x = self.config.width * 0.5
        if cfg.level_map and powerup_from_map is not None:
            self.powerup = powerup_from_map
        else:
            self.powerup = GameEntity(
                kind="boots",
                x=mid_x,
                y=0.6,
                w=self.config.powerup_w,
                h=self.config.powerup_h,
                active=True,
            )

        # Raccoon goal
        if cfg.level_map and raccoon_from_map is not None:
            self.raccoon = raccoon_from_map
        else:
            self.raccoon = GameEntity(
                kind="raccoon",
                x=level_length - 2.0,
                y=0.6,
                w=self.config.raccoon_w,
                h=self.config.raccoon_h,
            )

        # Variable jump state
        self.is_in_jump = False
        self.jump_hold_time_s = 0.0
        self.jump_cut_applied = False
        self.jump_cooldown_left_s = 0.0

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Call reset() before stepping a finished episode.")

        cfg = self.config
        dt = cfg.dt

        # Decrement jump cooldown timer
        if self.jump_cooldown_left_s > 0.0:
            self.jump_cooldown_left_s = max(0.0, self.jump_cooldown_left_s - dt)

        # Horizontal control
        ax = 0.0
        wants_left = action in (1, 4, 7, 10)
        wants_right = action in (2, 5, 8, 11)
        input_accel = cfg.move_accel * (1.0 if self.on_ground else cfg.air_control_scale)
        if wants_left and not wants_right:
            ax -= input_accel
        elif wants_right and not wants_left:
            ax += input_accel

        # Friction/damping: apply strong damping when no input; light damping when steering
        if (wants_left and wants_right) or (not wants_left and not wants_right):
            ax -= cfg.friction * self.vx
        else:
            ax -= 0.2 * cfg.friction * self.vx

        # Jump (pressure-sensitive)
        wants_jump = action in (3, 4, 5, 9, 10, 11)
        # Run modifier (only affects max speed while on ground)
        wants_run = action in (6, 7, 8, 9, 10, 11)
        jump_initiated = False
        if wants_jump and self.on_ground and not self.is_in_jump and self.jump_cooldown_left_s <= 0.0:
            mult = cfg.powerup_jump_multiplier if self.has_jump_powerup else 1.0
            self.vy = cfg.jump_velocity * mult
            self.on_ground = False
            self.is_in_jump = True
            self.jump_hold_time_s = 0.0
            self.jump_cut_applied = False
            jump_initiated = True
        # Track hold time while in jump
        if self.is_in_jump and wants_jump and self.jump_hold_time_s < cfg.variable_jump_max_hold_s:
            self.jump_hold_time_s += dt
        # If released early during ascent, cut vertical velocity proportionally to held time
        if (
            self.is_in_jump
            and not wants_jump
            and not self.jump_cut_applied
            and self.jump_hold_time_s < cfg.variable_jump_max_hold_s
            and self.vy > 0.0
        ):
            hold_frac = max(0.0, min(1.0, self.jump_hold_time_s / max(1e-6, cfg.variable_jump_max_hold_s)))
            self.vy = self.vy * hold_frac
            self.jump_cut_applied = True

        # Integrate velocity
        self.vx += ax * dt
        max_speed_now = cfg.max_speed * (cfg.run_speed_multiplier if (self.on_ground and wants_run) else 1.0)
        self.vx = float(np.clip(self.vx, -max_speed_now, max_speed_now))
        self.vy += cfg.gravity * dt

        # Axis-separated integration with rectangle player collider
        player_half_w = cfg.player_w * 0.5
        player_h = cfg.player_h
        was_on_ground = self.on_ground

        # Horizontal move and resolve against platform sides
        new_x = self.x + self.vx * dt
        for (px, py, pw, ph) in self.config.platforms:
            left = px
            right = px + pw
            bottom = py
            top = py + ph
            # check vertical overlap of player's AABB with platform
            player_bottom = self.y
            player_top = self.y + player_h
            vertical_overlap = (player_top > bottom) and (player_bottom < top)
            if not vertical_overlap:
                continue
            # moving right into left side
            if (self.x + player_half_w) <= left and (new_x + player_half_w) > left:
                new_x = left - player_half_w
                self.vx = 0.0
            # moving left into right side
            if (self.x - player_half_w) >= right and (new_x - player_half_w) < right:
                new_x = right + player_half_w
                self.vx = 0.0

        # Vertical move and resolve against platform tops/bottoms
        new_y = self.y + self.vy * dt
        landed = False
        for (px, py, pw, ph) in self.config.platforms:
            left = px
            right = px + pw
            bottom = py
            top = py + ph
            # check horizontal overlap of player's AABB with platform
            player_left = new_x - player_half_w
            player_right = new_x + player_half_w
            horizontal_overlap = (player_right > left) and (player_left < right)
            if not horizontal_overlap:
                continue
            # moving up into platform bottom (head hit)
            if (self.y + player_h) <= bottom and (new_y + player_h) > bottom:
                new_y = bottom - player_h
                self.vy = 0.0
                self.is_in_jump = False
            # moving down onto platform top (landing)
            if self.y >= top and new_y < top:
                new_y = top
                self.vy = 0.0
                self.on_ground = True
                landed = True

        if not landed:
            # Not on ground if we have vertical velocity
            self.on_ground = False if self.vy != 0.0 else self.on_ground

        # Commit new position
        self.prev_x = self.x
        self.x = new_x
        self.y = new_y

        # Update exploration bits
        if cfg.exploration_enabled:
            self._mark_explored(self.x, self.y)

        # If we just landed this frame, start jump cooldown
        if (not was_on_ground) and self.on_ground:
            self.jump_cooldown_left_s = cfg.jump_cooldown_s

        # End of jump when landing or starting to descend past apex
        if self.is_in_jump and (self.on_ground or self.vy <= 0.0):
            # Once descending, no further cut effects matter this jump
            self.is_in_jump = False

        # Raycast update cadence
        ray_updated = False
        if cfg.raycast_enabled:
            # Decrement and update when timer elapses
            if not hasattr(self, '_raycast_time_left_s'):
                self._raycast_time_left_s = 0.0
            self._raycast_time_left_s = float(self._raycast_time_left_s) - dt
            if self._raycast_time_left_s <= 0.0:
                types, dists = self._compute_raycast_observations()
                self._ray_obs_types = types
                self._ray_obs_dists = dists
                self._raycast_time_left_s += float(max(1e-6, cfg.raycast_interval_s))
                ray_updated = True

        # Coin collection after movement
        coin_reward_total = 0.0
        # keep player entity in sync with physics position
        self.player_entity.x = self.x
        self.player_entity.y = self.y + cfg.player_h * 0.5
        for coin in self.coins:
            if not coin.active:
                continue
            if entities_collide(self.player_entity, coin):
                coin.active = False
                coin_reward_total += cfg.coin_reward

        # Powerup collection
        if self.powerup and self.powerup.active:
            if entities_collide(self.player_entity, self.powerup):
                self.powerup.active = True  # stays in world but flagged active; no need to draw after pickup handled by renderer
                self.powerup.active = False
                self.has_jump_powerup = True

        # Terminal conditions
        self.timestep += 1
        # Success if colliding with raccoon
        reached_goal = entities_collide(self.player_entity, self.raccoon)
        fell_out = self.y < -10.0
        time_up = self.timestep >= cfg.episode_length
        left_out = self.x < 0.0

        # Allow finishing without all coins; incentivize coins and speed via rewards
        succeed = reached_goal
        self.done = bool(succeed or fell_out or time_up or left_out)

        # Reward shaping: forward progress and coin pickups; finish rewards scale with coins and speed; no survival bonus
        progress = max(0.0, self.x - self.prev_x)
        reward = progress
        reward += coin_reward_total
        if jump_initiated:
            reward -= cfg.jump_penalty
        if succeed:
            time_ratio = max(0.0, 1.0 - (self.timestep / float(max(1, cfg.episode_length))))
            reward += cfg.finish_base_bonus + cfg.finish_speed_bonus * time_ratio
        # Extreme penalty for not finishing when episode ends
        if self.done and not succeed:
            reward -= cfg.failure_penalty

        obs = self._get_observation()
        info = {
            "x": self.x,
            "y": self.y,
            "reached_goal": reached_goal,
            "fell_out": fell_out,
            "left_out": left_out,
            "time_up": time_up,
            "succeed": succeed,
            "coins_collected": self.coins_collected_count,
            "coins_total": len(self.coins),
            "collided_top_y": None,
            "time_remaining": max(0, cfg.episode_length - self.timestep),
            "has_jump_powerup": self.has_jump_powerup,
            # Raycast diagnostics for renderer/UI consumers
            "ray_updated": ray_updated if cfg.raycast_enabled else False,
            "ray_types": (self._ray_obs_types if cfg.raycast_enabled and hasattr(self, '_ray_obs_types') else None),
            "ray_dists": (self._ray_obs_dists if cfg.raycast_enabled and hasattr(self, '_ray_obs_dists') else None),
        }
        return obs, float(reward), self.done, info

    @property
    def coins_collected_count(self) -> int:
        return sum(1 for c in self.coins if not c.active) if hasattr(self, "coins") else 0

    def _get_observation(self) -> np.ndarray:
        cfg = self.config
        coins_total = max(1, len(self.coins)) if hasattr(self, "coins") else 1
        time_remaining_ratio = 1.0 - (self.timestep / float(max(1, cfg.episode_length)))
        base = [
            np.clip(self.x / max(cfg.width, 1e-6), 0.0, 1.0),
            np.clip(self.y / max(cfg.height, 1e-6), -1.0, 2.0),
            np.clip(self.vx / max(cfg.max_speed, 1e-6), -1.0, 1.0),
            np.clip(self.vy / max(cfg.jump_velocity, 1e-6), -2.0, 2.0),
            1.0 if self.on_ground else 0.0,
            float(self.coins_collected_count) / float(coins_total),
            np.clip(time_remaining_ratio, 0.0, 1.0),
        ]
        if cfg.raycast_enabled:
            types = getattr(self, '_ray_obs_types', [0.0] * 8)
            dists = getattr(self, '_ray_obs_dists', [1.0] * 8)
            base.extend(types)
            base.extend(dists)
            # Add normalized "age" of the raycast: 0 fresh (cast this frame), 1 stale (just before next cast)
            interval = float(max(1e-6, getattr(cfg, 'raycast_interval_s', 0.25)))
            time_left = float(getattr(self, '_raycast_time_left_s', 0.0))
            age_ratio = 1.0 - float(max(0.0, min(1.0, time_left / interval)))
            base.append(age_ratio)
        if cfg.exploration_enabled:
            bits = getattr(self, '_explore_bits', 0)
            # Append 64 binary features (0/1) row-major from bottom to top
            for i in range(64):
                base.append(1.0 if (bits >> i) & 1 else 0.0)
        return np.array(base, dtype=np.float32)

    def copy(self) -> "PlatformerEnv":
        # Lightweight copy for parallel experimentation if needed
        new_env = PlatformerEnv(self.config, seed=None)
        new_env.timestep = self.timestep
        new_env.x = self.x
        new_env.y = self.y
        new_env.vx = self.vx
        new_env.vy = self.vy
        new_env.on_ground = self.on_ground
        new_env.prev_x = self.prev_x
        new_env.done = self.done
        # Deep copy coin state
        new_env.coins = [{"x": c["x"], "y": c["y"], "collected": c["collected"]} for c in self.coins]
        # Copy powerup state
        new_env.powerup = {"x": self.powerup["x"], "y": self.powerup["y"], "collected": self.powerup["collected"]}
        new_env.has_jump_powerup = self.has_jump_powerup
        # Copy variable jump state
        new_env.is_in_jump = self.is_in_jump
        new_env.jump_hold_time_s = self.jump_hold_time_s
        new_env.jump_cut_applied = self.jump_cut_applied
        return new_env

    # --------------------------- Raycasting helpers ---------------------------
    def _compute_raycast_observations(self) -> Tuple[List[float], List[float]]:
        """Cast 8 rays (0..315 degrees every 45 deg) from player center.
        Returns two lists length 8 each: types (0..1) and normalized distances (0..1).
        Types mapping (normalized): 0.0 none, 0.25 platform, 0.5 coin, 0.75 powerup, 1.0 raccoon.
        Distances are clipped to raycast_max_dist and divided by that value.
        """
        cfg = self.config
        origin_x = self.x
        origin_y = self.y + cfg.player_h * 0.5
        max_dist = float(max(1e-6, cfg.raycast_max_dist))
        type_map = {
            'none': 0.0,
            'platform': 0.25,
            'coin': 0.5,
            'boots': 0.75,
            'raccoon': 1.0,
        }
        hit_types: List[float] = []
        hit_dists: List[float] = []
        # Identify platform we are currently standing on (if any)
        under_plat: Optional[Tuple[float, float, float, float]] = None
        for (px, py, pw, ph) in cfg.platforms:
            top = py + ph
            if abs(self.y - top) < 1e-6 and (self.x >= px - 1e-6) and (self.x <= px + pw + 1e-6):
                under_plat = (px, py, pw, ph)
                break

        for i in range(8):
            ang = math.radians(45.0 * i)
            dx = math.cos(ang)
            dy = math.sin(ang)
            t_best = None
            t_type: str = 'none'
            # platforms
            for (px, py, pw, ph) in cfg.platforms:
                t = self._ray_intersect_aabb(origin_x, origin_y, dx, dy, px, py, px + pw, py + ph)
                # Ignore immediate self-floor hit within a small epsilon distance
                if under_plat is not None and (px, py, pw, ph) == under_plat and t is not None:
                    epsilon = float(getattr(cfg, 'tile_size', 1.0)) * 0.5
                    if t < epsilon:
                        t = None
                if t is not None and 0.0 <= t <= max_dist:
                    if t_best is None or t < t_best:
                        t_best = t
                        t_type = 'platform'
            # coins
            for coin in getattr(self, 'coins', []):
                if not coin.active:
                    continue
                t = self._ray_intersect_aabb(origin_x, origin_y, dx, dy, coin.x - coin.w * 0.5, coin.y - coin.h * 0.5, coin.x + coin.w * 0.5, coin.y + coin.h * 0.5)
                if t is not None and 0.0 <= t <= max_dist:
                    if t_best is None or t < t_best:
                        t_best = t
                        t_type = 'coin'
            # powerup
            if getattr(self, 'powerup', None) and self.powerup.active:
                pu = self.powerup
                # Generous expansion so horizontal rays at torso height can still hit
                tile = float(getattr(cfg, 'tile_size', 1.0))
                expand_x = max(tile * 0.4, cfg.player_w * 0.25)
                expand_y = max(tile * 0.9, cfg.player_h * 0.6)
                t = self._ray_intersect_aabb(
                    origin_x, origin_y, dx, dy,
                    pu.x - pu.w * 0.5 - expand_x,
                    pu.y - pu.h * 0.5 - expand_y,
                    pu.x + pu.w * 0.5 + expand_x,
                    pu.y + pu.h * 0.5 + expand_y,
                )
                if t is not None and 0.0 <= t <= max_dist:
                    if t_best is None or t < t_best:
                        t_best = t
                        t_type = 'boots'
            # raccoon
            rac = getattr(self, 'raccoon', None)
            if rac is not None:
                t = self._ray_intersect_aabb(origin_x, origin_y, dx, dy, rac.x - rac.w * 0.5, rac.y - rac.h * 0.5, rac.x + rac.w * 0.5, rac.y + rac.h * 0.5)
                if t is not None and 0.0 <= t <= max_dist:
                    if t_best is None or t < t_best:
                        t_best = t
                        t_type = 'raccoon'
            if t_best is None:
                hit_types.append(type_map['none'])
                hit_dists.append(1.0)
            else:
                hit_types.append(type_map[t_type])
                hit_dists.append(float(max(0.0, min(1.0, t_best / max_dist))))
        return hit_types, hit_dists

    @staticmethod
    def _ray_intersect_aabb(ox: float, oy: float, dx: float, dy: float, minx: float, miny: float, maxx: float, maxy: float) -> Optional[float]:
        """Ray (origin + t*dir) vs AABB intersection using slabs. Returns t or None."""
        tmin = -float('inf')
        tmax = float('inf')
        # X slabs
        if abs(dx) < 1e-9:
            if ox < minx or ox > maxx:
                return None
        else:
            tx1 = (minx - ox) / dx
            tx2 = (maxx - ox) / dx
            tmin = max(tmin, min(tx1, tx2))
            tmax = min(tmax, max(tx1, tx2))
        # Y slabs
        if abs(dy) < 1e-9:
            if oy < miny or oy > maxy:
                return None
        else:
            ty1 = (miny - oy) / dy
            ty2 = (maxy - oy) / dy
            tmin = max(tmin, min(ty1, ty2))
            tmax = min(tmax, max(ty1, ty2))
        if tmax < 0 or tmin > tmax:
            return None
        # Hit at nearest positive t
        thit = tmin if tmin >= 0 else tmax
        return thit if thit >= 0 else None

    # ------------------------ Exploration bitfield helpers -------------------
    def _mark_explored(self, x: float, y: float) -> None:
        """Mark the 8x8 sector containing (x,y) as explored in the 64-bit field."""
        grid = int(max(1, getattr(self.config, 'exploration_grid_size', 8)))
        cell_w = float(self.config.width) / float(grid)
        cell_h = float(self.config.height) / float(grid)
        cx = int(min(grid - 1, max(0, int(x / max(1e-6, cell_w)))))
        cy = int(min(grid - 1, max(0, int(y / max(1e-6, cell_h)))))
        idx = cy * grid + cx  # row-major, bottom row cy=0
        self._explore_bits |= (1 << idx)

 
