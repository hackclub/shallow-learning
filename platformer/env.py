from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


Platform = Tuple[float, float, float, float]  # (x, y, w, h) in world units, y upwards


@dataclass
class GameConfig:
    width: float = 50.0
    height: float = 10.0
    x_goal: float = 40.0

    # Simulation timing
    tickrate_hz: float = 30.0  # simulation updates per second
    episode_time_s: float = 20.0  # episode duration in seconds
    dt: float = 0.1
    gravity: float = -25.0  # y-axis points upward
    move_accel: float = 120.0
    max_speed: float = 8.0
    jump_velocity: float = 10.0
    friction: float = 8.0

    episode_length: int = 250  # derived from episode_time_s * tickrate_hz

    # Platforms: specify as list of (x, y, w, h), y upwards
    platforms: List[Platform] = field(default_factory=list)

    # Coins and rewards
    coin_radius: float = 0.25
    coin_reward: float = 1000.0
    finish_speed_bonus: float = 50.0  # bonus scaled by remaining time ratio
    finish_base_bonus: float = 5.0  # flat bonus on finish
    failure_penalty: float = 5000.0  # applied on any non-successful termination
    jump_penalty: float = 0.2  # small penalty when a jump is initiated

    # Movement control
    air_control_scale: float = 0.333  # fraction of horizontal accel allowed while airborne

    # Powerup: jump multiplier near goal
    powerup_radius: float = 0.3
    powerup_jump_multiplier: float = 2.0
    powerup_offset_x: float = 1.5  # place this far before goal line
    powerup_height: float = 1.2     # y above ground
    # Raccoon goal entity
    raccoon_radius: float = 0.45

    # Variable jump (pressure-sensitive): max time the jump can be held
    variable_jump_max_hold_s: float = 0.333
    # Cooldown after landing before another jump is allowed
    jump_cooldown_s: float = 0.10

    def __post_init__(self) -> None:
        # Derive dt and episode_length from tickrate and episode time
        if self.tickrate_hz <= 0.0:
            self.tickrate_hz = 50.0
        self.dt = 1.0 / float(self.tickrate_hz)
        self.episode_length = int(max(1, round(self.episode_time_s * self.tickrate_hz)))
        if not self.platforms:
            # Ground and a few steps
            self.platforms = [
                (-1e6, -0.5, 2e6, 0.5),  # infinite ground strip at y=0 top
                # Top-left detour platform (with coin)
                (1.0, 7.5, 2.5, 0.4),
                (5.0, 1.0, 3.0, 0.5),
                (10.0, 2.0, 3.0, 0.5),
                (16.0, 3.5, 3.0, 0.5),
                # (23.0, 2.5, 3.0, 0.5),
                (30.0, 1.5, 3.5, 0.5),
            ]


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
    Observation (np.float32):
        [x/x_goal, y/height, vx/max_speed, vy/jump_velocity, on_ground(0/1),
         coins_collected/num_coins, time_remaining_ratio]
    """

    def __init__(self, config: Optional[GameConfig] = None, seed: Optional[int] = None) -> None:
        self.config = config or GameConfig()
        self.rng = np.random.default_rng(seed)
        self.reset()

    @property
    def action_size(self) -> int:
        return 6

    @property
    def observation_size(self) -> int:
        return 7

    def reset(self) -> np.ndarray:
        self.timestep = 0
        self.x = 2.0
        self.y = 1.0
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = True
        self.prev_x = self.x
        self.done = False

        # Coins: one centered on top of each non-ground platform
        self.coins: List[Dict[str, Any]] = []
        for (px, py, pw, ph) in self.config.platforms[1:]:  # skip ground
            cx = px + pw * 0.5
            cy = py + ph + self.config.coin_radius + 0.12
            self.coins.append({"x": cx, "y": cy, "collected": False})

        # Jump powerup near goal
        self.has_jump_powerup: bool = False
        # Place at middle-bottom of screen
        mid_x = self.config.width * 0.5
        self.powerup = {
            "x": mid_x,
            "y": 0.6,  # slightly above ground to be collectible
            "collected": False,
        }

        # Raccoon goal placed near previous goal line (collision center at (x,y))
        self.raccoon = {
            "x": self.config.x_goal,
            "y": 0.6,
        }

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
        wants_left = action in (1, 4)
        wants_right = action in (2, 5)
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
        wants_jump = action in (3, 4, 5)
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
        self.vx = float(np.clip(self.vx, -cfg.max_speed, cfg.max_speed))
        self.vy += cfg.gravity * dt

        # Axis-separated integration with full-rect collisions (AABB approx of player circle)
        pradius = 0.4
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
            player_top = self.y + 2.0 * pradius
            vertical_overlap = (player_top > bottom) and (player_bottom < top)
            if not vertical_overlap:
                continue
            # moving right into left side
            if (self.x + pradius) <= left and (new_x + pradius) > left:
                new_x = left - pradius
                self.vx = 0.0
            # moving left into right side
            if (self.x - pradius) >= right and (new_x - pradius) < right:
                new_x = right + pradius
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
            player_left = new_x - pradius
            player_right = new_x + pradius
            horizontal_overlap = (player_right > left) and (player_left < right)
            if not horizontal_overlap:
                continue
            # moving up into platform bottom (head hit)
            if (self.y + 2.0 * pradius) <= bottom and (new_y + 2.0 * pradius) > bottom:
                new_y = bottom - 2.0 * pradius
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

        # If we just landed this frame, start jump cooldown
        if (not was_on_ground) and self.on_ground:
            self.jump_cooldown_left_s = cfg.jump_cooldown_s

        # End of jump when landing or starting to descend past apex
        if self.is_in_jump and (self.on_ground or self.vy <= 0.0):
            # Once descending, no further cut effects matter this jump
            self.is_in_jump = False

        # Coin collection after movement
        coin_reward_total = 0.0
        pradius = 0.4  # player render radius used in renderer
        for coin in self.coins:
            if coin["collected"]:
                continue
            dx = self.x - coin["x"]
            dy = (self.y + pradius) - coin["y"]
            dist2 = dx * dx + dy * dy
            if dist2 <= (pradius + cfg.coin_radius) ** 2:
                coin["collected"] = True
                coin_reward_total += cfg.coin_reward

        # Powerup collection
        if self.powerup and not self.powerup["collected"]:
            dx = self.x - self.powerup["x"]
            dy = (self.y + pradius) - self.powerup["y"]
            if dx * dx + dy * dy <= (pradius + cfg.powerup_radius) ** 2:
                self.powerup["collected"] = True
                self.has_jump_powerup = True

        # Terminal conditions
        self.timestep += 1
        # Success if colliding with raccoon
        pradius = 0.4  # player radius
        rx = self.raccoon["x"] if hasattr(self, "raccoon") else cfg.x_goal
        ry = self.raccoon["y"] if hasattr(self, "raccoon") else 0.6
        dxg = self.x - rx
        dyg = (self.y + pradius) - ry
        reached_goal = (dxg * dxg + dyg * dyg) <= (pradius + cfg.raccoon_radius) ** 2
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
        }
        return obs, float(reward), self.done, info

    @property
    def coins_collected_count(self) -> int:
        return sum(1 for c in self.coins if c["collected"]) if hasattr(self, "coins") else 0

    def _get_observation(self) -> np.ndarray:
        cfg = self.config
        coins_total = max(1, len(self.coins)) if hasattr(self, "coins") else 1
        time_remaining_ratio = 1.0 - (self.timestep / float(max(1, cfg.episode_length)))
        obs = np.array([
            np.clip(self.x / max(cfg.x_goal, 1e-6), 0.0, 1.0),
            np.clip(self.y / max(cfg.height, 1e-6), -1.0, 2.0),
            np.clip(self.vx / max(cfg.max_speed, 1e-6), -1.0, 1.0),
            np.clip(self.vy / max(cfg.jump_velocity, 1e-6), -2.0, 2.0),
            1.0 if self.on_ground else 0.0,
            float(self.coins_collected_count) / float(coins_total),
            np.clip(time_remaining_ratio, 0.0, 1.0),
        ], dtype=np.float32)
        return obs

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
