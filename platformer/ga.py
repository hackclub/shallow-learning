from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import logging
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from .env import PlatformerEnv, GameConfig
from .policy import MLPPolicy, MLPPolicyConfig


logger = logging.getLogger(__name__)


@dataclass
class GAConfig:
    population_size: int = 64
    elite_fraction: float = 0.1
    # Linear annealing of mutation std from start -> end over generations
    mutation_std_start: float = 0.08
    mutation_std_end: float = 0.01
    crossover_fraction: float = 0.5
    generations: int = 50
    episodes_per_eval: int = 1
    seed: Optional[int] = None
    # Checkpoint config
    checkpoint_dir: Optional[str] = None
    # Vectorized evaluation batch size (None = evaluate whole population at once)
    eval_batch_size: Optional[int] = None
    # Parallel evaluation workers (None or <=1 disables multiprocessing)
    num_workers: Optional[int] = None


def evaluate_policy(policy: MLPPolicy, env: PlatformerEnv, episodes: int) -> float:
    total = 0.0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = policy.act(obs)
            obs, reward, done, _ = env.step(action)
            ep_ret += reward
        total += ep_ret
    return total / float(max(1, episodes))


def crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    assert parent_a.shape == parent_b.shape
    mask = rng.random(size=parent_a.shape) < 0.5
    child = np.where(mask, parent_a, parent_b)
    return child


def mutate(flat: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    return flat + rng.normal(0.0, std, size=flat.shape)


def train_ga(env: PlatformerEnv, policy_cfg: MLPPolicyConfig, ga_cfg: GAConfig) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(ga_cfg.seed)
    logger.info(
        "Starting GA training: generations=%d, population_size=%d, eval_episodes=%d",
        ga_cfg.generations,
        ga_cfg.population_size,
        ga_cfg.episodes_per_eval,
    )
    # Initialize population
    template = MLPPolicy(policy_cfg, rng)
    num_params = template.get_flat().size
    population = rng.normal(0.0, 0.02, size=(ga_cfg.population_size, num_params))

    best_weights = population[0].copy()
    best_fitness = -np.inf

    for gen in range(ga_cfg.generations):
        # Compute linearly annealed mutation std for this generation
        if ga_cfg.generations > 1:
            alpha = gen / float(ga_cfg.generations - 1)
        else:
            alpha = 1.0
        current_mutation_std = (
            (1.0 - alpha) * ga_cfg.mutation_std_start + alpha * ga_cfg.mutation_std_end
        )
        # Evaluate entire population with a vectorized simulator (optionally batched)
        if ga_cfg.num_workers is not None and ga_cfg.num_workers > 1:
            fitness = evaluate_population_vectorized_parallel(
                policy_cfg=policy_cfg,
                env_cfg=env.config,
                population=population,
                episodes=ga_cfg.episodes_per_eval,
                batch_size=ga_cfg.eval_batch_size,
                num_workers=int(ga_cfg.num_workers),
            )
        elif ga_cfg.eval_batch_size is not None and ga_cfg.eval_batch_size > 0 and ga_cfg.eval_batch_size < ga_cfg.population_size:
            fitness = evaluate_population_vectorized_batched(
                policy_cfg=policy_cfg,
                env_cfg=env.config,
                population=population,
                episodes=ga_cfg.episodes_per_eval,
                batch_size=int(ga_cfg.eval_batch_size),
            )
        else:
            fitness = evaluate_population_vectorized(
                policy_cfg=policy_cfg,
                env_cfg=env.config,
                population=population,
                episodes=ga_cfg.episodes_per_eval,
            )
        # Track best
        idx = int(np.argmax(fitness))
        improved = fitness[idx] > best_fitness
        if improved:
            best_fitness = float(fitness[idx])
            best_weights = population[idx].copy()

        # Progress logging
        mean_f = float(np.mean(fitness))
        max_f = float(np.max(fitness))
        logger.info(
            "gen %d/%d - max=%.3f mean=%.3f sigma=%.4f",
            gen + 1,
            ga_cfg.generations,
            max_f,
            mean_f,
            current_mutation_std,
        )

        # Save checkpoint on any improvement
        if improved and ga_cfg.checkpoint_dir is not None:
            os.makedirs(ga_cfg.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(
                ga_cfg.checkpoint_dir, f"best_gen{gen + 1}_fit{best_fitness:.3f}.npy"
            )
            np.save(ckpt_path, best_weights)
            logger.info("Saved improved checkpoint: %s", ckpt_path)

        # Selection
        elite_count = max(1, int(ga_cfg.elite_fraction * ga_cfg.population_size))
        elite_idx = np.argsort(fitness)[-elite_count:]
        elites = population[elite_idx]

        # New population from elites via crossover + mutation
        new_population = np.zeros_like(population)
        # carry elites
        new_population[:elite_count] = elites
        # fill rest
        for i in range(elite_count, ga_cfg.population_size):
            pa, pb = rng.choice(elite_count, size=2, replace=True)
            child = crossover(elites[pa], elites[pb], rng)
            child = mutate(child, current_mutation_std, rng)
            new_population[i] = child
        population = new_population

    logger.info("GA training complete. Best fitness=%.3f", best_fitness)
    return best_weights, best_fitness


def evaluate_population_vectorized(
    policy_cfg: MLPPolicyConfig,
    env_cfg: GameConfig,
    population: np.ndarray,
    episodes: int = 1,
) -> np.ndarray:
    """Vectorized evaluation of a whole population.

    - population: (P, num_params)
    - returns: fitness array (P,)
    """
    rng = np.random.default_rng(0)
    pop_size = population.shape[0]

    # Build per-layer weight tensors for the whole population
    # shapes like [(out, in+1), ...] from a template
    template_policy = MLPPolicy(policy_cfg, rng)
    layer_shapes = template_policy.shapes

    def unpack_population_layers(pop_flat: np.ndarray) -> list[np.ndarray]:
        """Unpack (P, num_params) into list of (P, out, in_plus_one)."""
        layers: list[np.ndarray] = []
        offset = 0
        for (out_s, in_b1) in layer_shapes:
            size = out_s * in_b1
            layer = pop_flat[:, offset : offset + size].reshape(pop_size, out_s, in_b1)
            layers.append(layer)
            offset += size
        return layers

    pop_layers = unpack_population_layers(population)

    # Static world data
    platforms = np.array(env_cfg.platforms, dtype=np.float32)  # (K, 4)
    K = platforms.shape[0]
    # coins from non-ground platforms
    coins_src = platforms[1:] if K > 1 else platforms[:0]
    if coins_src.shape[0] > 0:
        coin_x = coins_src[:, 0] + coins_src[:, 2] * 0.5
        coin_y = coins_src[:, 1] + coins_src[:, 3] + env_cfg.coin_radius + 0.05
    else:
        coin_x = np.zeros((0,), dtype=np.float32)
        coin_y = np.zeros((0,), dtype=np.float32)
    num_coins = coin_x.shape[0]

    # Powerup position (match env.reset placement)
    mid_x = env_cfg.width * 0.5
    power_x = mid_x
    power_y = 0.6

    pradius = 0.4
    coin_thresh2 = (pradius + env_cfg.coin_radius) ** 2
    pwr_thresh2 = (pradius + env_cfg.powerup_radius) ** 2

    # Helpers for batched policy forward with per-individual weights
    def policy_forward(obs: np.ndarray) -> np.ndarray:
        # obs: (P, in)
        x = obs.astype(np.float32)
        for li, w in enumerate(pop_layers):
            # w: (P, out, in+1)
            bias = np.ones((pop_size, 1), dtype=x.dtype)
            x_bias = np.concatenate([x, bias], axis=1)  # (P, in+1)
            # y[n, o] = sum_j w[n, o, j] * x_bias[n, j]
            x = np.einsum('noj,nj->no', w, x_bias)
            if li < len(pop_layers) - 1:
                x = np.tanh(x)
        return x  # (P, out)

    # Simulate multiple episodes and average
    fitness_accum = np.zeros((pop_size,), dtype=np.float32)
    for _ in range(max(1, episodes)):
        # State arrays per individual
        x = np.full((pop_size,), 2.0, dtype=np.float32)
        y = np.full((pop_size,), 1.0, dtype=np.float32)
        vx = np.zeros((pop_size,), dtype=np.float32)
        vy = np.zeros((pop_size,), dtype=np.float32)
        on_ground = np.ones((pop_size,), dtype=bool)
        prev_x = x.copy()
        done = np.zeros((pop_size,), dtype=bool)
        timestep = np.zeros((pop_size,), dtype=np.int32)

        # Jump and powerup state
        has_pwr = np.zeros((pop_size,), dtype=bool)
        is_in_jump = np.zeros((pop_size,), dtype=bool)
        jump_hold = np.zeros((pop_size,), dtype=np.float32)
        jump_cut_applied = np.zeros((pop_size,), dtype=bool)

        # Coins collected flags (P, C)
        coins_col = np.zeros((pop_size, num_coins), dtype=bool)

        returns = np.zeros((pop_size,), dtype=np.float32)

        # step loop
        max_steps = int(env_cfg.episode_length)
        dt = float(env_cfg.dt)
        for _t in range(max_steps):
            if done.all():
                break

            # Build observation (P, obs_size)
            coins_frac = (coins_col.sum(axis=1) / float(max(1, num_coins))).astype(np.float32)
            time_ratio = 1.0 - (timestep.astype(np.float32) / float(max(1, env_cfg.episode_length)))
            obs = np.stack([
                np.clip(x / max(env_cfg.x_goal, 1e-6), 0.0, 1.0),
                np.clip(y / max(env_cfg.height, 1e-6), -1.0, 2.0),
                np.clip(vx / max(env_cfg.max_speed, 1e-6), -1.0, 1.0),
                np.clip(vy / max(env_cfg.jump_velocity, 1e-6), -2.0, 2.0),
                on_ground.astype(np.float32),
                coins_frac,
                np.clip(time_ratio, 0.0, 1.0),
            ], axis=1)

            # Policy action per individual
            logits = policy_forward(obs)
            action = np.argmax(logits, axis=1)

            alive = ~done

            # Horizontal control
            wants_left = (action == 1) | (action == 4)
            wants_right = (action == 2) | (action == 5)
            input_accel = env_cfg.move_accel * np.where(on_ground, 1.0, env_cfg.air_control_scale)
            ax = np.zeros_like(vx)
            ax = np.where(wants_left & ~wants_right, ax - input_accel, ax)
            ax = np.where(wants_right & ~wants_left, ax + input_accel, ax)
            # friction
            no_input = (~wants_left & ~wants_right) | (wants_left & wants_right)
            ax = np.where(no_input, ax - env_cfg.friction * vx, ax)
            ax = np.where(~no_input, ax - 0.2 * env_cfg.friction * vx, ax)

            # Jump
            wants_jump = (action == 3) | (action == 4) | (action == 5)
            jump_initiated = wants_jump & on_ground & (~is_in_jump)
            mult = np.where(has_pwr, env_cfg.powerup_jump_multiplier, 1.0)
            vy = np.where(jump_initiated, env_cfg.jump_velocity * mult, vy)
            on_ground = np.where(jump_initiated, False, on_ground)
            is_in_jump = np.where(jump_initiated, True, is_in_jump)
            jump_hold = np.where(jump_initiated, 0.0, jump_hold)
            jump_cut_applied = np.where(jump_initiated, False, jump_cut_applied)

            # Track hold time while in jump
            can_hold = is_in_jump & wants_jump & (jump_hold < env_cfg.variable_jump_max_hold_s)
            jump_hold = np.where(can_hold, jump_hold + dt, jump_hold)

            # Early release cut
            cut_mask = (
                is_in_jump
                & (~wants_jump)
                & (~jump_cut_applied)
                & (jump_hold < env_cfg.variable_jump_max_hold_s)
                & (vy > 0.0)
            )
            hold_frac = np.clip(jump_hold / max(1e-6, env_cfg.variable_jump_max_hold_s), 0.0, 1.0)
            vy = np.where(cut_mask, vy * hold_frac, vy)
            jump_cut_applied = np.where(cut_mask, True, jump_cut_applied)

            # Integrate velocity
            vx = vx + ax * dt
            vx = np.clip(vx, -env_cfg.max_speed, env_cfg.max_speed)
            vy = vy + env_cfg.gravity * dt

            # Integrate position
            new_x = x + vx * dt
            new_y = y + vy * dt

            # Collisions (descending only)
            desc = new_y <= y
            if platforms.shape[0] > 0:
                # For each platform compute mask of crossing
                px = platforms[:, 0][None, :]
                py = platforms[:, 1][None, :]
                pw = platforms[:, 2][None, :]
                ph = platforms[:, 3][None, :]
                top_y = (py + ph)  # (1, K)

                within_x = (new_x[:, None] >= px) & (new_x[:, None] <= (px + pw))
                crosses_top = (y[:, None] >= top_y) & (new_y[:, None] <= top_y)
                mask = (desc[:, None]) & within_x & crosses_top
                # pick highest top_y collided per env
                cand_top = np.where(mask, top_y, -1e9)
                max_top = cand_top.max(axis=1)  # (P,)
                collided = max_top > -1e9 / 2
            else:
                collided = np.zeros((pop_size,), dtype=bool)
                max_top = np.zeros((pop_size,), dtype=np.float32)

            # Apply collision resolution
            vy = np.where(collided, 0.0, vy)
            new_y = np.where(collided, max_top, new_y)
            on_ground = np.where(collided, True, on_ground)
            # If no collision, update on_ground based on vy
            on_ground = np.where(~collided, np.where(vy != 0.0, False, on_ground), on_ground)

            # Commit position
            prev_x_alive = prev_x.copy()
            prev_x = x
            x = new_x
            y = new_y

            # End of jump
            end_jump = is_in_jump & (on_ground | (vy <= 0.0))
            is_in_jump = np.where(end_jump, False, is_in_jump)

            # Coin collection
            coin_reward_add = np.zeros((pop_size,), dtype=np.float32)
            if num_coins > 0:
                dx = x[:, None] - coin_x[None, :]
                dy = (y[:, None] + pradius) - coin_y[None, :]
                dist2 = dx * dx + dy * dy
                can_collect = dist2 <= coin_thresh2
                newly = (~coins_col) & can_collect
                # accumulate reward for new pickups
                coin_reward_add = newly.sum(axis=1).astype(np.float32) * float(env_cfg.coin_reward)
                coins_col = coins_col | newly

            # Powerup collection
            dxp = x - power_x
            dyp = (y + pradius) - power_y
            pwr_pick = (dxp * dxp + dyp * dyp) <= pwr_thresh2
            has_pwr = has_pwr | pwr_pick

            # Terminal conditions
            timestep = timestep + 1
            reached_goal = x >= env_cfg.x_goal
            fell_out = y < -10.0
            time_up = timestep >= env_cfg.episode_length
            left_out = x < 0.0
            succeed = reached_goal
            became_done = (~done) & (succeed | fell_out | time_up | left_out)
            done = done | became_done

            # Rewards (only for alive individuals)
            progress = np.maximum(0.0, x - prev_x)
            step_rew = progress + coin_reward_add
            step_rew = np.where(jump_initiated, step_rew - env_cfg.jump_penalty, step_rew)
            time_ratio_now = 1.0 - (timestep.astype(np.float32) / float(max(1, env_cfg.episode_length)))
            finish_bonus = env_cfg.finish_base_bonus + env_cfg.finish_speed_bonus * np.clip(time_ratio_now, 0.0, 1.0)
            step_rew = np.where(succeed & (~(timestep == 0)), step_rew + finish_bonus, step_rew)
            # Failure penalty applied when episode ends and did not succeed
            step_rew = np.where(became_done & (~succeed), step_rew - env_cfg.failure_penalty, step_rew)

            # Mask out those already done before this step
            step_rew = np.where(alive, step_rew, 0.0)
            returns += step_rew

        fitness_accum += returns

    return fitness_accum / float(max(1, episodes))


def evaluate_population_vectorized_batched(
    policy_cfg: MLPPolicyConfig,
    env_cfg: GameConfig,
    population: np.ndarray,
    episodes: int = 1,
    batch_size: int = 128,
) -> np.ndarray:
    """Evaluate population fitness in batches to limit memory/compute spikes."""
    P = population.shape[0]
    fitness = np.zeros((P,), dtype=np.float32)
    bs = max(1, int(batch_size))
    for start in range(0, P, bs):
        end = min(P, start + bs)
        fitness[start:end] = evaluate_population_vectorized(
            policy_cfg=policy_cfg,
            env_cfg=env_cfg,
            population=population[start:end],
            episodes=episodes,
        )
    return fitness


def _evaluate_population_chunk(args: tuple[MLPPolicyConfig, GameConfig, np.ndarray, int, Optional[int]]) -> np.ndarray:
    policy_cfg, env_cfg, pop_chunk, episodes, batch_size = args
    if batch_size is not None and batch_size > 0 and batch_size < pop_chunk.shape[0]:
        return evaluate_population_vectorized_batched(policy_cfg, env_cfg, pop_chunk, episodes, batch_size)
    else:
        return evaluate_population_vectorized(policy_cfg, env_cfg, pop_chunk, episodes)


def evaluate_population_vectorized_parallel(
    policy_cfg: MLPPolicyConfig,
    env_cfg: GameConfig,
    population: np.ndarray,
    episodes: int = 1,
    batch_size: Optional[int] = None,
    num_workers: int = 2,
) -> np.ndarray:
    """Evaluate population fitness using multiple processes.

    Splits population into roughly equal chunks per worker and evaluates each chunk
    using the vectorized evaluator in a separate process. Within each process, if
    batch_size is provided, the chunk is further evaluated in mini-batches.
    """
    P = population.shape[0]
    if num_workers <= 1 or P == 0:
        return _evaluate_population_chunk((policy_cfg, env_cfg, population, episodes, batch_size))

    # Determine splits
    n = int(max(1, num_workers))
    # Avoid creating more workers than individuals
    n = min(n, P)
    base = P // n
    rem = P % n
    starts = []
    s = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        starts.append((s, s + size))
        s += size

    fitness = np.zeros((P,), dtype=np.float32)
    with ProcessPoolExecutor(max_workers=n) as ex:
        futs = []
        for (s, e) in starts:
            pop_chunk = population[s:e]
            fut = ex.submit(_evaluate_population_chunk, (policy_cfg, env_cfg, pop_chunk, episodes, batch_size))
            futs.append((s, e, fut))
        for (s, e, fut) in futs:
            res = fut.result()
            fitness[s:e] = res
    return fitness
