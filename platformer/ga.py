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
        # Evaluate population using the single-source-of-truth env (optionally in parallel)
        fitness = evaluate_population_env(
            policy_cfg=policy_cfg,
            env_cfg=env.config,
            population=population,
            episodes=ga_cfg.episodes_per_eval,
            num_workers=int(ga_cfg.num_workers) if ga_cfg.num_workers is not None else 1,
            base_seed=ga_cfg.seed,
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
            # Compute best single-episode return among the evaluation episodes (deterministic ep seeds when GA seed set)
            episodes = max(1, int(ga_cfg.episodes_per_eval))
            best_single = -np.inf
            best_epseed: Optional[int] = None
            for ep_idx in range(episodes):
                ep_seed = int(ga_cfg.seed + ep_idx) if ga_cfg.seed is not None else None
                tmp_env = PlatformerEnv(env.config, seed=ep_seed)
                tmp_policy = MLPPolicy(policy_cfg)
                tmp_policy.set_flat(best_weights)
                single = float(evaluate_policy(tmp_policy, tmp_env, episodes=1))
                if single > best_single:
                    best_single = single
                    best_epseed = ep_seed

            os.makedirs(ga_cfg.checkpoint_dir, exist_ok=True)
            seed_tag = ga_cfg.seed if ga_cfg.seed is not None else 'none'
            epseed_tag = best_epseed if best_epseed is not None else 'none'
            ckpt_path = os.path.join(
                ga_cfg.checkpoint_dir,
                f"best_gen{gen + 1}_seed{seed_tag}_fit{best_fitness:.3f}_max{best_single:.3f}_epseed{epseed_tag}.npy",
            )
            np.save(ckpt_path, best_weights)
            logger.info(
                "Saved improved checkpoint: %s (best single=%.3f)",
                ckpt_path,
                best_single,
            )
            # Write deterministic metadata as JSON sidecar for exact replay
            try:
                import json
                meta = {
                    "gen": int(gen + 1),
                    "seed": (int(ga_cfg.seed) if ga_cfg.seed is not None else None),
                    "episodes_per_eval": int(episodes),
                    "fit": float(best_fitness),
                    "best_single": float(best_single),
                    "epseed": (int(best_epseed) if best_epseed is not None else None),
                }
                with open(ckpt_path + ".json", "w") as f:
                    json.dump(meta, f, separators=(",", ":"))
            except Exception as e:
                logger.warning("Failed to write checkpoint metadata JSON: %s", e)

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


def _eval_one_policy(args: tuple[MLPPolicyConfig, GameConfig, np.ndarray, int, int | None]) -> float:
    policy_cfg, env_cfg, flat, episodes, base_seed = args
    policy = MLPPolicy(policy_cfg)
    policy.set_flat(flat)
    total = 0.0
    for ep_idx in range(max(1, episodes)):
        seed = None if base_seed is None else int(base_seed) + ep_idx
        env = PlatformerEnv(env_cfg, seed=seed)
        total += evaluate_policy(policy, env, episodes=1)
    return total / float(max(1, episodes))


def evaluate_population_env(
    policy_cfg: MLPPolicyConfig,
    env_cfg: GameConfig,
    population: np.ndarray,
    episodes: int = 1,
    num_workers: int = 1,
    base_seed: int | None = None,
) -> np.ndarray:
    """Evaluate population using PlatformerEnv as the single source of truth.

    Optionally parallelizes across processes; each worker runs its own env.
    """
    P = population.shape[0]
    if P == 0:
        return np.zeros((0,), dtype=np.float32)

    if num_workers is not None and num_workers > 1:
        n = int(max(1, num_workers))
        with ProcessPoolExecutor(max_workers=n) as ex:
            it = ex.map(
                _eval_one_policy,
                [(policy_cfg, env_cfg, population[i], episodes, base_seed) for i in range(P)],
            )
            return np.array(list(it), dtype=np.float32)
    else:
        fitness = np.zeros((P,), dtype=np.float32)
        policy = MLPPolicy(policy_cfg)
        for i in range(P):
            policy.set_flat(population[i])
            total = 0.0
            for ep_idx in range(max(1, episodes)):
                seed = None if base_seed is None else int(base_seed) + ep_idx
                env = PlatformerEnv(env_cfg, seed=seed)
                total += evaluate_policy(policy, env, episodes=1)
            fitness[i] = total / float(max(1, episodes))
        return fitness
