from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import logging
import os
import numpy as np

from .env import PlatformerEnv
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
        # Evaluate
        fitness = np.zeros(ga_cfg.population_size, dtype=np.float32)
        for i in range(ga_cfg.population_size):
            template.set_flat(population[i])
            fitness[i] = evaluate_policy(template, env, ga_cfg.episodes_per_eval)
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
