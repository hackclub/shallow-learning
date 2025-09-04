from __future__ import annotations

import argparse
import sys
import numpy as np
import logging
import os

from .env import PlatformerEnv, GameConfig
from .policy import MLPPolicy, MLPPolicyConfig
from .ga import GAConfig, train_ga


def cmd_simulate(args: argparse.Namespace) -> int:
    env = PlatformerEnv(GameConfig())
    policy_cfg = MLPPolicyConfig(input_size=env.observation_size, output_size=env.action_size)
    policy = MLPPolicy(policy_cfg)

    episodes = int(args.episodes)
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = policy.act(obs)
            obs, reward, done, info = env.step(action)
            ep_ret += reward
        print(f"episode={ep} return={ep_ret:.3f} x={info['x']:.2f} reached_goal={info['reached_goal']}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    # Configure logging for training session
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Pre-run cleanup of checkpoints if requested
    if args.ckpt_dir:
        try:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            removed = 0
            for f in os.listdir(args.ckpt_dir):
                p = os.path.join(args.ckpt_dir, f)
                if os.path.isfile(p) and f.endswith(".npy"):
                    os.remove(p)
                    removed += 1
            if removed:
                logging.info("Removed %d existing checkpoint(s) in %s", removed, args.ckpt_dir)
        except Exception as e:
            logging.warning("Failed to clean checkpoint dir %s: %s", args.ckpt_dir, e)

    env = PlatformerEnv(GameConfig())
    policy_cfg = MLPPolicyConfig(input_size=env.observation_size, output_size=env.action_size)
    ga_cfg = GAConfig(
        population_size=int(args.pop_size),
        generations=int(args.generations),
        elite_fraction=float(args.elite_frac),
        mutation_std_start=float(args.mutation_std_start),
        mutation_std_end=float(args.mutation_std_end),
        episodes_per_eval=int(args.eval_episodes),
        seed=int(args.seed) if args.seed is not None else None,
        checkpoint_dir=args.ckpt_dir,
        eval_batch_size=int(args.eval_batch_size) if args.eval_batch_size is not None else None,
        num_workers=int(args.num_workers) if args.num_workers is not None else None,
    )
    weights, fitness = train_ga(env, policy_cfg, ga_cfg)
    print(f"best_fitness={fitness:.3f}")
    if args.save:
        np.save(args.save, weights)
        print(f"saved_weights={args.save}")

    # Suggest render command using the checkpoint directory
    if args.ckpt_dir and os.path.isdir(args.ckpt_dir):
        print("suggested_render:\n  python -m platformer render --speed 3 --ckpt-dir " + args.ckpt_dir)
        # Auto-render: open pygame and play through the checkpoints
        try:
            from .renderer import render as do_render
            files = sorted(
                [os.path.join(args.ckpt_dir, f) for f in os.listdir(args.ckpt_dir) if f.endswith('.npy')]
            )
            if files:
                do_render(files, speed=float(args.render_speed))
        except Exception as e:
            logging.warning("Auto-render skipped: %s", e)

    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from .renderer import render as do_render
    weights = list(args.weights or [])
    if args.ckpt_dir:
        try:
            files = sorted(
                [os.path.join(args.ckpt_dir, f) for f in os.listdir(args.ckpt_dir) if f.endswith(".npy")]
            )
            weights = files
        except Exception as e:
            logging.error("Failed to list checkpoints in %s: %s", args.ckpt_dir, e)
            return 1
    do_render(weights, speed=float(args.speed))
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    # play is just renderer without policy
    from .renderer import render as do_render
    do_render([], speed=float(args.speed))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="platformer", description="Headless platformer + GA trainer")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("simulate", help="Run headless episodes with a random-initialized policy")
    ps.add_argument("--episodes", type=int, default=5)
    ps.set_defaults(func=cmd_simulate)

    pt = sub.add_parser("train", help="Train a policy via GA and optionally save weights")
    pt.add_argument("--generations", type=int, default=20)
    pt.add_argument("--pop-size", type=int, default=64)
    pt.add_argument("--elite-frac", type=float, default=0.1)
    pt.add_argument("--mutation-std-start", type=float, default=0.08, help="Initial mutation sigma")
    pt.add_argument("--mutation-std-end", type=float, default=0.01, help="Final mutation sigma")
    pt.add_argument("--eval-episodes", type=int, default=1)
    pt.add_argument("--eval-batch-size", type=int, default=None, help="Batch size for vectorized evaluation (None = full pop)")
    pt.add_argument("--num-workers", type=int, default=None, help="Number of processes for parallel evaluation")
    pt.add_argument("--seed", type=int, default=None)
    pt.add_argument("--save", type=str, default=None)
    # checkpoint and render flags
    pt.add_argument("--ckpt-dir", type=str, default=None, help="Directory to save improved checkpoints")
    pt.add_argument("--render-speed", type=float, default=3.0, help="Speed multiplier for auto-render after training")
    pt.set_defaults(func=cmd_train)

    pr = sub.add_parser("render", help="Render a trained policy (requires pygame)")
    pr.add_argument("--ckpt-dir", type=str, default=None, help="Directory containing .npy checkpoints")
    pr.add_argument("--weights", type=str, nargs='*', help="Explicit .npy weight files (overridden by --ckpt-dir)", default=[])
    pr.add_argument("--speed", type=float, default=1.0, help="Speed multiplier (steps per frame)")
    pr.set_defaults(func=cmd_render)

    pp = sub.add_parser("play", help="Play manually (requires pygame)")
    pp.add_argument("--speed", type=float, default=1.0)
    pp.set_defaults(func=cmd_play)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
