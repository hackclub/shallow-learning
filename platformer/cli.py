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
                if not (f.endswith('.npy') or f.endswith('.json')):
                    continue
                p = os.path.join(args.ckpt_dir, f)
                if os.path.isfile(p) or os.path.islink(p):
                    try:
                        os.remove(p)
                        removed += 1
                    except Exception as ex:
                        logging.warning("Failed to remove %s: %s", p, ex)
            if removed:
                logging.info("Removed %d checkpoint file(s) in %s", removed, args.ckpt_dir)
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
    import re
    import time
    weights = list(args.weights or [])
    if args.ckpt_dir:
        try:
            files_all = [os.path.join(args.ckpt_dir, f) for f in os.listdir(args.ckpt_dir) if f.endswith(".npy")]
            # Prefer GA checkpoints with a generation tag; keep others after. Then let renderer sort by gen/fit desc.
            files_gen = [p for p in files_all if re.search(r"gen\d+", os.path.basename(p))]
            files_other = [p for p in files_all if p not in files_gen]
            files = files_gen + files_other
            # Optional selection by generation tag
            if getattr(args, 'gen', None) is not None:
                want_gen = int(args.gen)
                def gen_val(p: str) -> int:
                    m = re.search(r"gen(\d+)", os.path.basename(p))
                    return int(m.group(1)) if m else -1
                files = [p for p in files if gen_val(p) == want_gen]
            if bool(getattr(args, 'best_only', False)) and files:
                # Pick the file with the highest parsed fit value; fallback to last if unparsable
                def fit_val(p: str) -> float:
                    m = re.search(r"fit([0-9]+(?:\.[0-9]+)?)", os.path.basename(p))
                    return float(m.group(1)) if m else float('-inf')
                best = max(files, key=fit_val)
                if fit_val(best) == float('-inf'):
                    weights = [files[-1]]
                else:
                    weights = [best]
            else:
                weights = files
        except Exception as e:
            logging.error("Failed to list checkpoints in %s: %s", args.ckpt_dir, e)
            return 1
    # Optional watch mode: repeatedly play checkpoints when directory updates
    if getattr(args, 'watch', False) and args.ckpt_dir:
        print(f"[watch] Watching {args.ckpt_dir} for new checkpoints... (Ctrl+C to quit)")
        # First try event-driven watching; fall back to polling if unavailable
        try:
            import threading
            from watchdog.observers import Observer  # type: ignore
            from watchdog.events import FileSystemEventHandler  # type: ignore

            class _CkptHandler(FileSystemEventHandler):
                def __init__(self, directory: str, notify_evt: 'threading.Event') -> None:
                    super().__init__()
                    self._dir = directory
                    self._notify = notify_evt
                def _is_ckpt(self, path: str) -> bool:
                    try:
                        if os.path.dirname(path) != self._dir:
                            return False
                        return path.endswith('.npy') or path.endswith('.json')
                    except Exception:
                        return False
                def on_created(self, event):  # type: ignore[override]
                    if getattr(event, 'is_directory', False):
                        return
                    if self._is_ckpt(event.src_path):
                        print(f"[watch] created: {os.path.basename(event.src_path)}")
                        self._notify.set()
                def on_moved(self, event):  # type: ignore[override]
                    if getattr(event, 'is_directory', False):
                        return
                    if self._is_ckpt(getattr(event, 'dest_path', '')) or self._is_ckpt(getattr(event, 'src_path', '')):
                        print(
                            f"[watch] moved: src={os.path.basename(getattr(event, 'src_path', ''))} -> dst={os.path.basename(getattr(event, 'dest_path', ''))}"
                        )
                        self._notify.set()
                def on_modified(self, event):  # type: ignore[override]
                    if getattr(event, 'is_directory', False):
                        return
                    if self._is_ckpt(event.src_path):
                        print(f"[watch] modified: {os.path.basename(event.src_path)}")
                        self._notify.set()

            changed = threading.Event()
            observer = Observer()
            handler = _CkptHandler(os.path.abspath(args.ckpt_dir), changed)
            observer.schedule(handler, args.ckpt_dir, recursive=False)
            observer.start()
            try:
                last_snapshot: list[str] = []
                last_played: str | None = None
                # Initial playback of current latest, if any
                files_all = [os.path.join(args.ckpt_dir, f) for f in os.listdir(args.ckpt_dir) if f.endswith('.npy')]
                files_all.sort()
                if files_all:
                    newest0 = os.path.basename(files_all[-1])
                    print(f"[watch] initial snapshot: {len(files_all)} files; newest={newest0}")
                    try:
                        changed.clear()
                        do_render(
                            files_all,
                            speed=float(args.speed),
                            show_hitboxes=bool(args.hitboxes),
                            fullscreen=bool(args.fullscreen),
                            log_rays=bool(args.log_rays),
                            seed=args.seed,
                            oneshot=False,
                            contrail=bool(getattr(args, 'contrail', False)),
                            contrail_alpha=float(getattr(args, 'contrail_alpha', 0.01)),
                            contrail_mod=int(getattr(args, 'contrail_mod', 1)),
                            start_from_latest=True,
                            allow_navigation=True,
                            reload_event=changed,
                        )
                    except BaseException as e:
                        print(f"[watch] render aborted: {type(e).__name__}: {e}")
                        observer.stop()
                        raise
                    last_snapshot = list(files_all)
                    last_played = files_all[-1]
                while True:
                    # Wait for a change or periodic tick
                    changed.wait(timeout=1.0)
                    if not changed.is_set():
                        continue
                    changed.clear()
                    # Debounce write bursts a bit; also if another event arrives during debounce, keep it set
                    time.sleep(0.4)
                    # Keep 'changed' set so render() can detect and return
                    changed.set()
                    files_all = [os.path.join(args.ckpt_dir, f) for f in os.listdir(args.ckpt_dir) if f.endswith('.npy')]
                    files_all.sort()
                    newest = files_all[-1] if files_all else None
                    if files_all:
                        print(f"[watch] change observed: {len(files_all)} files; newest={os.path.basename(newest) if newest else 'None'}")
                    if files_all and (files_all != last_snapshot or newest != last_played):
                        print("[watch] reloading renderer with updated checkpoint set")
                        try:
                            changed.clear()
                            do_render(
                                files_all,
                                speed=float(args.speed),
                                show_hitboxes=bool(args.hitboxes),
                                fullscreen=bool(args.fullscreen),
                                log_rays=bool(args.log_rays),
                                seed=args.seed,
                                oneshot=False,
                                contrail=bool(getattr(args, 'contrail', False)),
                                contrail_alpha=float(getattr(args, 'contrail_alpha', 0.01)),
                                contrail_mod=int(getattr(args, 'contrail_mod', 1)),
                                start_from_latest=True,
                                allow_navigation=True,
                                reload_event=changed,
                            )
                        except BaseException as e:
                            print(f"[watch] render aborted: {type(e).__name__}: {e}")
                            observer.stop()
                            raise
                        last_snapshot = list(files_all)
                        last_played = newest
                    else:
                        print("[watch] no effective change in snapshot; skipping reload")
            except KeyboardInterrupt:
                print("Stopped watching.")
                observer.stop()
            finally:
                observer.join()
            return 0
        except Exception:
            print("Error: --watch requires the 'watchdog' package. Install with: pip install watchdog", file=sys.stderr)
            return 1
    else:
        do_render(
            weights,
            speed=float(args.speed),
            show_hitboxes=bool(args.hitboxes),
            fullscreen=bool(args.fullscreen),
            log_rays=bool(args.log_rays),
            seed=args.seed,
            contrail=bool(getattr(args, 'contrail', False)),
            contrail_alpha=float(getattr(args, 'contrail_alpha', 0.01)),
            contrail_mod=int(getattr(args, 'contrail_mod', 1)),
        )
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    # play is just renderer without policy
    from .renderer import render as do_render
    do_render([], speed=float(args.speed), show_hitboxes=bool(args.hitboxes), show_intro=True, fullscreen=bool(args.fullscreen), log_rays=bool(args.log_rays))
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
    pr.add_argument("--seed", type=int, default=None, help="Force a specific env seed for all files")
    pr.add_argument("--gen", type=int, default=None, help="When used with --ckpt-dir, render only checkpoints with this generation number")
    pr.add_argument("--best-only", action='store_true', help="When used with --ckpt-dir, render only the highest-fit checkpoint")
    pr.add_argument("--watch", action='store_true', help="Watch ckpt dir and auto-play the latest checkpoint when new files appear")
    pr.add_argument("--contrail", action='store_true', help="Do not clear screen each frame; draw character at low alpha to leave a trail")
    pr.add_argument("--contrail-alpha", type=float, default=0.01, help="Alpha (0..1) for contrail ghost (default 0.01)")
    pr.add_argument("--contrail-mod", type=int, default=1, help="Draw character ghost every N frames in contrail mode (default 1)")
    pr.add_argument("--hitboxes", action='store_true', help="Overlay collision hitboxes")
    pr.add_argument("--fullscreen", action='store_true', help="Open the window in fullscreen (toggle with F11)")
    pr.add_argument("--log-rays", action='store_true', help="Print raycast hits/distances during playback")
    pr.set_defaults(func=cmd_render)

    pp = sub.add_parser("play", help="Play manually (requires pygame)")
    pp.add_argument("--speed", type=float, default=1.0)
    pp.add_argument("--hitboxes", action='store_true', help="Overlay collision hitboxes")
    pp.add_argument("--fullscreen", action='store_true', help="Open the window in fullscreen (toggle with F11)")
    pp.add_argument("--log-rays", action='store_true', help="Print raycast hits/distances during play")
    pp.set_defaults(func=cmd_play)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
