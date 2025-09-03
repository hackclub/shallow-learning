# Platformer: Headless Arcade Environment + Optional Renderer + GA Trainer

A minimal Python scaffold for a 2D side-scrolling platformer environment that can run headless for fast simulation, with an optional Pygame renderer for visualization, and a simple genetic algorithm (GA) trainer to evolve a neural network policy.

## Features
- Headless physics simulation suitable for RL/EA training
- Optional Pygame renderer (only required for visualization)
- Simple NumPy MLP policy
- GA trainer with checkpoint-on-improvement
- CLI via `python -m platformer`

## Objective (default)
- Reach the goal before the timer. Collecting coins increases reward; finishing faster also increases reward.
- Rewards:
  - +progress for moving rightward
  - +coin_reward each time a coin is collected
  - +finish_base_bonus + finish_bonus * (coins_collected/total_coins) + finish_speed_bonus * time_remaining_ratio when reaching the goal (even without all coins)
  - -1.0 if the player falls out of the world
- No survival bonus.
- Episode ends when the player reaches the goal, falls, or the timer elapses.

## Powerups
- Jump x2 near the goal: a blue orb placed shortly before the finish line; collecting it doubles jump height for the remainder of the episode.

## Installation
```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# Pygame is optional, install if you want visualization
pip install pygame
```

## Usage
- Simulate episodes headless (no rendering):
```bash
python -m platformer simulate --episodes 5
```

- Train a policy with a genetic algorithm (GA) and save checkpoints on improvement:
```bash
python -m platformer train --generations 50 --pop-size 64 --ckpt-dir checkpoints --save best.npy
```

- Render all checkpoints in a directory (requires pygame). Use `--speed` to render faster by simulating multiple steps per frame:
```bash
python -m platformer render --speed 3 --ckpt-dir checkpoints
```

- Play manually with arrow keys (left/right) and space (jump):
```bash
python -m platformer play --speed 1
```

- After training, a suggested command will be printed if `--ckpt-dir` was used, showing how to render the saved checkpoints with `--speed`.

## Notes
- The environment uses simple kinematics and AABB collisions for a ground and a few platforms, each with a coin on top (except the ground).
- Observations include position, velocity, on-ground flag, fraction of coins collected, and time remaining ratio.
- Pygame is not a strict dependency. Install it only if you want to visualize or play.
