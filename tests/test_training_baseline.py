from __future__ import annotations

import os
import glob

from platformer.env import PlatformerEnv, GameConfig
from platformer.policy import MLPPolicyConfig
from platformer.ga import GAConfig, train_ga


def test_training_saves_at_least_one_checkpoint(tmp_path):
	ckpt_dir = tmp_path / "ckpts"
	os.makedirs(ckpt_dir, exist_ok=True)

	# Faster, simpler env for unit tests
	cfg = GameConfig()
	cfg.exploration_enabled = False
	cfg.raycast_enabled = False
	cfg.episode_time_s = 2.0
	cfg.tickrate_hz = 10.0
	cfg.__post_init__()
	env = PlatformerEnv(cfg)
	policy_cfg = MLPPolicyConfig(input_size=env.observation_size, output_size=env.action_size)
	ga_cfg = GAConfig(
		population_size=12,
		generations=3,
		elite_fraction=0.15,
		mutation_std_start=0.20,
		mutation_std_end=0.15,
		episodes_per_eval=1,
		seed=123,
		checkpoint_dir=str(ckpt_dir),
		num_workers=1,
	)
	_, _ = train_ga(env, policy_cfg, ga_cfg)
	ckpts = sorted(glob.glob(os.path.join(str(ckpt_dir), "*.npy")))
	assert len(ckpts) >= 1


def test_training_progress_without_exploration_has_multiple_improvements(tmp_path):
	ckpt_dir = tmp_path / "ckpts_noexplore"
	os.makedirs(ckpt_dir, exist_ok=True)

	cfg = GameConfig()
	cfg.exploration_enabled = False
	cfg.raycast_enabled = False
	cfg.episode_time_s = 2.0
	cfg.tickrate_hz = 10.0
	cfg.__post_init__()
	env = PlatformerEnv(cfg)
	policy_cfg = MLPPolicyConfig(input_size=env.observation_size, output_size=env.action_size)
	ga_cfg = GAConfig(
		population_size=16,
		generations=5,
		elite_fraction=0.15,
		mutation_std_start=0.25,
		mutation_std_end=0.15,
		episodes_per_eval=1,
		seed=123,
		checkpoint_dir=str(ckpt_dir),
		num_workers=1,
	)
	_, _ = train_ga(env, policy_cfg, ga_cfg)
	ckpts = sorted(glob.glob(os.path.join(str(ckpt_dir), "*.npy")))
	assert len(ckpts) >= 2
