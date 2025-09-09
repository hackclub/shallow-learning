from __future__ import annotations

import os
import glob
import numpy as np

from platformer.env import PlatformerEnv, GameConfig
from platformer.policy import MLPPolicy, MLPPolicyConfig
from platformer.ga import GAConfig, train_ga


def test_epseed_replays_best_single_episode(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    env = PlatformerEnv(GameConfig())
    policy_cfg = MLPPolicyConfig(input_size=env.observation_size, output_size=env.action_size)
    ga_cfg = GAConfig(
        population_size=16,
        generations=3,
        elite_fraction=0.2,
        mutation_std_start=0.15,
        mutation_std_end=0.10,
        episodes_per_eval=2,
        seed=42,
        checkpoint_dir=str(ckpt_dir),
        num_workers=1,
    )

    _w, _b = train_ga(env, policy_cfg, ga_cfg)
    ckpts = sorted(glob.glob(os.path.join(str(ckpt_dir), "*.npy")))
    if not ckpts:
        return
    # Use the last (best) checkpoint
    path = ckpts[-1]
    base = os.path.basename(path)

    # Parse epseed and max from filename
    import re
    m_ep = re.search(r"epseed(\d+)", base)
    # Allow integer formatting for max (no decimals) as produced by baseline env
    # Support negative values too
    m_max = re.search(r"max(-?[0-9]+(?:\.[0-9]+)?)", base)
    if m_max is None:
        m_max = re.search(r"max(-?\d+)$", os.path.splitext(base)[0])
    assert m_ep is not None and m_max is not None
    epseed = int(m_ep.group(1))
    max_logged = float(m_max.group(1))

    # Load weights and re-score a single episode with epseed; expect exact match
    flat = np.load(path)
    env2 = PlatformerEnv(GameConfig(), seed=epseed)
    policy = MLPPolicy(policy_cfg)
    policy.set_flat(flat)

    # Single episode return should equal logged max
    obs = env2.reset()
    done = False
    total = 0.0
    while not done:
        action = policy.act(obs)
        obs, r, done, _ = env2.step(action)
        total += float(r)

    assert abs(total - max_logged) < 1e-3

