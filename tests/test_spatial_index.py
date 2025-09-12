import math

from platformer.env import PlatformerEnv, GameConfig


def test_coin_query_deduplicates_across_cells() -> None:
    # Use default ASCII map; coins are placed at (c+0.5, r+0.5) with size 1x1,
    # which straddles tile cell boundaries and inserts into multiple grid cells.
    cfg = GameConfig()
    env = PlatformerEnv(cfg)
    env.reset()  # builds spatial grid

    # Pick a point near the middle of the map; query a small box that should
    # intersect some coins (the exact count isn't important as long as it's >0)
    # Build a query box around the player position that spans multiple cells.
    x = env.x
    y = env.y
    half = 1.0
    coins = list(env._query_coins_aabb(x - half, y - half, x + half, y + half))

    # If there are no coins in range, widen the box a bit to ensure coverage
    if not coins:
        coins = list(env._query_coins_aabb(x - 2.0, y - 2.0, x + 2.0, y + 2.0))

    # Should not raise and should not contain duplicates of the same object
    ids = [id(c) for c in coins]
    assert len(ids) == len(set(ids))


def test_platform_query_returns_some_entries() -> None:
    cfg = GameConfig()
    env = PlatformerEnv(cfg)
    env.reset()

    # Query around the player's current AABB and ensure we get at least one platform
    half_w = cfg.player_w * 0.5
    plats = list(env._query_platforms_aabb(env.x - half_w, env.y, env.x + half_w, env.y + cfg.player_h))
    assert isinstance(plats, list)
    assert len(plats) >= 1


