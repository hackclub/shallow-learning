from __future__ import annotations

import os
import numpy as np

from types import SimpleNamespace

from platformer.cli import cmd_render


def test_render_best_only_selects_highest_fit(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Create dummy checkpoint files with varying fit numbers
    fits = [10.0, 25.5, 17.2]
    paths = []
    for i, f in enumerate(fits, start=1):
        p = ckpt_dir / f"best_gen{i}_seed123_fit{f:.3f}.npy"
        # Content is irrelevant because we stub render; write a small array
        np.save(p, np.array([1, 2, 3], dtype=np.float32))
        paths.append(str(p))

    captured = {}

    def fake_render(weights_paths, **kwargs):  # type: ignore
        captured["weights"] = list(weights_paths or [])
        captured["kwargs"] = kwargs

    # Monkeypatch renderer.render before cmd_render imports it
    import platformer.renderer as renderer

    monkeypatch.setattr(renderer, "render", fake_render)

    args = SimpleNamespace(
        ckpt_dir=str(ckpt_dir),
        weights=[],
        speed=1.0,
        seed=None,
        hitboxes=False,
        fullscreen=False,
        log_rays=False,
        best_only=True,
    )

    rc = cmd_render(args)
    assert rc == 0
    assert "weights" in captured
    # Only one file should be selected and it should be the highest fit (25.5)
    assert len(captured["weights"]) == 1
    assert captured["weights"][0].endswith("fit25.500.npy")


def test_render_gen_filter(tmp_path, monkeypatch):
    ckpt_dir = tmp_path / "ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    for gen, fit in [(1, 10.0), (2, 20.0), (2, 15.0), (3, 5.0)]:
        p = ckpt_dir / f"best_gen{gen}_seed123_fit{fit:.3f}.npy"
        np.save(p, np.array([1, 2, 3], dtype=np.float32))

    captured = {}

    def fake_render(weights_paths, **kwargs):  # type: ignore
        captured["weights"] = list(weights_paths or [])

    import platformer.renderer as renderer
    monkeypatch.setattr(renderer, "render", fake_render)

    args = SimpleNamespace(
        ckpt_dir=str(ckpt_dir),
        weights=[],
        speed=1.0,
        seed=None,
        hitboxes=False,
        fullscreen=False,
        log_rays=False,
        best_only=False,
        gen=2,
    )

    rc = cmd_render(args)
    assert rc == 0
    assert len(captured["weights"]) == 2
    assert all("gen2_" in os.path.basename(w) for w in captured["weights"])  # type: ignore

