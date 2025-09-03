from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class MLPPolicyConfig:
    input_size: int
    hidden_sizes: Tuple[int, ...] = (32, 32)
    output_size: int = 6


class MLPPolicy:
    def __init__(self, cfg: MLPPolicyConfig, rng: np.random.Generator | None = None) -> None:
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.shapes = self._layer_shapes()
        self.num_params = sum(w.size for w in self._init_params(scale=0.01))
        self.params = self._init_params(scale=0.01)

    def _layer_shapes(self) -> List[Tuple[int, int]]:
        sizes = [self.cfg.input_size, *self.cfg.hidden_sizes, self.cfg.output_size]
        shapes: List[Tuple[int, int]] = []
        for in_s, out_s in zip(sizes[:-1], sizes[1:]):
            shapes.append((out_s, in_s + 1))  # +1 for bias
        return shapes

    def _init_params(self, scale: float) -> List[np.ndarray]:
        return [self.rng.normal(0.0, scale, size=shape) for shape in self.shapes]

    def get_flat(self) -> np.ndarray:
        return np.concatenate([w.ravel() for w in self.params])

    def set_flat(self, flat: np.ndarray) -> None:
        offset = 0
        new_params: List[np.ndarray] = []
        for shape in self.shapes:
            size = int(np.prod(shape))
            new_params.append(flat[offset:offset + size].reshape(shape))
            offset += size
        self.params = new_params

    def forward(self, obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float32)
        for i, w in enumerate(self.params):
            x = np.concatenate([x, np.ones(1, dtype=x.dtype)], axis=0)  # bias
            x = w @ x
            if i < len(self.params) - 1:
                x = np.tanh(x)
        return x

    def act(self, obs: np.ndarray) -> int:
        logits = self.forward(obs)
        return int(np.argmax(logits))
