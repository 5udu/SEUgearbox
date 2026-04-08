from __future__ import annotations

import numpy as np

from hho import HarrisHawkOptimization


def test_hho_improves_simple_sphere_objective() -> None:
    def sphere(position: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.sum(position**2)), position.copy()

    optimizer = HarrisHawkOptimization(
        fitness_function=sphere,
        n_hawks=12,
        dim=3,
        max_iter=40,
        lb=[-5.0, -5.0, -5.0],
        ub=[5.0, 5.0, 5.0],
        seed=7,
    )

    best_position, best_score, payload = optimizer.optimize(verbose=False)

    assert best_position.shape == (3,)
    assert np.linalg.norm(payload) < 0.5
    assert best_score > -0.3
