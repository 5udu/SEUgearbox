from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np


FitnessFunction = Callable[[np.ndarray], tuple[float, object]]


@dataclass(frozen=True)
class OptimizationResult:
    best_position: np.ndarray
    best_score: float
    best_payload: object


class HarrisHawkOptimization:
    """A lightweight Harris Hawks Optimizer for bounded continuous search."""

    def __init__(
        self,
        fitness_function: FitnessFunction,
        n_hawks: int,
        dim: int,
        max_iter: int,
        lb: Iterable[float],
        ub: Iterable[float],
        seed: Optional[int] = None,
    ) -> None:
        self.fitness_function = fitness_function
        self.n_hawks = int(n_hawks)
        self.dim = int(dim)
        self.max_iter = int(max_iter)
        self.lb = np.asarray(list(lb), dtype=float)
        self.ub = np.asarray(list(ub), dtype=float)
        self.rng = np.random.default_rng(seed)
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.n_hawks < 2:
            raise ValueError("n_hawks must be at least 2")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.lb.shape != (self.dim,) or self.ub.shape != (self.dim,):
            raise ValueError("lb and ub must match dim")
        if np.any(self.lb >= self.ub):
            raise ValueError("each lower bound must be smaller than its upper bound")

    def _evaluate_population(
        self, hawks: np.ndarray
    ) -> tuple[np.ndarray, list[object]]:
        fitness = np.empty(self.n_hawks, dtype=float)
        payloads: list[object] = [None] * self.n_hawks
        for index, hawk in enumerate(hawks):
            fitness[index], payloads[index] = self.fitness_function(hawk.copy())
        return fitness, payloads

    def optimize(self, verbose: bool = True) -> tuple[np.ndarray, float, object]:
        hawks = self.rng.uniform(self.lb, self.ub, size=(self.n_hawks, self.dim))
        fitness, payloads = self._evaluate_population(hawks)

        best_index = int(np.argmin(fitness))
        rabbit_pos = hawks[best_index].copy()
        rabbit_energy = float(fitness[best_index])
        rabbit_payload = payloads[best_index]

        global_best = OptimizationResult(
            best_position=rabbit_pos.copy(),
            best_score=rabbit_energy,
            best_payload=rabbit_payload,
        )

        stagnation_count = 0
        last_best_score = rabbit_energy

        for iteration in range(self.max_iter):
            escaping_energy = self._escaping_energy(iteration)
            stagnation_limit = max(2, int(2 + 3 * iteration / self.max_iter))

            for hawk_index in range(self.n_hawks):
                hawks[hawk_index] = self._propose_position(
                    hawks=hawks,
                    hawk_index=hawk_index,
                    rabbit_pos=rabbit_pos,
                    escaping_energy=escaping_energy,
                )
                hawks[hawk_index] = np.clip(hawks[hawk_index], self.lb, self.ub)

                current_score, current_payload = self.fitness_function(hawks[hawk_index].copy())
                if current_score < fitness[hawk_index]:
                    fitness[hawk_index] = current_score
                    payloads[hawk_index] = current_payload

                if current_score < rabbit_energy:
                    rabbit_energy = current_score
                    rabbit_pos = hawks[hawk_index].copy()
                    rabbit_payload = current_payload

                if current_score < global_best.best_score:
                    global_best = OptimizationResult(
                        best_position=hawks[hawk_index].copy(),
                        best_score=float(current_score),
                        best_payload=current_payload,
                    )

            if abs(last_best_score - rabbit_energy) < 1e-6:
                stagnation_count += 1
                if stagnation_count >= stagnation_limit:
                    rabbit_pos, rabbit_energy, rabbit_payload = self._perturb_rabbit(
                        rabbit_pos,
                        rabbit_energy,
                        rabbit_payload,
                        iteration,
                    )
                    if rabbit_energy < global_best.best_score:
                        global_best = OptimizationResult(
                            best_position=rabbit_pos.copy(),
                            best_score=float(rabbit_energy),
                            best_payload=rabbit_payload,
                        )
                    stagnation_count = 0
            else:
                stagnation_count = 0

            last_best_score = rabbit_energy

            if verbose:
                print(
                    f"Iteration {iteration + 1}/{self.max_iter} | "
                    f"current={-rabbit_energy:.4f} | best={-global_best.best_score:.4f}"
                )

        return (
            global_best.best_position.copy(),
            -global_best.best_score,
            global_best.best_payload,
        )

    def _escaping_energy(self, iteration: int) -> float:
        e0 = self.rng.uniform(-1.0, 1.0)
        return 2 * e0 * (1 - (iteration / self.max_iter) ** 2)

    def _propose_position(
        self,
        hawks: np.ndarray,
        hawk_index: int,
        rabbit_pos: np.ndarray,
        escaping_energy: float,
    ) -> np.ndarray:
        hawk = hawks[hawk_index].copy()
        abs_energy = abs(escaping_energy)

        if abs_energy >= 1:
            if self.rng.random() < 0.5:
                partner_index = int(self.rng.integers(self.n_hawks))
                partner = hawks[partner_index]
                return partner - self.rng.random() * np.abs(
                    partner - 2 * self.rng.random() * hawk
                )
            mean_hawk = hawks.mean(axis=0)
            return (
                (rabbit_pos - mean_hawk) * self.rng.random()
                + self.lb
                + self.rng.random() * (self.ub - self.lb)
            )

        jump_strength = 2 * (1 - self.rng.random())
        random_gate = self.rng.random()
        if random_gate >= 0.5 and abs_energy >= 0.5:
            return rabbit_pos - escaping_energy * np.abs(rabbit_pos - hawk)
        if random_gate >= 0.5 and abs_energy < 0.5:
            return rabbit_pos - escaping_energy * np.abs(2 * rabbit_pos - hawk)
        return rabbit_pos - escaping_energy * np.abs(jump_strength * rabbit_pos - hawk)

    def _perturb_rabbit(
        self,
        rabbit_pos: np.ndarray,
        rabbit_energy: float,
        rabbit_payload: object,
        iteration: int,
    ) -> tuple[np.ndarray, float, object]:
        scale = 0.1 * (1 - iteration / self.max_iter)
        candidate = rabbit_pos + self.rng.uniform(-scale, scale, self.dim) * (self.ub - self.lb)
        candidate = np.clip(candidate, self.lb, self.ub)
        candidate_score, candidate_payload = self.fitness_function(candidate.copy())
        if candidate_score < rabbit_energy:
            return candidate, float(candidate_score), candidate_payload
        return rabbit_pos, rabbit_energy, rabbit_payload
