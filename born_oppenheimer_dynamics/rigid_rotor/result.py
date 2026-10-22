"""Results returned by rigid-rotor propagation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RigidRotorResult:
    """Sampled angular observables and the final rotational state."""

    times: np.ndarray
    electric_field: np.ndarray
    norms: np.ndarray
    expectation_cos_theta: np.ndarray
    expectation_cos_theta_squared: np.ndarray
    angular_momentum_populations: np.ndarray
    solver_iterations: np.ndarray
    residual_norms: np.ndarray
    solver_methods: np.ndarray
    final_state: np.ndarray
    ell: np.ndarray
    bond_length: float | None
    rotational_constant: float
    time_step: float
    n_steps: int

    @property
    def expec_cost(self):
        """Compatibility alias for ``expectation_cos_theta``."""
        return self.expectation_cos_theta

    @property
    def expec_cost_sq(self):
        """Compatibility alias for ``expectation_cos_theta_squared``."""
        return self.expectation_cos_theta_squared

    def save(self, path):
        """Save all result arrays and scalar metadata to a compressed NPZ file."""
        np.savez_compressed(
            Path(path),
            times=self.times,
            electric_field=self.electric_field,
            norms=self.norms,
            expectation_cos_theta=self.expectation_cos_theta,
            expectation_cos_theta_squared=self.expectation_cos_theta_squared,
            angular_momentum_populations=self.angular_momentum_populations,
            solver_iterations=self.solver_iterations,
            residual_norms=self.residual_norms,
            solver_methods=self.solver_methods,
            final_state=self.final_state,
            ell=self.ell,
            bond_length=np.nan if self.bond_length is None else self.bond_length,
            rotational_constant=self.rotational_constant,
            time_step=self.time_step,
            n_steps=self.n_steps,
        )
