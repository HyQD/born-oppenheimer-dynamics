"""Results returned by vibrational-rotor propagation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VibrationalRotorResult:
    """Sampled observables and the final rovibrational wavefunction."""

    times: np.ndarray
    electric_field: np.ndarray
    norms: np.ndarray
    expectation_r: np.ndarray
    expectation_cos_theta: np.ndarray
    expectation_cos_theta_squared: np.ndarray
    expectation_z: np.ndarray
    solver_iterations: np.ndarray
    residual_norms: np.ndarray
    solver_methods: np.ndarray
    final_state: np.ndarray
    radial_grid: np.ndarray
    radial_weights: np.ndarray
    ell: np.ndarray
    time_step: float
    n_steps: int

    @property
    def expec_r(self):
        """Compatibility alias for ``expectation_r``."""
        return self.expectation_r

    @property
    def expec_cost(self):
        """Compatibility alias for ``expectation_cos_theta``."""
        return self.expectation_cos_theta

    @property
    def expec_cost_sq(self):
        """Compatibility alias for ``expectation_cos_theta_squared``."""
        return self.expectation_cos_theta_squared

    @property
    def expec_z(self):
        """Compatibility alias for ``expectation_z``."""
        return self.expectation_z

    def save(self, path):
        """Save all result arrays to a compressed NPZ file."""
        np.savez_compressed(
            Path(path),
            times=self.times,
            electric_field=self.electric_field,
            norms=self.norms,
            expectation_r=self.expectation_r,
            expectation_cos_theta=self.expectation_cos_theta,
            expectation_cos_theta_squared=self.expectation_cos_theta_squared,
            expectation_z=self.expectation_z,
            solver_iterations=self.solver_iterations,
            residual_norms=self.residual_norms,
            solver_methods=self.solver_methods,
            final_state=self.final_state,
            radial_grid=self.radial_grid,
            radial_weights=self.radial_weights,
            ell=self.ell,
            time_step=self.time_step,
            n_steps=self.n_steps,
        )
