"""Complex absorbing potentials for nuclear radial grids."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np


def _finite_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar.")
    return float(value)


@dataclass(frozen=True)
class CubicCAP:
    r"""Cubic complex absorbing potential near the outer radial boundary.

    The potential is zero through ``alpha * r_max`` and is
    ``-1j * eta * (r - alpha * r_max)**3`` beyond that point.
    """

    alpha: float
    eta: float = 1.0

    def __post_init__(self):
        alpha = _finite_real(self.alpha, "alpha")
        eta = _finite_real(self.eta, "eta")
        if not 0 <= alpha < 1:
            raise ValueError("alpha must satisfy 0 <= alpha < 1.")
        if eta <= 0:
            raise ValueError("eta must be positive.")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "eta", eta)

    def __call__(self, radial_grid, r_max):
        """Evaluate the CAP on ``radial_grid`` for outer boundary ``r_max``."""
        radial_grid = np.asarray(radial_grid)
        if (
            radial_grid.ndim != 1
            or np.iscomplexobj(radial_grid)
            or not np.all(np.isfinite(radial_grid))
        ):
            raise ValueError("radial_grid must be a finite real one-dimensional array.")
        r_max = _finite_real(r_max, "r_max")
        if r_max <= 0:
            raise ValueError("r_max must be positive.")
        displacement = np.maximum(radial_grid - self.alpha * r_max, 0.0)
        return -1j * self.eta * displacement**3
