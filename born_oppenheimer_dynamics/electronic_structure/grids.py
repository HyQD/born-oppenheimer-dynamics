"""Grid construction shared by electronic-structure backends."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Real

import numpy as np


def _finite_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar.")
    return float(value)


def radial_grid_from_range(radial_range: Iterable[Real]) -> np.ndarray:
    """Return an inclusive, uniformly spaced radial grid.

    radial_range is (r_min, r_max, dr) in atomic units. The interval must
    contain an integral number of steps within floating-point tolerance.
    numpy.linspace is used so that the last grid point is exactly r_max.
    """
    try:
        values = tuple(radial_range)
    except TypeError as error:
        raise TypeError(
            "radial_range must be an iterable (r_min, r_max, dr)."
        ) from error
    if len(values) != 3:
        raise ValueError("radial_range must contain exactly (r_min, r_max, dr).")

    r_min = _finite_real(values[0], "r_min")
    r_max = _finite_real(values[1], "r_max")
    dr = _finite_real(values[2], "dr")
    if r_min <= 0:
        raise ValueError("r_min must be positive.")
    if r_max < r_min:
        raise ValueError("r_max must be greater than or equal to r_min.")
    if dr <= 0:
        raise ValueError("dr must be positive.")

    step_count = (r_max - r_min) / dr
    rounded_count = round(step_count)
    endpoint = r_min + rounded_count * dr
    tolerance = (
        100
        * np.finfo(float).eps
        * max(1.0, abs(r_min), abs(r_max), abs(endpoint))
    )
    if rounded_count < 0 or not np.isclose(
        endpoint, r_max, rtol=0.0, atol=tolerance
    ):
        raise ValueError(
            "The interval from r_min to r_max must contain an integral "
            "number of steps of size dr."
        )

    grid = np.linspace(r_min, r_max, rounded_count + 1)
    grid.setflags(write=False)
    return grid
