"""Angular operators for an axisymmetric diatomic rotor."""

from dataclasses import dataclass

import numpy as np
from grid_lib.spherical_coordinates import LM_to_I, get_y


@dataclass(frozen=True)
class RotorAngularOperators:
    """Projected angular multiplication operators in a selected l sector."""

    ell: np.ndarray
    identity: np.ndarray
    cos_theta: np.ndarray
    cos_theta_squared: np.ndarray
    cos_theta_cubed: np.ndarray
    sin_theta_squared_cos_theta: np.ndarray
    cos_two_theta: np.ndarray


def rotor_angular_operators(l_max: int, ell=None) -> RotorAngularOperators:
    """Construct exact m=0 angular matrices from grid-lib Gaunt coefficients."""
    if isinstance(l_max, (bool, np.bool_)) or not isinstance(
        l_max, (int, np.integer)
    ):
        raise TypeError("l_max must be a nonnegative integer.")
    if l_max < 0:
        raise ValueError("l_max must be nonnegative.")
    l_max = int(l_max)
    if ell is None:
        ell = np.arange(l_max + 1)
    ell = np.asarray(ell)
    if (
        ell.ndim != 1
        or ell.dtype.kind not in "iu"
        or len(ell) == 0
        or np.any(ell < 0)
        or np.any(ell > l_max)
        or np.any(np.diff(ell) <= 0)
    ):
        raise ValueError(
            "ell must be a nonempty increasing integer subset of 0..l_max."
        )
    ell = ell.astype(int)

    maximum_multipole = 3
    y = get_y(l_max, 0, maximum_multipole, 0)

    def multipole(degree):
        index = LM_to_I(degree, 0, maximum_multipole, 0)
        return y[index][np.ix_(ell, ell)]

    identity = np.eye(len(ell))
    p1 = np.sqrt(4 * np.pi / 3) * multipole(1)
    p2 = np.sqrt(4 * np.pi / 5) * multipole(2)
    p3 = np.sqrt(4 * np.pi / 7) * multipole(3)
    cos_squared = identity / 3 + 2 * p2 / 3
    cos_cubed = (3 * p1 + 2 * p3) / 5
    sin_squared_cos = p1 - cos_cubed

    return RotorAngularOperators(
        ell=ell,
        identity=identity,
        cos_theta=p1,
        cos_theta_squared=cos_squared,
        cos_theta_cubed=cos_cubed,
        sin_theta_squared_cos_theta=sin_squared_cos,
        cos_two_theta=2 * cos_squared - identity,
    )
