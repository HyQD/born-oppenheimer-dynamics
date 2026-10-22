"""Shared angular-rotor interaction machinery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def select_angular_sector(requested, initial_rotational, has_odd_interaction):
    """Select a full or parity-restricted angular-momentum sector."""
    if requested not in {"auto", "full", "even", "odd"}:
        raise ValueError("angular_sector must be 'auto', 'full', 'even', or 'odd'.")
    if requested == "auto":
        if has_odd_interaction or initial_rotational is None:
            return "full"
        return "even" if initial_rotational % 2 == 0 else "odd"
    if requested != "full" and has_odd_interaction:
        raise ValueError(
            "An even/odd angular sector cannot be used with dipole or "
            "first-hyperpolarizability coupling."
        )
    if initial_rotational is not None and requested != "full":
        expected = "even" if initial_rotational % 2 == 0 else "odd"
        if requested != expected:
            raise ValueError(
                f"The initial rotational state requires the {expected} sector."
            )
    return requested


def angular_momenta(l_max, sector):
    """Return angular momenta retained by an angular sector."""
    if sector == "full":
        ell = np.arange(l_max + 1)
    elif sector == "even":
        ell = np.arange(0, l_max + 1, 2)
    else:
        ell = np.arange(1, l_max + 1, 2)
    if len(ell) == 0:
        raise ValueError("l_max does not contain the selected angular sector.")
    return ell


def _coefficient(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or np.iscomplexobj(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite real one-dimensional array.")
    return np.asarray(array, dtype=float)


@dataclass(frozen=True)
class RotorFieldInteraction:
    """Length-gauge interaction shared by rigid and vibrational rotors."""

    angular: object
    dipole: np.ndarray
    alpha_parallel: np.ndarray
    alpha_perpendicular: np.ndarray
    beta_parallel: np.ndarray
    beta_perpendicular: np.ndarray

    def __post_init__(self):
        names = (
            "dipole",
            "alpha_parallel",
            "alpha_perpendicular",
            "beta_parallel",
            "beta_perpendicular",
        )
        arrays = tuple(_coefficient(getattr(self, name), name) for name in names)
        size = len(arrays[0])
        if size == 0 or any(len(array) != size for array in arrays[1:]):
            raise ValueError("All rotor interaction coefficients must have one size.")
        for name, array in zip(names, arrays):
            object.__setattr__(self, name, array)
        object.__setattr__(
            self,
            "_has_hyperpolarizability",
            bool(
                np.any(self.beta_parallel != 0)
                or np.any(self.beta_perpendicular != 0)
            ),
        )
        object.__setattr__(
            self,
            "_operator_norms",
            (
                np.linalg.norm(self.angular.cos_theta, 2),
                np.linalg.norm(self.angular.cos_theta_squared, 2),
                np.linalg.norm(self.angular.cos_theta_cubed, 2),
                np.linalg.norm(self.angular.sin_theta_squared_cos_theta, 2),
            ),
        )

    @property
    def alpha_anisotropy(self):
        return self.alpha_parallel - self.alpha_perpendicular

    @property
    def has_odd_interaction(self):
        return bool(
            np.any(self.dipole != 0)
            or np.any(self.beta_parallel != 0)
            or np.any(self.beta_perpendicular != 0)
        )

    def apply(self, state, field):
        """Apply the field interaction to angular-by-coordinate coefficients."""
        result = -field * (
            self.angular.cos_theta @ (self.dipole[None, :] * state)
        )
        result -= 0.5 * field**2 * (
            self.alpha_perpendicular[None, :] * state
            + self.angular.cos_theta_squared
            @ (self.alpha_anisotropy[None, :] * state)
        )
        if self._has_hyperpolarizability:
            result -= field**3 / 6 * (
                self.angular.cos_theta_cubed
                @ (self.beta_parallel[None, :] * state)
                + 3
                * self.angular.sin_theta_squared_cos_theta
                @ (self.beta_perpendicular[None, :] * state)
            )
        return result

    def bound(self, field):
        """Return a spectral-norm upper bound for fixed-point iteration."""
        cos_norm, cos_squared_norm, cos_cubed_norm, sin_squared_cos_norm = (
            self._operator_norms
        )
        bound = abs(field) * np.max(np.abs(self.dipole)) * cos_norm
        bound += 0.5 * field**2 * (
            np.max(np.abs(self.alpha_perpendicular))
            + np.max(np.abs(self.alpha_anisotropy))
            * cos_squared_norm
        )
        bound += abs(field) ** 3 / 6 * (
            np.max(np.abs(self.beta_parallel))
            * cos_cubed_norm
            + 3
            * np.max(np.abs(self.beta_perpendicular))
            * sin_squared_cos_norm
        )
        return float(bound)
