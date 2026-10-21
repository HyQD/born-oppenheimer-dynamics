"""Electronic potential and response-property curves for diatomics."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline


def _curve(values, name, size, *, optional=False):
    if values is None and optional:
        return None
    array = np.asarray(values)
    if array.shape != (size,) or np.iscomplexobj(array):
        raise ValueError(
            f"{name} must be a real one-dimensional curve of length {size}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class InterpolatedDiatomicSurfaces:
    """Electronic curves evaluated on a nuclear radial grid."""

    electronic_energy: np.ndarray
    dipole: np.ndarray | None
    polarizability_parallel: np.ndarray
    polarizability_perpendicular: np.ndarray
    beta_parallel: np.ndarray | None
    beta_perpendicular: np.ndarray | None


class DiatomicSurfaces:
    """Tabulated Born-Oppenheimer energy and electric-response curves.

    All quantities use atomic units. The molecular axis for the underlying
    clamped-nuclei electronic calculation is the body-fixed z axis.
    """

    def __init__(
        self,
        internuclear_distance,
        electronic_energy,
        *,
        polarizability_parallel,
        polarizability_perpendicular,
        dipole=None,
        beta_parallel=None,
        beta_perpendicular=None,
        label=None,
    ):
        distance = np.asarray(internuclear_distance)
        if (
            distance.ndim != 1
            or len(distance) < 2
            or np.iscomplexobj(distance)
            or not np.all(np.isfinite(distance))
            or np.any(distance <= 0)
            or np.any(np.diff(distance) <= 0)
        ):
            raise ValueError(
                "internuclear_distance must be finite, positive, and strictly "
                "increasing."
            )
        self.internuclear_distance = np.array(distance, dtype=float, copy=True)
        self.internuclear_distance.setflags(write=False)
        size = len(distance)
        self.electronic_energy = _curve(electronic_energy, "electronic_energy", size)
        self.dipole = _curve(dipole, "dipole", size, optional=True)
        self.polarizability_parallel = _curve(
            polarizability_parallel, "polarizability_parallel", size
        )
        self.polarizability_perpendicular = _curve(
            polarizability_perpendicular, "polarizability_perpendicular", size
        )
        if (beta_parallel is None) != (beta_perpendicular is None):
            raise ValueError(
                "beta_parallel and beta_perpendicular must either both be supplied "
                "or both be omitted."
            )
        self.beta_parallel = _curve(
            beta_parallel, "beta_parallel", size, optional=True
        )
        self.beta_perpendicular = _curve(
            beta_perpendicular, "beta_perpendicular", size, optional=True
        )
        self.label = None if label is None else str(label)

    @property
    def r_min(self) -> float:
        return float(self.internuclear_distance[0])

    @property
    def r_max(self) -> float:
        return float(self.internuclear_distance[-1])

    def interpolate(self, internuclear_distance) -> InterpolatedDiatomicSurfaces:
        """Evaluate every supplied curve on a nuclear grid.

        Values below the first electronic-structure point are spline
        extrapolated. Values beyond the final point are rejected.
        """
        points = np.asarray(internuclear_distance)
        if (
            points.ndim != 1
            or np.iscomplexobj(points)
            or not np.all(np.isfinite(points))
            or np.any(points <= 0)
        ):
            raise ValueError("Interpolation points must be finite and positive.")
        tolerance = 100 * np.finfo(float).eps * max(1.0, self.r_max)
        if np.max(points) > self.r_max + tolerance:
            raise ValueError(
                f"The nuclear grid extends to {np.max(points):.12g} bohr, beyond "
                f"the supplied electronic data at {self.r_max:.12g} bohr."
            )

        def evaluate(values):
            if values is None:
                return None
            return np.asarray(
                CubicSpline(self.internuclear_distance, values)(points), dtype=float
            )

        return InterpolatedDiatomicSurfaces(
            electronic_energy=evaluate(self.electronic_energy),
            dipole=evaluate(self.dipole),
            polarizability_parallel=evaluate(self.polarizability_parallel),
            polarizability_perpendicular=evaluate(
                self.polarizability_perpendicular
            ),
            beta_parallel=evaluate(self.beta_parallel),
            beta_perpendicular=evaluate(self.beta_perpendicular),
        )

    @classmethod
    def from_npz(cls, path, *, energy_state=0, label=None):
        """Load the established ``R``, ``E_R``, and ``alpha`` NPZ layout."""
        path = Path(path)
        with np.load(path) as data:
            distance = data["R"]
            energy = data["E_R"]
            if energy.ndim == 2:
                energy = energy[energy_state]
            alpha = data["alpha"]
            if alpha.shape != (len(distance), 3, 3):
                raise ValueError("alpha must have shape (n_distance, 3, 3).")
            perpendicular = alpha[:, 0, 0]
            parallel = alpha[:, 2, 2]
            dipole = data["dipole"] if "dipole" in data else None
            beta_parallel = data["beta_parallel"] if "beta_parallel" in data else None
            beta_perpendicular = (
                data["beta_perpendicular"] if "beta_perpendicular" in data else None
            )
        return cls(
            distance,
            energy,
            polarizability_parallel=parallel,
            polarizability_perpendicular=perpendicular,
            dipole=dipole,
            beta_parallel=beta_parallel,
            beta_perpendicular=beta_perpendicular,
            label=path.stem if label is None else label,
        )


def h2plus_aug_cc_pv5z_surfaces() -> DiatomicSurfaces:
    """Load the packaged H2+ aug-cc-pV5Z reference curves."""
    resource = files(__package__).joinpath(
        "data/h2+_aug-cc-pv5z_rmin=0.1_rmax=12.0.npz"
    )
    with resource.open("rb") as stream:
        with np.load(stream) as data:
            distance = data["R"].copy()
            energy = data["E_R"][0].copy()
            alpha = data["alpha"].copy()
    return DiatomicSurfaces(
        distance,
        energy,
        polarizability_parallel=alpha[:, 2, 2],
        polarizability_perpendicular=alpha[:, 0, 0],
        dipole=None,
        label="H2+ aug-cc-pV5Z",
    )
