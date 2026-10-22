"""High-level time-dependent vibrational-rotor interface."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

import numpy as np
from grid_lib.pseudospectral_grids import FEMDVR, GaussLegendreLobatto, Linear_map
from scipy.linalg import eigh

from .angular import rotor_angular_operators
from .propagator import SplitCrankNicolson
from .result import VibrationalRotorResult
from .surfaces import DiatomicSurfaces
from .systems import DiatomicMolecule


def _finite_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real scalar.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar.")
    return float(value)


def _positive_real(value, name):
    result = _finite_real(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _radial_grid(*, r_max, nodes, n_elements, element_size, n_points):
    if nodes is None:
        r_max = _positive_real(r_max, "r_max")
        if element_size is not None:
            element_size = _positive_real(element_size, "element_size")
            element_count = r_max / element_size
            rounded_count = round(element_count)
            if rounded_count < 1 or not np.isclose(
                element_count, rounded_count, rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    "r_max must be an integer multiple of element_size."
                )
            if n_elements is not None and _integer(
                n_elements, "n_elements", 1
            ) != rounded_count:
                raise ValueError(
                    "n_elements must agree with r_max and element_size."
                )
            n_elements = rounded_count
        else:
            n_elements = 10 if n_elements is None else _integer(
                n_elements, "n_elements", 1
            )
        boundaries = np.linspace(0.0, r_max, n_elements + 1)
    else:
        if element_size is not None:
            raise ValueError("element_size cannot be combined with explicit nodes.")
        boundaries = np.asarray(nodes)
        if (
            boundaries.ndim != 1
            or len(boundaries) < 2
            or np.iscomplexobj(boundaries)
            or not np.all(np.isfinite(boundaries))
            or boundaries[0] != 0
            or np.any(np.diff(boundaries) <= 0)
        ):
            raise ValueError(
                "nodes must be finite, increasing element boundaries starting at zero."
            )
        boundaries = boundaries.astype(float)
        if r_max is not None and not np.isclose(
            _positive_real(r_max, "r_max"), boundaries[-1], rtol=0, atol=1e-12
        ):
            raise ValueError("r_max must equal the final explicit node.")
        if n_elements is not None and _integer(
            n_elements, "n_elements", 1
        ) != len(boundaries) - 1:
            raise ValueError("n_elements must agree with the explicit nodes.")

    count = len(boundaries) - 1
    points = np.asarray(n_points)
    if points.ndim == 0:
        if isinstance(n_points, (bool, np.bool_)) or points.dtype.kind not in "iu":
            raise TypeError("n_points must contain integers.")
        points = np.full(count, int(points))
    elif points.shape == (count,) and points.dtype.kind in "iu":
        points = points.astype(int)
    else:
        raise ValueError("n_points must be one integer or one integer per element.")
    if np.any(points < 3):
        raise ValueError("Every element must contain at least three points.")
    return FEMDVR(
        boundaries, points, Linear_map, GaussLegendreLobatto, symmetrize=False
    )


@dataclass(frozen=True)
class RovibrationalState:
    """Field-free vibrational and rotational quantum numbers."""

    vibrational: int = 0
    rotational: int = 0

    def __post_init__(self):
        object.__setattr__(
            self,
            "vibrational",
            _integer(self.vibrational, "vibrational quantum number"),
        )
        object.__setattr__(
            self,
            "rotational",
            _integer(self.rotational, "rotational quantum number"),
        )

def _state_specification(initial_state):
    if isinstance(initial_state, str):
        if initial_state == "groundstate":
            return RovibrationalState()
        raise ValueError("initial_state string must be 'groundstate'.")
    if isinstance(initial_state, RovibrationalState):
        return initial_state
    if isinstance(initial_state, tuple) and len(initial_state) == 2:
        return RovibrationalState(*initial_state)
    return None


class VibrationalRotorDiatomic:
    r"""Time-dependent nuclear vibrational-rotor dynamics for a diatomic.

    The molecular orientation is expanded in ``Y_l0`` and the electric field
    is linearly polarized along the laboratory z axis. Electronic curves are
    body-fixed quantities calculated with the molecular axis on z. The
    length-gauge interaction is

    ``-mu*E*cos(theta) - E**2*(alpha_perp + Delta alpha*cos(theta)**2)/2``

    plus the optional first-hyperpolarizability contribution

    ``-E**3*(beta_parallel*cos(theta)**3 +
    3*beta_perp*sin(theta)**2*cos(theta))/6``.

    ``cap`` may be ``None`` or a callable with signature
    ``cap(radial_grid, r_max)``. It is applied only during propagation, not
    while constructing a field-free rovibrational initial state.
    """

    def __init__(
        self,
        molecule,
        surfaces,
        *,
        r_max=None,
        nodes=None,
        n_elements=None,
        element_size=None,
        n_points=15,
        l_max=12,
        angular_sector="auto",
        initial_state="groundstate",
        fields=None,
        cap=None,
        initial_time=0.0,
        final_time=None,
        time_step=None,
        integrator_options=None,
        sample_interval=1,
    ):
        if not isinstance(molecule, DiatomicMolecule):
            molecule = DiatomicMolecule(tuple(molecule))
        if not isinstance(surfaces, DiatomicSurfaces):
            raise TypeError("surfaces must be a DiatomicSurfaces instance.")
        self.molecule = molecule
        self.surfaces = surfaces
        self.l_max = _integer(l_max, "l_max")
        self.initial_state_specification = _state_specification(initial_state)
        self.fields = self._field_terms(fields)

        if r_max is None and nodes is None:
            r_max = surfaces.r_max
        self.grid = _radial_grid(
            r_max=r_max,
            nodes=nodes,
            n_elements=n_elements,
            element_size=element_size,
            n_points=n_points,
        )
        if self.grid.r[-1] > surfaces.r_max + 1e-12:
            raise ValueError(
                f"The radial boundary {self.grid.r[-1]:.12g} bohr exceeds the "
                f"electronic-surface limit {surfaces.r_max:.12g} bohr."
            )
        self.r = np.asarray(self.grid.r[1:-1], dtype=float)
        self.radial_weights = np.asarray(self.grid.weights[1:-1], dtype=float)
        self._sqrt_weights = np.sqrt(self.radial_weights)
        self.cap = cap
        self.cap_potential = self._evaluate_cap(cap)
        interpolated = surfaces.interpolate(self.r)
        self.electronic_energy = interpolated.electronic_energy
        if interpolated.dipole is None:
            if not molecule.homonuclear:
                raise ValueError(
                    "A permanent-dipole curve is required for a heteronuclear molecule."
                )
            self.dipole = np.zeros_like(self.r)
        else:
            self.dipole = interpolated.dipole
        self.alpha_parallel = interpolated.polarizability_parallel
        self.alpha_perpendicular = interpolated.polarizability_perpendicular
        self.alpha_anisotropy = self.alpha_parallel - self.alpha_perpendicular
        self.beta_parallel = (
            np.zeros_like(self.r)
            if interpolated.beta_parallel is None
            else interpolated.beta_parallel
        )
        self.beta_perpendicular = (
            np.zeros_like(self.r)
            if interpolated.beta_perpendicular is None
            else interpolated.beta_perpendicular
        )
        self.has_odd_interaction = bool(
            np.any(self.dipole != 0)
            or np.any(self.beta_parallel != 0)
            or np.any(self.beta_perpendicular != 0)
        )

        self.angular_sector = self._angular_sector(
            angular_sector, self.initial_state_specification
        )
        if self.angular_sector == "full":
            self.ell = np.arange(self.l_max + 1)
        elif self.angular_sector == "even":
            self.ell = np.arange(0, self.l_max + 1, 2)
        else:
            self.ell = np.arange(1, self.l_max + 1, 2)
        if len(self.ell) == 0:
            raise ValueError("l_max does not contain the selected angular sector.")
        self.angular = rotor_angular_operators(self.l_max, self.ell)

        self.field_free_blocks = self._build_field_free_blocks()
        self.initial_state = self._initial_state(initial_state)
        self.propagation_blocks = self._build_propagation_blocks()

        self.initial_time = _finite_real(initial_time, "initial_time")
        if final_time is None:
            final_time = self._infer_final_time()
        self.final_time = _finite_real(final_time, "final_time")
        if self.final_time <= self.initial_time:
            raise ValueError("final_time must be larger than initial_time.")
        if time_step is None:
            raise ValueError("time_step is required.")
        self.requested_time_step = _positive_real(time_step, "time_step")
        duration = self.final_time - self.initial_time
        self.n_steps = int(np.ceil(duration / self.requested_time_step))
        self.time_step = duration / self.n_steps
        self.sample_interval = _integer(sample_interval, "sample_interval", 1)

        options = dict(integrator_options or {})
        self.propagator = SplitCrankNicolson(
            self.propagation_blocks,
            time_step=self.time_step,
            field=self.electric_field,
            interaction=self._interaction,
            interaction_bound=self._interaction_bound,
            **options,
        )

    def _evaluate_cap(self, cap):
        if cap is None:
            return None
        if not callable(cap):
            raise TypeError("cap must be callable or None.")
        r_max = float(self.grid.nodes[-1])
        values = np.asarray(cap(self.r.copy(), r_max))
        if values.shape != self.r.shape:
            raise ValueError(
                f"cap must return an array with shape {self.r.shape}, "
                f"not {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("cap must return finite values.")
        values = np.asarray(values, dtype=np.complex128)
        if np.any(values.imag > 0):
            raise ValueError(
                "The imaginary part of a complex absorbing potential must "
                "be non-positive."
            )
        return values

    @staticmethod
    def _field_terms(fields):
        if fields is None:
            return ()
        if callable(fields):
            terms = (fields,)
        elif isinstance(fields, Iterable) and not isinstance(fields, (str, bytes)):
            terms = tuple(fields)
        else:
            raise TypeError("fields must be a callable or an iterable of callables.")
        if not all(callable(field) for field in terms):
            raise TypeError("Every field must be callable.")
        for field in terms:
            gauge = getattr(field, "gauge", "length")
            if gauge != "length":
                raise ValueError(
                    "VibrationalRotorDiatomic supports length-gauge fields only."
                )
        return terms

    def _infer_final_time(self):
        if not self.fields:
            raise ValueError("final_time is required for field-free propagation.")
        ends = []
        for field in self.fields:
            support = getattr(field, "time_support", (None, None))
            if support[1] is None:
                raise ValueError(
                    "final_time is required when a field has open time support."
                )
            ends.append(support[1])
        return max(ends)

    def _angular_sector(self, requested, state_specification):
        if requested not in {"auto", "full", "even", "odd"}:
            raise ValueError("angular_sector must be 'auto', 'full', 'even', or 'odd'.")
        if requested == "auto":
            if self.has_odd_interaction or state_specification is None:
                return "full"
            return "even" if state_specification.rotational % 2 == 0 else "odd"
        if requested != "full" and self.has_odd_interaction:
            raise ValueError(
                "An even/odd angular sector cannot be used with dipole or "
                "first-hyperpolarizability coupling."
            )
        if state_specification is not None and requested != "full":
            expected = "even" if state_specification.rotational % 2 == 0 else "odd"
            if requested != expected:
                raise ValueError(
                    f"The initial rotational state requires the {expected} sector."
                )
        return requested

    def _build_field_free_blocks(self):
        derivative = np.asarray(self.grid.D2[1:-1, 1:-1], dtype=float)
        weighted_derivative = (
            self._sqrt_weights[:, None]
            * derivative
            / self._sqrt_weights[None, :]
        )
        weighted_derivative = 0.5 * (weighted_derivative + weighted_derivative.T)
        mass = self.molecule.reduced_mass
        kinetic_radial = -weighted_derivative / (2 * mass)
        blocks = np.empty((len(self.ell), len(self.r), len(self.r)))
        for index, angular_momentum in enumerate(self.ell):
            centrifugal = angular_momentum * (angular_momentum + 1) / (
                2 * mass * self.r**2
            )
            blocks[index] = kinetic_radial + np.diag(
                centrifugal + self.electronic_energy
            )
        return blocks

    def _build_propagation_blocks(self):
        if self.cap_potential is None:
            return self.field_free_blocks
        blocks = np.asarray(self.field_free_blocks, dtype=np.complex128).copy()
        diagonal = np.arange(len(self.r))
        blocks[:, diagonal, diagonal] += self.cap_potential[None, :]
        return blocks

    def _initial_state(self, initial_state):
        specification = self.initial_state_specification
        if specification is not None:
            if specification.rotational > self.l_max:
                raise ValueError("The initial rotational quantum number exceeds l_max.")
            matches = np.flatnonzero(self.ell == specification.rotational)
            if len(matches) != 1:
                raise ValueError("The selected angular sector omits the initial state.")
            channel = int(matches[0])
            size = len(self.r)
            if specification.vibrational >= size:
                raise ValueError(
                    "The vibrational quantum number exceeds the radial basis."
                )
            _, eigenvectors = eigh(
                self.field_free_blocks[channel],
                subset_by_index=[specification.vibrational, specification.vibrational],
            )
            state = np.zeros((len(self.ell), size), dtype=np.complex128)
            state[channel] = eigenvectors[:, 0]
        else:
            values = np.asarray(initial_state)
            expected = (len(self.ell), len(self.r))
            if values.shape != expected or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"A custom initial_state must be finite with shape {expected}."
                )
            state = np.asarray(values, dtype=np.complex128) * self._sqrt_weights
        norm = np.linalg.norm(state)
        if norm == 0:
            raise ValueError("initial_state cannot be zero.")
        return state / norm

    def electric_field(self, time):
        """Return the summed length-gauge electric field at ``time``."""
        value = sum(np.asarray(field(time)) for field in self.fields)
        value = np.asarray(value)
        if value.ndim != 0 or np.iscomplexobj(value) or not np.isfinite(value):
            raise ValueError("Every field must return a finite real scalar.")
        return float(value)

    def _interaction(self, state, field):
        result = -field * (
            self.angular.cos_theta @ (self.dipole[None, :] * state)
        )
        field_squared = field**2
        result -= 0.5 * field_squared * (
            self.alpha_perpendicular[None, :] * state
            + self.angular.cos_theta_squared
            @ (self.alpha_anisotropy[None, :] * state)
        )
        if np.any(self.beta_parallel != 0) or np.any(self.beta_perpendicular != 0):
            result -= field**3 / 6 * (
                self.angular.cos_theta_cubed
                @ (self.beta_parallel[None, :] * state)
                + 3
                * self.angular.sin_theta_squared_cos_theta
                @ (self.beta_perpendicular[None, :] * state)
            )
        return result

    def _interaction_bound(self, field):
        bound = abs(field) * np.max(np.abs(self.dipole)) * np.linalg.norm(
            self.angular.cos_theta, 2
        )
        bound += 0.5 * field**2 * (
            np.max(np.abs(self.alpha_perpendicular))
            + np.max(np.abs(self.alpha_anisotropy))
            * np.linalg.norm(self.angular.cos_theta_squared, 2)
        )
        bound += abs(field) ** 3 / 6 * (
            np.max(np.abs(self.beta_parallel))
            * np.linalg.norm(self.angular.cos_theta_cubed, 2)
            + 3
            * np.max(np.abs(self.beta_perpendicular))
            * np.linalg.norm(self.angular.sin_theta_squared_cos_theta, 2)
        )
        return float(bound)

    def _observables(self, state):
        probability = np.abs(state) ** 2
        norm = float(np.sum(probability).real)
        expectation_r = float(np.sum(probability * self.r[None, :]).real)

        def angular_expectation(matrix, radial_factor=1.0):
            value = np.einsum(
                "lr,lk,kr,r->",
                state.conj(),
                matrix,
                state,
                np.broadcast_to(radial_factor, self.r.shape),
            )
            return float(np.real_if_close(value).real)

        orientation = angular_expectation(self.angular.cos_theta)
        alignment = angular_expectation(self.angular.cos_theta_squared)
        expectation_z = angular_expectation(self.angular.cos_theta, self.r)
        expectation_r_cos_theta_squared = angular_expectation(
            self.angular.cos_theta_squared, self.r
        )
        return (
            norm,
            expectation_r,
            orientation,
            alignment,
            expectation_z,
            expectation_r_cos_theta_squared,
        )

    def _radial_probability(self, state):
        r"""Return ``P(R) = R^2 integral |Psi(R, Omega)|^2 dOmega``.

        The propagated state contains quadrature-weighted samples of the
        reduced radial amplitudes ``u_l(R)`` in
        ``Psi = sum_l u_l(R) Y_l0(Omega) / R``. Orthogonality of the spherical
        harmonics therefore gives ``P(R) = sum_l |u_l(R)|^2``.
        """
        reduced_radial_state = state / self._sqrt_weights[None, :]
        return np.sum(np.abs(reduced_radial_state) ** 2, axis=0)

    def run(self, *, progress=False) -> VibrationalRotorResult:
        """Propagate the initial state and return sampled observables."""
        state = self.initial_state.copy()
        indices = list(range(0, self.n_steps + 1, self.sample_interval))
        if indices[-1] != self.n_steps:
            indices.append(self.n_steps)
        sample_set = set(indices)
        sampled = {
            "times": [],
            "electric_field": [],
            "norms": [],
            "expectation_r": [],
            "expectation_cos_theta": [],
            "expectation_cos_theta_squared": [],
            "expectation_z": [],
            "expectation_r_cos_theta_squared": [],
            "radial_probability": [],
            "solver_iterations": [],
            "residual_norms": [],
            "solver_methods": [],
        }

        def record(step, iterations, residual, method):
            time = self.initial_time + step * self.time_step
            (
                norm,
                mean_r,
                orientation,
                alignment,
                mean_z,
                mean_r_cos_theta_squared,
            ) = self._observables(state)
            sampled["times"].append(time)
            sampled["electric_field"].append(self.electric_field(time))
            sampled["norms"].append(norm)
            sampled["expectation_r"].append(mean_r)
            sampled["expectation_cos_theta"].append(orientation)
            sampled["expectation_cos_theta_squared"].append(alignment)
            sampled["expectation_z"].append(mean_z)
            sampled["expectation_r_cos_theta_squared"].append(
                mean_r_cos_theta_squared
            )
            sampled["radial_probability"].append(self._radial_probability(state))
            sampled["solver_iterations"].append(iterations)
            sampled["residual_norms"].append(residual)
            sampled["solver_methods"].append(method)

        record(0, 0, 0.0, "initial")
        iterator = range(self.n_steps)
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc="Vibrational rotor")
        for step in iterator:
            time = self.initial_time + step * self.time_step
            state, diagnostics = self.propagator.step(state, time)
            if step + 1 in sample_set:
                record(
                    step + 1,
                    diagnostics.iterations,
                    diagnostics.residual_norm,
                    diagnostics.method,
                )

        return VibrationalRotorResult(
            times=np.asarray(sampled["times"]),
            electric_field=np.asarray(sampled["electric_field"]),
            norms=np.asarray(sampled["norms"]),
            expectation_r=np.asarray(sampled["expectation_r"]),
            expectation_cos_theta=np.asarray(sampled["expectation_cos_theta"]),
            expectation_cos_theta_squared=np.asarray(
                sampled["expectation_cos_theta_squared"]
            ),
            expectation_z=np.asarray(sampled["expectation_z"]),
            expectation_r_cos_theta_squared=np.asarray(
                sampled["expectation_r_cos_theta_squared"]
            ),
            radial_probability=np.asarray(sampled["radial_probability"]),
            solver_iterations=np.asarray(sampled["solver_iterations"], dtype=int),
            residual_norms=np.asarray(sampled["residual_norms"]),
            solver_methods=np.asarray(sampled["solver_methods"]),
            final_state=state / self._sqrt_weights[None, :],
            radial_grid=self.r.copy(),
            radial_weights=self.radial_weights.copy(),
            ell=self.ell.copy(),
            time_step=self.time_step,
            n_steps=self.n_steps,
        )
