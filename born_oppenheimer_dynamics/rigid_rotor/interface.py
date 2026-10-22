"""High-level time-dependent rigid-rotor interface."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

import numpy as np

from .._rotor import RotorFieldInteraction, angular_momenta, select_angular_sector
from ..vibrational_rotor.angular import rotor_angular_operators
from ..vibrational_rotor.propagator import SplitCrankNicolson
from ..vibrational_rotor.surfaces import DiatomicSurfaces
from ..vibrational_rotor.systems import DiatomicMolecule
from .result import RigidRotorResult


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


@dataclass(frozen=True)
class RotationalState:
    """Field-free rigid-rotor angular-momentum quantum number."""

    rotational: int = 0

    def __post_init__(self):
        object.__setattr__(
            self,
            "rotational",
            _integer(self.rotational, "rotational quantum number"),
        )


def _state_specification(initial_state):
    if isinstance(initial_state, str):
        if initial_state == "groundstate":
            return RotationalState()
        raise ValueError("initial_state string must be 'groundstate'.")
    if isinstance(initial_state, RotationalState):
        return initial_state
    if isinstance(initial_state, (int, np.integer)) and not isinstance(
        initial_state, (bool, np.bool_)
    ):
        return RotationalState(initial_state)
    return None


class RigidRotorDiatomic:
    r"""Time-dependent rigid-rotor dynamics for a diatomic molecule.

    The orientation is expanded in ``Y_l0`` and the electric field is linearly
    polarized along the laboratory z axis. The field-free Hamiltonian is
    ``B J**2``. Supply either ``rotational_constant`` directly in atomic units,
    or ``bond_length`` so that ``B = 1 / (2 mu R**2)`` is inferred from the
    isotope masses. The length-gauge interaction is

    ``-mu*E*cos(theta) - E**2*(alpha_perp + Delta alpha*cos(theta)**2)/2``

    plus the optional first-hyperpolarizability contribution

    ``-E**3*(beta_parallel*cos(theta)**3 +
    3*beta_perp*sin(theta)**2*cos(theta))/6``.

    Electronic energy at the fixed geometry is an irrelevant scalar phase
    and is therefore omitted from the Hamiltonian.
    """

    def __init__(
        self,
        molecule,
        *,
        bond_length=None,
        rotational_constant=None,
        polarizability_parallel,
        polarizability_perpendicular,
        dipole=None,
        beta_parallel=None,
        beta_perpendicular=None,
        l_max=12,
        angular_sector="auto",
        initial_state="groundstate",
        fields=None,
        initial_time=0.0,
        final_time=None,
        time_step=None,
        integrator_options=None,
        sample_interval=1,
    ):
        if not isinstance(molecule, DiatomicMolecule):
            molecule = DiatomicMolecule(tuple(molecule))
        self.molecule = molecule
        if (bond_length is None) == (rotational_constant is None):
            raise ValueError(
                "Supply exactly one of bond_length and rotational_constant."
            )
        if rotational_constant is None:
            self.bond_length = _positive_real(bond_length, "bond_length")
            self.rotational_constant = 1.0 / (
                2 * self.molecule.reduced_mass * self.bond_length**2
            )
        else:
            self.bond_length = None
            self.rotational_constant = _positive_real(
                rotational_constant, "rotational_constant"
            )
        self.l_max = _integer(l_max, "l_max")
        self.initial_state_specification = _state_specification(initial_state)
        self.fields = self._field_terms(fields)

        self.alpha_parallel = _finite_real(
            polarizability_parallel, "polarizability_parallel"
        )
        self.alpha_perpendicular = _finite_real(
            polarizability_perpendicular, "polarizability_perpendicular"
        )
        if dipole is None:
            if not molecule.homonuclear:
                raise ValueError(
                    "A permanent dipole is required for a heteronuclear molecule."
                )
            dipole = 0.0
        self.dipole = _finite_real(dipole, "dipole")
        if (beta_parallel is None) != (beta_perpendicular is None):
            raise ValueError(
                "beta_parallel and beta_perpendicular must either both be "
                "supplied or both be omitted."
            )
        self.beta_parallel = (
            0.0
            if beta_parallel is None
            else _finite_real(beta_parallel, "beta_parallel")
        )
        self.beta_perpendicular = (
            0.0
            if beta_perpendicular is None
            else _finite_real(beta_perpendicular, "beta_perpendicular")
        )
        self.alpha_anisotropy = self.alpha_parallel - self.alpha_perpendicular
        self.has_odd_interaction = bool(
            self.dipole != 0
            or self.beta_parallel != 0
            or self.beta_perpendicular != 0
        )

        initial_rotational = (
            None
            if self.initial_state_specification is None
            else self.initial_state_specification.rotational
        )
        self.angular_sector = select_angular_sector(
            angular_sector, initial_rotational, self.has_odd_interaction
        )
        self.ell = angular_momenta(self.l_max, self.angular_sector)
        self.angular = rotor_angular_operators(self.l_max, self.ell)
        self.field_interaction = RotorFieldInteraction(
            self.angular,
            np.asarray([self.dipole]),
            np.asarray([self.alpha_parallel]),
            np.asarray([self.alpha_perpendicular]),
            np.asarray([self.beta_parallel]),
            np.asarray([self.beta_perpendicular]),
        )

        energies = self.rotational_constant * self.ell * (self.ell + 1)
        self.field_free_blocks = energies[:, None, None]
        self.initial_state = self._initial_state(initial_state)

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
            self.field_free_blocks,
            time_step=self.time_step,
            field=self.electric_field,
            interaction=self._interaction,
            interaction_bound=self._interaction_bound,
            **options,
        )

    @classmethod
    def from_surfaces(cls, molecule, surfaces, *, bond_length, **options):
        """Construct a rigid rotor from response curves at one bond length.

        The electronic energy is not used because it contributes only a
        global phase at fixed internuclear distance.
        """
        if not isinstance(surfaces, DiatomicSurfaces):
            raise TypeError("surfaces must be a DiatomicSurfaces instance.")
        distance = _positive_real(bond_length, "bond_length")
        tolerance = 100 * np.finfo(float).eps * max(1.0, surfaces.r_max)
        if (
            distance < surfaces.r_min - tolerance
            or distance > surfaces.r_max + tolerance
        ):
            raise ValueError(
                f"bond_length must lie within the supplied electronic data "
                f"[{surfaces.r_min:.12g}, {surfaces.r_max:.12g}] bohr."
            )
        interpolated = surfaces.interpolate(np.asarray([distance]))

        def scalar(values):
            return None if values is None else float(values[0])

        return cls(
            molecule,
            bond_length=distance,
            dipole=scalar(interpolated.dipole),
            polarizability_parallel=scalar(interpolated.polarizability_parallel),
            polarizability_perpendicular=scalar(
                interpolated.polarizability_perpendicular
            ),
            beta_parallel=scalar(interpolated.beta_parallel),
            beta_perpendicular=scalar(interpolated.beta_perpendicular),
            **options,
        )

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
                    "RigidRotorDiatomic supports length-gauge fields only."
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

    def _initial_state(self, initial_state):
        specification = self.initial_state_specification
        if specification is not None:
            if specification.rotational > self.l_max:
                raise ValueError("The initial rotational quantum number exceeds l_max.")
            matches = np.flatnonzero(self.ell == specification.rotational)
            if len(matches) != 1:
                raise ValueError("The selected angular sector omits the initial state.")
            state = np.zeros((len(self.ell), 1), dtype=np.complex128)
            state[int(matches[0]), 0] = 1.0
        else:
            values = np.asarray(initial_state)
            expected = (len(self.ell),)
            if values.shape != expected or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"A custom initial_state must be finite with shape {expected}."
                )
            state = np.asarray(values, dtype=np.complex128)[:, None]
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
        return self.field_interaction.apply(state, field)

    def _interaction_bound(self, field):
        return self.field_interaction.bound(field)

    def _observables(self, state):
        coefficients = state[:, 0]
        populations = np.abs(coefficients) ** 2
        norm = float(np.sum(populations).real)
        orientation = np.vdot(
            coefficients, self.angular.cos_theta @ coefficients
        )
        alignment = np.vdot(
            coefficients, self.angular.cos_theta_squared @ coefficients
        )
        return (
            norm,
            float(np.real_if_close(orientation).real),
            float(np.real_if_close(alignment).real),
            populations,
        )

    def run(self, *, progress=False) -> RigidRotorResult:
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
            "expectation_cos_theta": [],
            "expectation_cos_theta_squared": [],
            "angular_momentum_populations": [],
            "solver_iterations": [],
            "residual_norms": [],
            "solver_methods": [],
        }

        def record(step, iterations, residual, method):
            time = self.initial_time + step * self.time_step
            norm, orientation, alignment, populations = self._observables(state)
            sampled["times"].append(time)
            sampled["electric_field"].append(self.electric_field(time))
            sampled["norms"].append(norm)
            sampled["expectation_cos_theta"].append(orientation)
            sampled["expectation_cos_theta_squared"].append(alignment)
            sampled["angular_momentum_populations"].append(populations)
            sampled["solver_iterations"].append(iterations)
            sampled["residual_norms"].append(residual)
            sampled["solver_methods"].append(method)

        record(0, 0, 0.0, "initial")
        iterator = range(self.n_steps)
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc="Rigid rotor")
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

        return RigidRotorResult(
            times=np.asarray(sampled["times"]),
            electric_field=np.asarray(sampled["electric_field"]),
            norms=np.asarray(sampled["norms"]),
            expectation_cos_theta=np.asarray(sampled["expectation_cos_theta"]),
            expectation_cos_theta_squared=np.asarray(
                sampled["expectation_cos_theta_squared"]
            ),
            angular_momentum_populations=np.asarray(
                sampled["angular_momentum_populations"]
            ),
            solver_iterations=np.asarray(sampled["solver_iterations"], dtype=int),
            residual_norms=np.asarray(sampled["residual_norms"]),
            solver_methods=np.asarray(sampled["solver_methods"]),
            final_state=state[:, 0].copy(),
            ell=self.ell.copy(),
            bond_length=self.bond_length,
            rotational_constant=self.rotational_constant,
            time_step=self.time_step,
            n_steps=self.n_steps,
        )
