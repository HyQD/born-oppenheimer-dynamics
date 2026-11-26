"""One-electron H2+ electronic states in Gaussian basis sets."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from numbers import Real
from types import MappingProxyType

import numpy as np
from scipy.linalg import eigh

from .grids import radial_grid_from_range
from .results import DiatomicElectronicData


def _finite_positive_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive real scalar.")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real scalar.")
    return float(value)


def _load_optional_dependencies():
    try:
        import basis_set_exchange as bse
        from pyscf import gto
    except ImportError as error:
        raise ImportError(
            "The H2+ GTO backend requires the optional PySCF dependencies. "
            'Install them with: python -m pip install -e ".[pyscf]".'
        ) from error
    return bse, gto


@dataclass(frozen=True)
class _CanonicalEigenproblem:
    energies: np.ndarray
    coefficients: np.ndarray
    overlap_eigenvalues: np.ndarray
    retained_overlap_eigenvalues: np.ndarray
    orthogonalizer: np.ndarray
    orthogonal_hamiltonian: np.ndarray
    orthogonal_coefficients: np.ndarray


def _canonical_eigh(hamiltonian, overlap, threshold):
    """Solve a generalized Hermitian eigenproblem after overlap filtering."""
    hamiltonian = np.asarray(hamiltonian)
    overlap = np.asarray(overlap)
    if (
        hamiltonian.ndim != 2
        or hamiltonian.shape[0] != hamiltonian.shape[1]
        or overlap.shape != hamiltonian.shape
    ):
        raise ValueError("hamiltonian and overlap must be square matrices.")
    if (
        np.iscomplexobj(hamiltonian)
        or np.iscomplexobj(overlap)
        or not np.all(np.isfinite(hamiltonian))
        or not np.all(np.isfinite(overlap))
    ):
        raise ValueError("hamiltonian and overlap must be finite real matrices.")
    if not np.allclose(hamiltonian, hamiltonian.T, rtol=0.0, atol=1e-12):
        raise ValueError("hamiltonian must be symmetric.")
    if not np.allclose(overlap, overlap.T, rtol=0.0, atol=1e-12):
        raise ValueError("overlap must be symmetric.")
    threshold = _finite_positive_real(threshold, "linear_dependence_threshold")

    hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)
    overlap = 0.5 * (overlap + overlap.T)
    overlap_eigenvalues, overlap_eigenvectors = eigh(overlap)
    retained = overlap_eigenvalues > threshold
    if not np.any(retained):
        raise ValueError(
            "All AO combinations were removed by the linear-dependence "
            "threshold."
        )

    retained_values = overlap_eigenvalues[retained]
    orthogonalizer = (
        overlap_eigenvectors[:, retained] / np.sqrt(retained_values)[None, :]
    )
    orthogonal_hamiltonian = np.einsum(
        "pi,pq,qj->ij",
        orthogonalizer,
        hamiltonian,
        orthogonalizer,
        optimize=True,
    )
    orthogonal_hamiltonian = 0.5 * (
        orthogonal_hamiltonian + orthogonal_hamiltonian.T
    )
    energies, orthogonal_coefficients = eigh(orthogonal_hamiltonian)
    coefficients = orthogonalizer @ orthogonal_coefficients

    arrays = (
        energies,
        coefficients,
        overlap_eigenvalues,
        retained_values,
        orthogonalizer,
        orthogonal_hamiltonian,
        orthogonal_coefficients,
    )
    frozen = []
    for values in arrays:
        result = np.array(values, dtype=float, copy=True)
        result.setflags(write=False)
        frozen.append(result)
    return _CanonicalEigenproblem(*frozen)


@dataclass(frozen=True)
class _H2PlusElectronicPoint:
    internuclear_distance: float
    electronic_energy: float
    orbital_energies: np.ndarray
    dipole_matrix: np.ndarray
    minimum_overlap_eigenvalue: float
    minimum_retained_overlap_eigenvalue: float
    discarded_overlap_modes: int
    retained_electronic_dimension: int
    orthonormality_error: float
    projected_eigenproblem_residual: float

    def __post_init__(self):
        for name in ("orbital_energies", "dipole_matrix"):
            values = np.asarray(getattr(self, name))
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values.")
            result = np.array(values, dtype=float, copy=True)
            result.setflags(write=False)
            object.__setattr__(self, name, result)


@dataclass(frozen=True)
class _H2PlusAOIntegrals:
    molecule: object
    hamiltonian: np.ndarray
    overlap: np.ndarray
    position: np.ndarray


class _H2PlusGTOBackend:
    """Resolve one GTO basis and solve H2+ at arbitrary bond lengths."""

    def __init__(self, basis, *, linear_dependence_threshold=1e-8):
        if not isinstance(basis, str):
            raise TypeError("basis must be a non-empty string.")
        basis = basis.strip()
        if not basis:
            raise ValueError("basis must be a non-empty string.")
        self.linear_dependence_threshold = _finite_positive_real(
            linear_dependence_threshold, "linear_dependence_threshold"
        )
        bse, gto = _load_optional_dependencies()
        try:
            basis_metadata = bse.get_basis(basis, elements=["H"])
            basis_text = bse.get_basis(
                basis,
                elements=["H"],
                fmt="nwchem",
                header=False,
            )
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"Could not obtain a hydrogen basis named {basis!r} "
                "from Basis Set Exchange."
            ) from error
        try:
            parsed_basis = gto.basis.parse(basis_text)
        except Exception as error:
            raise ValueError(
                f"PySCF could not parse the Basis Set Exchange basis {basis!r}."
            ) from error

        self._gto = gto
        self._parsed_basis = parsed_basis
        self.basis_metadata = MappingProxyType(
            {
                "requested_name": basis,
                "name": str(basis_metadata["name"]),
                "version": str(basis_metadata["version"]),
                "revision_description": str(
                    basis_metadata.get("revision_description", "")
                ),
                "pyscf_version": version("pyscf"),
                "basis_set_exchange_version": version("basis-set-exchange"),
            }
        )

    def build_molecule(self, internuclear_distance):
        """Build a centered, one-electron H2+ PySCF molecule."""
        distance = _finite_positive_real(
            internuclear_distance, "internuclear_distance"
        )
        return self._gto.M(
            atom=[
                ("H", (0.0, 0.0, -distance / 2)),
                ("H", (0.0, 0.0, distance / 2)),
            ],
            basis={"H": self._parsed_basis},
            unit="Bohr",
            charge=1,
            spin=1,
            cart=False,
            verbose=0,
        )

    def solve(self, internuclear_distance):
        """Return the complete retained finite-basis spectrum at one distance."""
        integrals = self.ao_integrals(internuclear_distance)
        molecule = integrals.molecule
        distance = float(internuclear_distance)
        overlap = integrals.overlap
        hamiltonian = integrals.hamiltonian
        position = integrals.position

        solution = _canonical_eigh(
            hamiltonian,
            overlap,
            self.linear_dependence_threshold,
        )
        coefficients = solution.coefficients
        electronic_dipole = -np.einsum(
            "pi,xpq,qj->xij",
            coefficients,
            position,
            coefficients,
            optimize=True,
        )
        electronic_dipole = 0.5 * (
            electronic_dipole + electronic_dipole.transpose(0, 2, 1)
        )

        identity = np.eye(len(solution.energies))
        orthonormality_error = np.linalg.norm(
            coefficients.T @ overlap @ coefficients - identity
        )
        residual = (
            hamiltonian @ coefficients
            - (overlap @ coefficients) * solution.energies[None, :]
        )
        projected_eigenproblem_residual = np.linalg.norm(
            coefficients.T @ residual
        )
        discarded = (
            len(solution.overlap_eigenvalues)
            - len(solution.retained_overlap_eigenvalues)
        )

        return _H2PlusElectronicPoint(
            internuclear_distance=distance,
            electronic_energy=float(
                solution.energies[0] + molecule.energy_nuc()
            ),
            orbital_energies=solution.energies,
            dipole_matrix=electronic_dipole,
            minimum_overlap_eigenvalue=float(
                solution.overlap_eigenvalues[0]
            ),
            minimum_retained_overlap_eigenvalue=float(
                solution.retained_overlap_eigenvalues[0]
            ),
            discarded_overlap_modes=discarded,
            retained_electronic_dimension=len(solution.energies),
            orthonormality_error=float(orthonormality_error),
            projected_eigenproblem_residual=float(
                projected_eigenproblem_residual
            ),
        )

    def ao_integrals(self, internuclear_distance):
        """Return AO matrices at one internuclear distance."""
        molecule = self.build_molecule(internuclear_distance)
        overlap = molecule.intor_symmetric("int1e_ovlp")
        hamiltonian = (
            molecule.intor_symmetric("int1e_kin")
            + molecule.intor_symmetric("int1e_nuc")
        )
        with molecule.with_common_orig((0.0, 0.0, 0.0)):
            position = molecule.intor_symmetric("int1e_r", comp=3)
        arrays = []
        for values in (hamiltonian, overlap, position):
            result = np.array(values, dtype=float, copy=True)
            result.setflags(write=False)
            arrays.append(result)
        return _H2PlusAOIntegrals(molecule, *arrays)


def _nonnegative_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative real scalar.")
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative real scalar.")
    return float(value)


def _frequency_array(frequencies):
    if isinstance(frequencies, Real) and not isinstance(
        frequencies, (bool, np.bool_)
    ):
        values = np.asarray([frequencies])
    else:
        try:
            values = np.asarray(tuple(frequencies))
        except TypeError as error:
            raise TypeError(
                "frequencies must be a real scalar or an iterable of real "
                "scalars."
            ) from error
    if (
        values.ndim != 1
        or len(values) == 0
        or np.iscomplexobj(values)
        or values.dtype.kind not in "iuf"
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "frequencies must be a non-empty finite real one-dimensional array."
        )
    values = np.asarray(values, dtype=float)
    if np.any(values < 0):
        raise ValueError("frequencies must be non-negative.")
    values = np.sort(values)
    if np.any(np.diff(values) == 0):
        raise ValueError("frequencies must not contain duplicates.")
    values.setflags(write=False)
    return values


def _transition_diagnostics(
    gaps,
    transition_dipoles,
    frequency,
    strength_threshold,
):
    strengths = np.abs(transition_dipoles) ** 2
    closest_energy = np.zeros(3)
    closest_strength = np.zeros(3)
    closest_index = np.full(3, -1, dtype=int)
    absolute_detuning = np.zeros(3)
    relative_detuning = np.zeros(3)
    has_allowed = np.zeros(3, dtype=bool)

    for component in range(3):
        allowed = strengths[component] > strength_threshold
        if not np.any(allowed):
            continue
        allowed_indices = np.flatnonzero(allowed)
        detuning = np.abs(gaps[allowed] - frequency)
        local_index = int(np.argmin(detuning))
        state_index = int(allowed_indices[local_index])
        gap = float(gaps[state_index])
        has_allowed[component] = True
        closest_energy[component] = gap
        closest_strength[component] = strengths[component, state_index]
        closest_index[component] = state_index + 1
        absolute_detuning[component] = detuning[local_index]
        relative_detuning[component] = detuning[local_index] / max(
            gap, frequency, np.finfo(float).tiny
        )

    return {
        "closest_transition_energy": closest_energy,
        "closest_transition_strength": closest_strength,
        "closest_transition_index": closest_index,
        "absolute_detuning": absolute_detuning,
        "relative_detuning": relative_detuning,
        "has_dipole_allowed_transition": has_allowed,
    }


def _sum_over_states_polarizability(
    point,
    frequencies,
    *,
    transition_strength_threshold,
    resonance_atol,
    resonance_rtol,
    geometry_label=None,
):
    gaps = point.orbital_energies[1:] - point.orbital_energies[0]
    if len(gaps) == 0:
        raise ValueError(
            "The retained basis contains no excited state for the response."
        )
    transition_dipoles = point.dipole_matrix[:, 0, 1:]
    strengths = np.abs(transition_dipoles) ** 2
    active_states = np.any(strengths > transition_strength_threshold, axis=0)

    tensors = np.zeros((len(frequencies), 3, 3))
    diagnostics = []
    labels = ("x", "y", "z")
    if geometry_label is None:
        geometry_label = f"at R={point.internuclear_distance:.12g}"
    for frequency_index, frequency in enumerate(frequencies):
        values = _transition_diagnostics(
            gaps,
            transition_dipoles,
            frequency,
            transition_strength_threshold,
        )
        for component, has_allowed in enumerate(
            values["has_dipole_allowed_transition"]
        ):
            if not has_allowed:
                continue
            detuning = values["absolute_detuning"][component]
            transition_energy = values["closest_transition_energy"][component]
            tolerance = resonance_atol + resonance_rtol * max(
                transition_energy, frequency
            )
            if detuning <= tolerance:
                raise ValueError(
                    "Requested frequency is resonant with a dipole-allowed "
                    f"transition {geometry_label}: "
                    f"omega={frequency:.12g}, "
                    f"DeltaE={transition_energy:.12g}, "
                    f"detuning={detuning:.3e}, component={labels[component]}, "
                    "state="
                    f"{values['closest_transition_index'][component]}."
                )

        detuning = np.abs(gaps - frequency)
        tolerance = resonance_atol + resonance_rtol * np.maximum(
            gaps, frequency
        )
        singular = detuning <= tolerance
        if np.any(singular & active_states):
            raise RuntimeError(
                "Internal resonance detection failed for an active transition."
            )
        denominator = gaps**2 - frequency**2
        factors = np.zeros_like(gaps)
        regular = ~singular
        factors[regular] = 2 * gaps[regular] / denominator[regular]
        tensor = np.einsum(
            "n,in,jn->ij",
            factors,
            transition_dipoles,
            transition_dipoles,
            optimize=True,
        )
        tensors[frequency_index] = 0.5 * (tensor + tensor.T)
        diagnostics.append(values)
    return tensors, diagnostics


def _off_diagonal_maximum(tensor):
    off_diagonal = np.array(tensor, copy=True)
    diagonal = np.arange(3)
    off_diagonal[..., diagonal, diagonal] = 0.0
    return np.max(np.abs(off_diagonal), axis=(-2, -1))


def compute_h2plus_gto(
    *,
    basis,
    radial_range,
    frequencies=0.0,
    linear_dependence_threshold=1e-8,
    transition_strength_threshold=1e-12,
    resonance_atol=1e-8,
    resonance_rtol=1e-6,
    progress=False,
):
    """Compute H2+ BO energies and undamped GTO polarizabilities."""
    radial_grid = radial_grid_from_range(radial_range)
    frequency_values = _frequency_array(frequencies)
    transition_strength_threshold = _nonnegative_real(
        transition_strength_threshold, "transition_strength_threshold"
    )
    resonance_atol = _nonnegative_real(resonance_atol, "resonance_atol")
    resonance_rtol = _nonnegative_real(resonance_rtol, "resonance_rtol")
    if resonance_atol == resonance_rtol == 0:
        raise ValueError(
            "At least one of resonance_atol and resonance_rtol must be positive."
        )
    if not isinstance(progress, (bool, np.bool_)):
        raise TypeError("progress must be a boolean.")

    backend = _H2PlusGTOBackend(
        basis,
        linear_dependence_threshold=linear_dependence_threshold,
    )
    n_frequency = len(frequency_values)
    n_distance = len(radial_grid)
    electronic_energy = np.empty(n_distance)
    polarizability = np.empty((n_frequency, n_distance, 3, 3))
    ionization_threshold = np.empty(n_distance)
    minimum_excitation_energy = np.empty(n_distance)
    minimum_overlap_eigenvalue = np.empty(n_distance)
    minimum_retained_overlap_eigenvalue = np.empty(n_distance)
    discarded_overlap_modes = np.empty(n_distance, dtype=int)
    retained_electronic_dimension = np.empty(n_distance, dtype=int)
    orthonormality_error = np.empty(n_distance)
    projected_eigenproblem_residual = np.empty(n_distance)
    ground_state_dipole = np.empty((n_distance, 3))

    diagnostic_shape = (n_frequency, n_distance, 3)
    closest_transition_energy = np.zeros(diagnostic_shape)
    closest_transition_strength = np.zeros(diagnostic_shape)
    closest_transition_index = np.full(diagnostic_shape, -1, dtype=int)
    absolute_detuning = np.zeros(diagnostic_shape)
    relative_detuning = np.zeros(diagnostic_shape)
    has_dipole_allowed_transition = np.zeros(diagnostic_shape, dtype=bool)

    iterator = enumerate(radial_grid)
    if progress:
        from tqdm.auto import tqdm

        iterator = enumerate(tqdm(radial_grid, desc="H2+ GTO properties"))
    for distance_index, distance in iterator:
        point = backend.solve(distance)
        threshold = -point.orbital_energies[0]
        if threshold <= 0:
            raise ValueError(
                "The finite-basis H2+ ground state is not bound at "
                f"R={distance:.12g}."
            )
        invalid = frequency_values >= threshold
        if np.any(invalid):
            frequency = frequency_values[np.flatnonzero(invalid)[0]]
            raise ValueError(
                "Requested frequency reaches the ionization continuum at "
                f"R={distance:.12g}: omega={frequency:.12g}, "
                f"threshold={threshold:.12g}."
            )

        tensor, point_diagnostics = _sum_over_states_polarizability(
            point,
            frequency_values,
            transition_strength_threshold=transition_strength_threshold,
            resonance_atol=resonance_atol,
            resonance_rtol=resonance_rtol,
        )
        electronic_energy[distance_index] = point.electronic_energy
        polarizability[:, distance_index] = tensor
        ionization_threshold[distance_index] = threshold
        minimum_excitation_energy[distance_index] = (
            point.orbital_energies[1] - point.orbital_energies[0]
        )
        minimum_overlap_eigenvalue[distance_index] = (
            point.minimum_overlap_eigenvalue
        )
        minimum_retained_overlap_eigenvalue[distance_index] = (
            point.minimum_retained_overlap_eigenvalue
        )
        discarded_overlap_modes[distance_index] = (
            point.discarded_overlap_modes
        )
        retained_electronic_dimension[distance_index] = (
            point.retained_electronic_dimension
        )
        orthonormality_error[distance_index] = point.orthonormality_error
        projected_eigenproblem_residual[distance_index] = (
            point.projected_eigenproblem_residual
        )
        ground_state_dipole[distance_index] = point.dipole_matrix[:, 0, 0]
        for frequency_index, values in enumerate(point_diagnostics):
            index = (frequency_index, distance_index)
            closest_transition_energy[index] = values[
                "closest_transition_energy"
            ]
            closest_transition_strength[index] = values[
                "closest_transition_strength"
            ]
            closest_transition_index[index] = values[
                "closest_transition_index"
            ]
            absolute_detuning[index] = values["absolute_detuning"]
            relative_detuning[index] = values["relative_detuning"]
            has_dipole_allowed_transition[index] = values[
                "has_dipole_allowed_transition"
            ]

    minimum_absolute_detuning = np.zeros((n_frequency, 3))
    minimum_relative_detuning = np.zeros((n_frequency, 3))
    for frequency_index in range(n_frequency):
        for component in range(3):
            allowed = has_dipole_allowed_transition[
                frequency_index, :, component
            ]
            if np.any(allowed):
                minimum_absolute_detuning[frequency_index, component] = np.min(
                    absolute_detuning[frequency_index, allowed, component]
                )
                minimum_relative_detuning[frequency_index, component] = np.min(
                    relative_detuning[frequency_index, allowed, component]
                )

    diagnostics = {
        "ionization_threshold": ionization_threshold,
        "minimum_excitation_energy": minimum_excitation_energy,
        "minimum_overlap_eigenvalue": minimum_overlap_eigenvalue,
        "minimum_retained_overlap_eigenvalue": (
            minimum_retained_overlap_eigenvalue
        ),
        "discarded_overlap_modes": discarded_overlap_modes,
        "retained_electronic_dimension": retained_electronic_dimension,
        "orthonormality_error": orthonormality_error,
        "projected_eigenproblem_residual": projected_eigenproblem_residual,
        "ground_state_dipole": ground_state_dipole,
        "closest_transition_energy": closest_transition_energy,
        "closest_transition_strength": closest_transition_strength,
        "closest_transition_index": closest_transition_index,
        "absolute_detuning": absolute_detuning,
        "relative_detuning": relative_detuning,
        "has_dipole_allowed_transition": has_dipole_allowed_transition,
        "minimum_absolute_detuning": minimum_absolute_detuning,
        "minimum_relative_detuning": minimum_relative_detuning,
        "cylindrical_symmetry_error": np.abs(
            polarizability[..., 0, 0] - polarizability[..., 1, 1]
        ),
        "maximum_off_diagonal_polarizability": _off_diagonal_maximum(
            polarizability
        ),
    }
    metadata = {
        "backend": "pyscf-gto-one-electron",
        "system": "H2+",
        "charge": 1,
        "spin": 1,
        "units": "atomic",
        "geometry": "H 0 0 -R/2; H 0 0 R/2",
        "basis": dict(backend.basis_metadata),
        "linear_dependence_threshold": backend.linear_dependence_threshold,
        "transition_strength_threshold": transition_strength_threshold,
        "resonance_atol": resonance_atol,
        "resonance_rtol": resonance_rtol,
        "response": "undamped sum over states",
    }
    return DiatomicElectronicData(
        internuclear_distance=radial_grid,
        frequencies=frequency_values,
        electronic_energy=electronic_energy,
        polarizability=polarizability,
        diagnostics=diagnostics,
        metadata=metadata,
    )


def compute_hydrogen_gto_polarizability(
    *,
    basis,
    frequencies=0.0,
    linear_dependence_threshold=1e-8,
    transition_strength_threshold=1e-12,
    resonance_atol=1e-8,
    resonance_rtol=1e-6,
):
    """Return the isolated-H polarizability tensor in a GTO basis.

    The returned array has shape ``(n_frequency, 3, 3)``. Frequencies are
    sorted into increasing order using the same convention as
    :func:`compute_h2plus_gto`.
    """
    frequency_values = _frequency_array(frequencies)
    transition_strength_threshold = _nonnegative_real(
        transition_strength_threshold, "transition_strength_threshold"
    )
    resonance_atol = _nonnegative_real(resonance_atol, "resonance_atol")
    resonance_rtol = _nonnegative_real(resonance_rtol, "resonance_rtol")
    if resonance_atol == resonance_rtol == 0:
        raise ValueError(
            "At least one of resonance_atol and resonance_rtol must be positive."
        )

    backend = _H2PlusGTOBackend(
        basis,
        linear_dependence_threshold=linear_dependence_threshold,
    )
    molecule = backend._gto.M(
        atom=[("H", (0.0, 0.0, 0.0))],
        basis={"H": backend._parsed_basis},
        unit="Bohr",
        charge=0,
        spin=1,
        cart=False,
        verbose=0,
    )
    overlap = molecule.intor_symmetric("int1e_ovlp")
    hamiltonian = (
        molecule.intor_symmetric("int1e_kin")
        + molecule.intor_symmetric("int1e_nuc")
    )
    with molecule.with_common_orig((0.0, 0.0, 0.0)):
        position = molecule.intor_symmetric("int1e_r", comp=3)
    solution = _canonical_eigh(
        hamiltonian,
        overlap,
        backend.linear_dependence_threshold,
    )
    threshold = -solution.energies[0]
    invalid = frequency_values >= threshold
    if np.any(invalid):
        frequency = frequency_values[np.flatnonzero(invalid)[0]]
        raise ValueError(
            "Requested frequency reaches the ionization continuum for "
            f"isolated H: omega={frequency:.12g}, "
            f"threshold={threshold:.12g}."
        )

    dipole_matrix = -np.einsum(
        "pi,xpq,qj->xij",
        solution.coefficients,
        position,
        solution.coefficients,
        optimize=True,
    )
    point = _H2PlusElectronicPoint(
        internuclear_distance=0.0,
        electronic_energy=float(solution.energies[0]),
        orbital_energies=solution.energies,
        dipole_matrix=dipole_matrix,
        minimum_overlap_eigenvalue=float(solution.overlap_eigenvalues[0]),
        minimum_retained_overlap_eigenvalue=float(
            solution.retained_overlap_eigenvalues[0]
        ),
        discarded_overlap_modes=(
            len(solution.overlap_eigenvalues)
            - len(solution.retained_overlap_eigenvalues)
        ),
        retained_electronic_dimension=len(solution.energies),
        orthonormality_error=0.0,
        projected_eigenproblem_residual=0.0,
    )
    polarizability, _ = _sum_over_states_polarizability(
        point,
        frequency_values,
        transition_strength_threshold=transition_strength_threshold,
        resonance_atol=resonance_atol,
        resonance_rtol=resonance_rtol,
        geometry_label="for isolated H",
    )
    polarizability.setflags(write=False)
    return polarizability


def _finite_field_static_polarizability(
    backend,
    internuclear_distance,
    *,
    field_strength=1e-3,
):
    """Return a four-point finite-field static polarizability tensor."""
    field_strength = _finite_positive_real(field_strength, "field_strength")
    integrals = backend.ao_integrals(internuclear_distance)

    def dipole(field):
        perturbed_hamiltonian = integrals.hamiltonian + np.einsum(
            "x,xij->ij", field, integrals.position
        )
        solution = _canonical_eigh(
            perturbed_hamiltonian,
            integrals.overlap,
            backend.linear_dependence_threshold,
        )
        ground_state = solution.coefficients[:, 0]
        return -np.einsum(
            "p,xpq,q->x",
            ground_state,
            integrals.position,
            ground_state,
            optimize=True,
        )

    tensor = np.empty((3, 3))
    for component in range(3):
        direction = np.zeros(3)
        direction[component] = field_strength
        positive = dipole(direction)
        negative = dipole(-direction)
        positive_two = dipole(2 * direction)
        negative_two = dipole(-2 * direction)
        tensor[:, component] = (
            8 * (positive - negative) - (positive_two - negative_two)
        ) / (12 * field_strength)
    return 0.5 * (tensor + tensor.T)


def _resolvent_polarizability(
    backend,
    internuclear_distance,
    frequencies,
):
    """Return dynamic polarizabilities from direct frequency-domain solves."""
    frequency_values = _frequency_array(frequencies)
    integrals = backend.ao_integrals(internuclear_distance)
    solution = _canonical_eigh(
        integrals.hamiltonian,
        integrals.overlap,
        backend.linear_dependence_threshold,
    )
    hamiltonian = solution.orthogonal_hamiltonian
    coefficients = solution.orthogonal_coefficients
    ground_state = coefficients[:, 0]
    ground_energy = solution.energies[0]
    dipole = -np.einsum(
        "pi,xpq,qj->xij",
        solution.orthogonalizer,
        integrals.position,
        solution.orthogonalizer,
        optimize=True,
    )
    projector = np.outer(ground_state, ground_state)
    complement = np.eye(len(ground_state)) - projector
    shifted = complement @ (
        hamiltonian - ground_energy * np.eye(len(ground_state))
    ) @ complement
    right_hand_sides = np.array(
        [complement @ dipole[component] @ ground_state for component in range(3)]
    )

    tensors = np.empty((len(frequency_values), 3, 3))
    for frequency_index, frequency in enumerate(frequency_values):
        minus = shifted - frequency * complement + projector
        plus = shifted + frequency * complement + projector
        response_minus = np.linalg.solve(minus, right_hand_sides.T).T
        response_plus = np.linalg.solve(plus, right_hand_sides.T).T
        tensor = np.empty((3, 3))
        for i in range(3):
            for j in range(3):
                tensor[i, j] = (
                    right_hand_sides[i] @ response_minus[j]
                    + right_hand_sides[j] @ response_plus[i]
                )
        tensors[frequency_index] = 0.5 * (tensor + tensor.T)
    return tensors
