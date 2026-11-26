import builtins

import numpy as np
import pytest
from scipy.linalg import eigh

from born_oppenheimer_dynamics.electronic_structure.pyscf_gto import (
    _H2PlusGTOBackend,
    _canonical_eigh,
    _load_optional_dependencies,
)


def test_canonical_eigh_matches_generalized_solver():
    hamiltonian = np.array([[1.0, 0.2], [0.2, 2.0]])
    overlap = np.array([[1.0, 0.1], [0.1, 1.5]])

    expected_energies, _ = eigh(hamiltonian, overlap)
    result = _canonical_eigh(hamiltonian, overlap, 1e-12)

    np.testing.assert_allclose(result.energies, expected_energies)
    np.testing.assert_allclose(
        result.coefficients.T @ overlap @ result.coefficients,
        np.eye(2),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        hamiltonian @ result.coefficients,
        (overlap @ result.coefficients) * result.energies[None, :],
        atol=1e-14,
    )


def test_canonical_eigh_removes_linearly_dependent_overlap_modes():
    hamiltonian = np.diag([1.0, 2.0])
    overlap = np.diag([1.0, 1e-12])

    result = _canonical_eigh(hamiltonian, overlap, 1e-10)

    np.testing.assert_allclose(result.energies, [1.0])
    np.testing.assert_allclose(result.overlap_eigenvalues, [1e-12, 1.0])
    np.testing.assert_allclose(result.retained_overlap_eigenvalues, [1.0])
    assert result.coefficients.shape == (2, 1)

    with pytest.raises(ValueError, match="All AO combinations"):
        _canonical_eigh(hamiltonian, overlap, 2.0)


@pytest.mark.parametrize(
    ("hamiltonian", "overlap", "message"),
    [
        (np.ones((2, 3)), np.eye(2), "square"),
        (np.eye(2), np.ones((2, 3)), "square"),
        (np.array([[1.0, 1.0], [0.0, 1.0]]), np.eye(2), "symmetric"),
        (np.eye(2), np.array([[1.0, 1.0], [0.0, 1.0]]), "symmetric"),
    ],
)
def test_canonical_eigh_validates_matrices(hamiltonian, overlap, message):
    with pytest.raises(ValueError, match=message):
        _canonical_eigh(hamiltonian, overlap, 1e-10)


def test_h2plus_backend_builds_centered_one_electron_molecule():
    backend = _H2PlusGTOBackend("sto-3g")

    molecule = backend.build_molecule(2.0)

    assert molecule.nelectron == 1
    assert molecule.spin == 1
    assert molecule.charge == 1
    np.testing.assert_allclose(
        molecule.atom_coords(),
        [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
    )
    assert molecule.energy_nuc() == pytest.approx(0.5)
    assert backend.basis_metadata["name"] == "STO-3G"
    assert backend.basis_metadata["requested_name"] == "sto-3g"
    assert backend.basis_metadata["pyscf_version"]
    assert backend.basis_metadata["basis_set_exchange_version"]


def test_h2plus_backend_solves_complete_sto3g_spectrum():
    backend = _H2PlusGTOBackend("sto-3g")

    point = backend.solve(2.0)

    assert point.internuclear_distance == 2.0
    assert point.electronic_energy == pytest.approx(-0.5826953691131758)
    assert point.electronic_energy == pytest.approx(
        point.orbital_energies[0] + 0.5
    )
    assert point.orbital_energies.shape == (2,)
    assert point.dipole_matrix.shape == (3, 2, 2)
    np.testing.assert_allclose(
        point.dipole_matrix,
        point.dipole_matrix.transpose(0, 2, 1),
        atol=1e-15,
    )
    np.testing.assert_allclose(point.dipole_matrix[:, 0, 0], 0.0, atol=1e-14)
    assert abs(point.dipole_matrix[2, 0, 1]) > 1.0
    assert point.discarded_overlap_modes == 0
    assert point.retained_electronic_dimension == 2
    assert point.orthonormality_error < 1e-14
    assert point.projected_eigenproblem_residual < 1e-14
    assert not point.orbital_energies.flags.writeable
    assert not point.dipole_matrix.flags.writeable


def test_aug_cc_pv5z_short_range_energy_and_linear_dependence():
    backend = _H2PlusGTOBackend("aug-cc-pv5z")

    point = backend.solve(0.1)

    assert point.electronic_energy == pytest.approx(8.02189639805504)
    assert point.minimum_overlap_eigenvalue < 1e-8
    assert point.minimum_retained_overlap_eigenvalue > 1e-8
    assert point.discarded_overlap_modes == 1
    assert point.retained_electronic_dimension == 159
    assert point.orthonormality_error < 1e-8
    assert point.projected_eigenproblem_residual < 1e-7


@pytest.mark.parametrize("distance", [0.0, -1.0, np.inf, True])
def test_h2plus_backend_rejects_invalid_distance(distance):
    backend = _H2PlusGTOBackend("sto-3g")
    with pytest.raises((TypeError, ValueError), match="internuclear_distance"):
        backend.solve(distance)


def test_h2plus_backend_reports_unknown_basis():
    with pytest.raises(ValueError, match="Basis Set Exchange"):
        _H2PlusGTOBackend("definitely-not-a-basis")


def test_optional_dependency_error_is_actionable(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.split(".", 1)[0] == "basis_set_exchange":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=r"\.\[pyscf\]"):
        _load_optional_dependencies()
