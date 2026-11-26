import numpy as np
import pytest

from born_oppenheimer_dynamics import (
    compute_h2plus_gto,
    compute_hydrogen_gto_polarizability,
    h2plus_aug_cc_pv5z_surfaces,
)
from born_oppenheimer_dynamics.electronic_structure.pyscf_gto import (
    _H2PlusGTOBackend,
    _finite_field_static_polarizability,
    _frequency_array,
    _resolvent_polarizability,
)


def test_frequency_array_normalizes_scalar_and_sorts_sequence():
    np.testing.assert_array_equal(_frequency_array(0.057), [0.057])
    np.testing.assert_array_equal(
        _frequency_array([0.1, 0.0, 0.057]),
        [0.0, 0.057, 0.1],
    )


@pytest.mark.parametrize(
    ("frequencies", "message"),
    [
        ([], "non-empty"),
        ([-0.1], "non-negative"),
        ([0.1, 0.1], "duplicates"),
        ([0.0, np.inf], "finite"),
        ([0.0, 0.1j], "real"),
        (True, "real scalar"),
    ],
)
def test_frequency_array_rejects_invalid_input(frequencies, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _frequency_array(frequencies)


def test_compute_h2plus_gto_static_and_dynamic_sto3g():
    result = compute_h2plus_gto(
        basis="sto-3g",
        radial_range=(2.0, 2.0, 0.1),
        frequencies=[0.1, 0.0],
    )

    np.testing.assert_array_equal(result.internuclear_distance, [2.0])
    np.testing.assert_array_equal(result.frequencies, [0.0, 0.1])
    assert result.electronic_energy[0] == pytest.approx(-0.5826953691131758)
    assert result.polarizability.shape == (2, 1, 3, 3)
    assert result.polarizability_parallel[0, 0] == pytest.approx(
        5.327201565330635
    )
    assert result.polarizability_parallel[1, 0] == pytest.approx(
        5.571296796563015
    )
    np.testing.assert_allclose(result.polarizability_perpendicular, 0.0)
    np.testing.assert_allclose(
        result.diagnostics["ground_state_dipole"], 0.0, atol=1e-14
    )
    assert result.diagnostics["ionization_threshold"][0] == pytest.approx(
        1.0826953691131758
    )
    assert result.metadata["backend"] == "pyscf-gto-one-electron"
    assert result.metadata["basis"]["name"] == "STO-3G"


def test_static_sum_over_states_agrees_with_finite_field():
    backend = _H2PlusGTOBackend("sto-3g")
    result = compute_h2plus_gto(
        basis="sto-3g",
        radial_range=(2.0, 2.0, 0.1),
        frequencies=0.0,
    )

    errors = []
    for field_strength in (3e-3, 1e-3, 3e-4):
        finite_field = _finite_field_static_polarizability(
            backend,
            2.0,
            field_strength=field_strength,
        )
        errors.append(
            np.linalg.norm(
                result.polarizability[0, 0] - finite_field
            )
        )

    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 1e-9


def test_dynamic_sum_over_states_agrees_with_resolvent():
    frequencies = [0.0, 0.1, 0.2]
    backend = _H2PlusGTOBackend("sto-3g")
    result = compute_h2plus_gto(
        basis="sto-3g",
        radial_range=(2.0, 2.0, 0.1),
        frequencies=frequencies,
    )

    resolvent = _resolvent_polarizability(backend, 2.0, frequencies)

    np.testing.assert_allclose(
        result.polarizability[:, 0],
        resolvent,
        rtol=1e-13,
        atol=1e-13,
    )


def test_isolated_hydrogen_polarizability_is_isotropic():
    polarizability = compute_hydrogen_gto_polarizability(
        basis="aug-cc-pV5Z",
        frequencies=[0.114, 0.0, 0.057],
    )

    assert polarizability.shape == (3, 3, 3)
    diagonal = np.diagonal(polarizability, axis1=-2, axis2=-1)
    np.testing.assert_allclose(
        diagonal,
        np.repeat(diagonal[:, :1], 3, axis=1),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        polarizability - np.eye(3) * diagonal[:, :, None],
        0.0,
        atol=1e-12,
    )
    assert diagonal[0, 0] == pytest.approx(4.5, rel=2e-3)


def test_aug_cc_pv5z_static_response_matches_packaged_reference():
    result = compute_h2plus_gto(
        basis="aug-cc-pv5z",
        radial_range=(2.0, 2.0, 0.1),
        frequencies=0.0,
    )
    reference = h2plus_aug_cc_pv5z_surfaces()
    reference_index = np.flatnonzero(
        np.isclose(reference.internuclear_distance, 2.0)
    )[0]

    assert result.electronic_energy[0] == pytest.approx(
        reference.electronic_energy[reference_index],
        abs=1e-12,
    )
    assert result.polarizability_parallel[0, 0] == pytest.approx(
        reference.polarizability_parallel[reference_index],
        rel=1e-9,
    )
    assert result.polarizability_perpendicular[0, 0] == pytest.approx(
        reference.polarizability_perpendicular[reference_index],
        rel=1e-9,
    )
    assert result.diagnostics["cylindrical_symmetry_error"][0, 0] < 1e-12
    assert (
        result.diagnostics["maximum_off_diagonal_polarizability"][0, 0]
        < 1e-12
    )


def test_response_diagnostics_identify_closest_allowed_transition():
    result = compute_h2plus_gto(
        basis="sto-3g",
        radial_range=(2.0, 2.0, 0.1),
        frequencies=[0.0, 0.1],
    )
    diagnostics = result.diagnostics

    assert diagnostics["closest_transition_energy"].shape == (2, 1, 3)
    assert diagnostics["closest_transition_index"][0, 0, 2] == 1
    assert diagnostics["closest_transition_strength"][0, 0, 2] > 1.0
    assert diagnostics["has_dipole_allowed_transition"][0, 0, 2]
    assert not diagnostics["has_dipole_allowed_transition"][0, 0, 0]
    assert diagnostics["closest_transition_index"][0, 0, 0] == -1
    assert diagnostics["minimum_absolute_detuning"].shape == (2, 3)
    assert diagnostics["minimum_relative_detuning"].shape == (2, 3)


def test_compute_h2plus_gto_rejects_discrete_resonance():
    point = _H2PlusGTOBackend("sto-3g").solve(2.0)
    resonant_frequency = (
        point.orbital_energies[1] - point.orbital_energies[0]
    )

    with pytest.raises(ValueError, match=r"resonant.*R=2.*component=z"):
        compute_h2plus_gto(
            basis="sto-3g",
            radial_range=(2.0, 2.0, 0.1),
            frequencies=resonant_frequency,
        )


def test_compute_h2plus_gto_rejects_ionizing_frequency():
    with pytest.raises(ValueError, match=r"ionization continuum.*R=2"):
        compute_h2plus_gto(
            basis="sto-3g",
            radial_range=(2.0, 2.0, 0.1),
            frequencies=1.1,
        )


def test_transition_threshold_can_exclude_negligible_transitions():
    point = _H2PlusGTOBackend("sto-3g").solve(2.0)
    transition_frequency = (
        point.orbital_energies[1] - point.orbital_energies[0]
    )

    result = compute_h2plus_gto(
        basis="sto-3g",
        radial_range=(2.0, 2.0, 0.1),
        frequencies=transition_frequency,
        transition_strength_threshold=10.0,
    )

    np.testing.assert_allclose(result.polarizability, 0.0)
    assert not np.any(
        result.diagnostics["has_dipole_allowed_transition"]
    )


def test_compute_h2plus_gto_validates_response_options():
    with pytest.raises(ValueError, match="At least one"):
        compute_h2plus_gto(
            basis="sto-3g",
            radial_range=(2.0, 2.0, 0.1),
            resonance_atol=0.0,
            resonance_rtol=0.0,
        )
    with pytest.raises(TypeError, match="progress"):
        compute_h2plus_gto(
            basis="sto-3g",
            radial_range=(2.0, 2.0, 0.1),
            progress=1,
        )
