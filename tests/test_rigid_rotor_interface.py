import numpy as np
import pytest
from electromagnetic_fields.dipole_fields import SineSquareLaser

from born_oppenheimer_dynamics import (
    DiatomicMolecule,
    DiatomicSurfaces,
    RigidRotorDiatomic,
    RotationalState,
)


@pytest.fixture
def homonuclear_molecule():
    return DiatomicMolecule(
        ("H-1", "H-1"),
        reduced_mass_override=900.0,
    )


@pytest.fixture
def surfaces():
    distance = np.linspace(1.0, 4.0, 7)
    return DiatomicSurfaces(
        distance,
        0.1 * (distance - 2.0) ** 2,
        polarizability_parallel=2.0 + distance,
        polarizability_perpendicular=1.0 + 0.5 * distance,
    )


def test_field_free_ground_state_uses_even_sector_and_stays_stationary(
    homonuclear_molecule,
    tmp_path,
):
    calculation = RigidRotorDiatomic(
        homonuclear_molecule,
        bond_length=2.0,
        polarizability_parallel=4.0,
        polarizability_perpendicular=2.0,
        l_max=4,
        final_time=0.2,
        time_step=0.05,
    )

    result = calculation.run()

    assert calculation.rotational_constant == pytest.approx(1 / (2 * 900 * 4))
    assert calculation.angular_sector == "even"
    np.testing.assert_array_equal(result.ell, [0, 2, 4])
    np.testing.assert_allclose(result.norms, 1.0, atol=1e-15)
    np.testing.assert_allclose(result.expectation_cos_theta, 0.0, atol=1e-15)
    np.testing.assert_allclose(result.expectation_cos_theta_squared, 1 / 3)
    np.testing.assert_allclose(result.angular_momentum_populations[:, 0], 1.0)
    np.testing.assert_allclose(result.angular_momentum_populations[:, 1:], 0.0)
    assert set(result.solver_methods[1:]) == {"field-free"}

    output_path = tmp_path / "rigid_result.npz"
    result.save(output_path)
    with np.load(output_path) as stored:
        np.testing.assert_array_equal(
            stored["angular_momentum_populations"],
            result.angular_momentum_populations,
        )
        assert stored["bond_length"] == pytest.approx(2.0)
        assert stored["rotational_constant"] == pytest.approx(
            calculation.rotational_constant
        )


def test_rotational_constant_can_be_supplied_directly(homonuclear_molecule):
    calculation = RigidRotorDiatomic(
        homonuclear_molecule,
        rotational_constant=3.4e-5,
        polarizability_parallel=4.0,
        polarizability_perpendicular=2.0,
        final_time=0.1,
        time_step=0.05,
    )

    assert calculation.bond_length is None
    assert calculation.rotational_constant == pytest.approx(3.4e-5)


@pytest.mark.parametrize(
    "rotor_parameters",
    [{}, {"bond_length": 2.0, "rotational_constant": 3.4e-5}],
)
def test_exactly_one_rotational_parameter_is_required(
    homonuclear_molecule,
    rotor_parameters,
):
    with pytest.raises(ValueError, match="exactly one"):
        RigidRotorDiatomic(
            homonuclear_molecule,
            polarizability_parallel=4.0,
            polarizability_perpendicular=2.0,
            final_time=0.1,
            time_step=0.05,
            **rotor_parameters,
        )


def test_odd_initial_state_selects_odd_sector(homonuclear_molecule):
    calculation = RigidRotorDiatomic(
        homonuclear_molecule,
        bond_length=2.0,
        polarizability_parallel=4.0,
        polarizability_perpendicular=2.0,
        l_max=5,
        initial_state=RotationalState(1),
        final_time=0.1,
        time_step=0.05,
    )

    assert calculation.angular_sector == "odd"
    np.testing.assert_array_equal(calculation.ell, [1, 3, 5])
    np.testing.assert_array_equal(calculation.initial_state[:, 0], [1.0, 0.0, 0.0])


def test_from_surfaces_interpolates_response_at_fixed_bond_length(
    homonuclear_molecule,
    surfaces,
):
    calculation = RigidRotorDiatomic.from_surfaces(
        homonuclear_molecule,
        surfaces,
        bond_length=2.5,
        l_max=2,
        final_time=0.1,
        time_step=0.05,
    )

    assert calculation.dipole == 0.0
    assert calculation.alpha_parallel == pytest.approx(4.5)
    assert calculation.alpha_perpendicular == pytest.approx(2.25)


@pytest.mark.parametrize("bond_length", [0.9, 4.1])
def test_from_surfaces_rejects_bond_length_outside_data(
    homonuclear_molecule,
    surfaces,
    bond_length,
):
    with pytest.raises(ValueError, match="within the supplied electronic data"):
        RigidRotorDiatomic.from_surfaces(
            homonuclear_molecule,
            surfaces,
            bond_length=bond_length,
            final_time=0.1,
            time_step=0.05,
        )


def test_heteronuclear_dipole_activates_full_sector_and_orientation():
    calculation = RigidRotorDiatomic(
        DiatomicMolecule(
            ("Li-7", "H-1"),
            reduced_mass_override=1000.0,
        ),
        bond_length=3.0,
        dipole=0.5,
        polarizability_parallel=5.0,
        polarizability_perpendicular=3.0,
        l_max=3,
        fields=lambda time: 0.01,
        final_time=0.2,
        time_step=0.05,
    )

    result = calculation.run()

    assert calculation.angular_sector == "full"
    np.testing.assert_array_equal(result.ell, [0, 1, 2, 3])
    assert np.any(np.abs(result.expectation_cos_theta[1:]) > 0)
    np.testing.assert_allclose(result.norms, 1.0, atol=2e-12)
    np.testing.assert_allclose(
        result.angular_momentum_populations.sum(axis=1),
        result.norms,
        atol=1e-15,
    )


def test_heteronuclear_molecule_requires_permanent_dipole():
    with pytest.raises(ValueError, match="permanent dipole"):
        RigidRotorDiatomic(
            DiatomicMolecule(
                ("Li-7", "H-1"),
                reduced_mass_override=1000.0,
            ),
            bond_length=3.0,
            polarizability_parallel=5.0,
            polarizability_perpendicular=3.0,
            final_time=0.1,
            time_step=0.05,
        )


def test_custom_initial_state_is_normalized(homonuclear_molecule):
    calculation = RigidRotorDiatomic(
        homonuclear_molecule,
        bond_length=2.0,
        polarizability_parallel=4.0,
        polarizability_perpendicular=2.0,
        l_max=2,
        angular_sector="full",
        initial_state=np.array([1.0, 1.0j, 0.0]),
        final_time=0.1,
        time_step=0.05,
    )

    np.testing.assert_allclose(np.linalg.norm(calculation.initial_state), 1.0)


def test_velocity_gauge_field_is_rejected(homonuclear_molecule):
    pulse = SineSquareLaser(
        E0=1e-3,
        omega=1.0,
        ncycles=1,
        gauge="velocity",
    )
    with pytest.raises(ValueError, match="length-gauge"):
        RigidRotorDiatomic(
            homonuclear_molecule,
            bond_length=2.0,
            polarizability_parallel=4.0,
            polarizability_perpendicular=2.0,
            fields=pulse,
            final_time=0.1,
            time_step=0.05,
        )
