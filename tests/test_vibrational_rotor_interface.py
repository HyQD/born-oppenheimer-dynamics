import numpy as np
import pytest
from electromagnetic_fields.dipole_fields import SineSquareLaser

from born_oppenheimer_dynamics.vibrational_rotor import (
    DiatomicMolecule,
    DiatomicSurfaces,
    VibrationalRotor,
)


@pytest.fixture
def distance():
    return np.linspace(0.1, 6.0, 60)


@pytest.fixture
def homonuclear_surfaces(distance):
    return DiatomicSurfaces(
        distance,
        0.1 * (distance - 2.0) ** 2,
        polarizability_parallel=np.full_like(distance, 2.0),
        polarizability_perpendicular=np.ones_like(distance),
    )


def test_field_free_ground_state_uses_even_sector_and_stays_stationary(
    homonuclear_surfaces,
):
    calculation = VibrationalRotor(
        DiatomicMolecule(("H-1", "H-1")),
        homonuclear_surfaces,
        nodes=[0.0, 3.0, 6.0],
        n_points=7,
        l_max=4,
        initial_state="groundstate",
        final_time=0.2,
        time_step=0.05,
    )

    result = calculation.run()

    np.testing.assert_array_equal(calculation.ell, [0, 2, 4])
    assert calculation.angular_sector == "even"
    np.testing.assert_allclose(result.norms, 1.0, atol=2e-14)
    np.testing.assert_allclose(result.expectation_r, result.expectation_r[0])
    np.testing.assert_allclose(result.expectation_cos_theta, 0.0, atol=1e-15)
    np.testing.assert_allclose(result.expectation_cos_theta_squared, 1 / 3)
    assert set(result.solver_methods[1:]) == {"field-free"}


def test_heteronuclear_dipole_uses_full_sector(distance):
    surfaces = DiatomicSurfaces(
        distance,
        0.1 * (distance - 2.0) ** 2,
        dipole=np.full_like(distance, 0.2),
        polarizability_parallel=np.full_like(distance, 2.0),
        polarizability_perpendicular=np.ones_like(distance),
        beta_parallel=np.full_like(distance, 0.1),
        beta_perpendicular=np.full_like(distance, 0.05),
    )
    calculation = VibrationalRotor(
        DiatomicMolecule(("Li-7", "H-1")),
        surfaces,
        nodes=[0.0, 3.0, 6.0],
        n_points=7,
        l_max=3,
        fields=lambda time: 0.01,
        final_time=0.1,
        time_step=0.05,
    )

    result = calculation.run()

    assert calculation.angular_sector == "full"
    np.testing.assert_array_equal(calculation.ell, [0, 1, 2, 3])
    assert np.any(result.expectation_cos_theta[1:] != 0)
    np.testing.assert_allclose(result.norms, 1.0, atol=2e-12)


def test_heteronuclear_surface_requires_permanent_dipole(homonuclear_surfaces):
    with pytest.raises(ValueError, match="permanent-dipole"):
        VibrationalRotor(
            DiatomicMolecule(("Li-7", "H-1")),
            homonuclear_surfaces,
            nodes=[0.0, 3.0, 6.0],
            n_points=5,
            final_time=0.1,
            time_step=0.05,
        )

def test_radial_boundary_cannot_exceed_surface_data(homonuclear_surfaces):
    with pytest.raises(ValueError, match="exceeds"):
        VibrationalRotor(
            DiatomicMolecule(("H-1", "H-1")),
            homonuclear_surfaces,
            r_max=6.1,
            n_elements=2,
            n_points=5,
            final_time=0.1,
            time_step=0.05,
        )


def test_finite_pulse_support_can_define_final_time(homonuclear_surfaces):
    pulse = SineSquareLaser(
        E0=1e-3,
        omega=1.0,
        ncycles=0.1,
        gauge="length",
    )
    calculation = VibrationalRotor(
        DiatomicMolecule(("H-1", "H-1")),
        homonuclear_surfaces,
        nodes=[0.0, 3.0, 6.0],
        n_points=5,
        l_max=2,
        fields=[pulse],
        time_step=0.1,
    )

    assert calculation.final_time == pytest.approx(pulse.time_support[1])
    assert calculation.time_step <= 0.1


def test_velocity_gauge_field_is_rejected(homonuclear_surfaces):
    pulse = SineSquareLaser(
        E0=1e-3,
        omega=1.0,
        ncycles=1,
        gauge="velocity",
    )
    with pytest.raises(ValueError, match="length-gauge"):
        VibrationalRotor(
            DiatomicMolecule(("H-1", "H-1")),
            homonuclear_surfaces,
            nodes=[0.0, 3.0, 6.0],
            n_points=5,
            fields=pulse,
            final_time=0.1,
            time_step=0.05,
        )
