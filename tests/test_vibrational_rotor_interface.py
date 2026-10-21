import numpy as np
import pytest
from electromagnetic_fields.dipole_fields import SineSquareLaser

from born_oppenheimer_dynamics.vibrational_rotor import (
    CubicCAP,
    DiatomicMolecule,
    DiatomicSurfaces,
    VibrationalRotorDiatomic,
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
    tmp_path,
):
    calculation = VibrationalRotorDiatomic(
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
    assert result.radial_probability.shape == (
        len(result.times),
        len(result.radial_grid),
    )
    np.testing.assert_allclose(
        result.radial_probability @ result.radial_weights,
        result.norms,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        (result.radial_probability * result.radial_grid[None, :])
        @ result.radial_weights,
        result.expectation_r,
        atol=2e-14,
    )
    output_path = tmp_path / "result.npz"
    result.save(output_path)
    with np.load(output_path) as stored:
        np.testing.assert_array_equal(
            stored["radial_probability"], result.radial_probability
        )
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
    calculation = VibrationalRotorDiatomic(
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
        VibrationalRotorDiatomic(
            DiatomicMolecule(("Li-7", "H-1")),
            homonuclear_surfaces,
            nodes=[0.0, 3.0, 6.0],
            n_points=5,
            final_time=0.1,
            time_step=0.05,
        )


def test_radial_boundary_cannot_exceed_surface_data(homonuclear_surfaces):
    with pytest.raises(ValueError, match="exceeds"):
        VibrationalRotorDiatomic(
            DiatomicMolecule(("H-1", "H-1")),
            homonuclear_surfaces,
            r_max=6.1,
            n_elements=2,
            n_points=5,
            final_time=0.1,
            time_step=0.05,
        )


def test_element_size_constructs_uniform_nodes_ending_at_r_max(
    homonuclear_surfaces,
):
    calculation = VibrationalRotorDiatomic(
        DiatomicMolecule(("H-1", "H-1")),
        homonuclear_surfaces,
        r_max=6.0,
        element_size=0.2,
        n_points=3,
        final_time=0.1,
        time_step=0.05,
    )

    np.testing.assert_allclose(np.diff(calculation.grid.nodes), 0.2)
    assert calculation.grid.nodes[-1] == 6.0


def test_element_size_must_divide_r_max(homonuclear_surfaces):
    with pytest.raises(ValueError, match="integer multiple"):
        VibrationalRotorDiatomic(
            DiatomicMolecule(("H-1", "H-1")),
            homonuclear_surfaces,
            r_max=6.0,
            element_size=0.7,
            n_points=3,
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
    calculation = VibrationalRotorDiatomic(
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
        VibrationalRotorDiatomic(
            DiatomicMolecule(("H-1", "H-1")),
            homonuclear_surfaces,
            nodes=[0.0, 3.0, 6.0],
            n_points=5,
            fields=pulse,
            final_time=0.1,
            time_step=0.05,
        )


def test_cap_is_excluded_from_initial_state_hamiltonian(homonuclear_surfaces):
    common_options = dict(
        nodes=[0.0, 3.0, 6.0],
        n_points=5,
        l_max=0,
        final_time=0.1,
        time_step=0.05,
    )
    molecule = DiatomicMolecule(("H-1", "H-1"))
    without_cap = VibrationalRotorDiatomic(
        molecule,
        homonuclear_surfaces,
        **common_options,
    )
    with_cap = VibrationalRotorDiatomic(
        molecule,
        homonuclear_surfaces,
        cap=CubicCAP(alpha=0.5, eta=0.2),
        **common_options,
    )

    np.testing.assert_array_equal(
        with_cap.field_free_blocks, without_cap.field_free_blocks
    )
    np.testing.assert_array_equal(with_cap.initial_state, without_cap.initial_state)
    assert np.iscomplexobj(with_cap.propagation_blocks)
    diagonal_difference = np.diagonal(
        with_cap.propagation_blocks - with_cap.field_free_blocks,
        axis1=1,
        axis2=2,
    )
    np.testing.assert_allclose(diagonal_difference, with_cap.cap_potential[None, :])


def test_cap_absorbs_without_normalizing_observables(homonuclear_surfaces):
    calculation = VibrationalRotorDiatomic(
        DiatomicMolecule(("H-1", "H-1")),
        homonuclear_surfaces,
        nodes=[0.0, 3.0, 6.0],
        n_points=3,
        l_max=0,
        initial_state=np.array([[0.0, 0.0, 1.0]]),
        cap=CubicCAP(alpha=0.5, eta=0.2),
        final_time=0.2,
        time_step=0.02,
    )

    result = calculation.run()

    assert result.norms[-1] < result.norms[0]
    assert np.all(np.diff(result.norms) <= 2e-14)
    np.testing.assert_allclose(
        (result.radial_probability * result.radial_grid[None, :])
        @ result.radial_weights,
        result.expectation_r,
    )
    assert result.expectation_r[-1] != pytest.approx(
        result.expectation_r[-1] / result.norms[-1]
    )


@pytest.mark.parametrize(
    ("cap", "exception", "message"),
    [
        (1.0, TypeError, "callable or None"),
        (lambda r, r_max: np.zeros(1), ValueError, "shape"),
        (lambda r, r_max: 1j * np.ones_like(r), ValueError, "non-positive"),
    ],
)
def test_invalid_cap_is_rejected(homonuclear_surfaces, cap, exception, message):
    with pytest.raises(exception, match=message):
        VibrationalRotorDiatomic(
            DiatomicMolecule(("H-1", "H-1")),
            homonuclear_surfaces,
            nodes=[0.0, 3.0, 6.0],
            n_points=3,
            cap=cap,
            final_time=0.1,
            time_step=0.05,
        )
