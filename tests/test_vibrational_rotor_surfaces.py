import numpy as np
import pytest
from matter_field_params import reduced_mass

from born_oppenheimer_dynamics.vibrational_rotor import (
    DiatomicMolecule,
    DiatomicSurfaces,
    h2plus_aug_cc_pv5z_surfaces,
)


def test_diatomic_molecule_uses_isotope_reduced_mass():
    molecule = DiatomicMolecule(("Li-7", "H-1"))

    assert molecule.element_symbols == ("Li", "H")
    assert not molecule.homonuclear
    assert molecule.reduced_mass == pytest.approx(
        reduced_mass(("Li-7", "H-1"))
    )

def test_hydrogen_isotopologues_share_the_electronic_element():
    assert DiatomicMolecule(("H-1", "D")).homonuclear


def test_surface_interpolation_and_upper_bound():
    distance = np.linspace(0.1, 4.0, 40)
    surfaces = DiatomicSurfaces(
        distance,
        distance**2,
        polarizability_parallel=2 * distance,
        polarizability_perpendicular=distance,
    )

    interpolated = surfaces.interpolate(np.array([0.05, 1.0, 3.5]))
    np.testing.assert_allclose(interpolated.electronic_energy[1:], [1.0, 12.25])
    assert interpolated.dipole is None
    with pytest.raises(ValueError, match="beyond"):
        surfaces.interpolate(np.array([4.1]))


def test_hyperpolarizability_components_are_a_pair():
    distance = np.linspace(0.1, 2.0, 20)
    with pytest.raises(ValueError, match="both"):
        DiatomicSurfaces(
            distance,
            distance,
            polarizability_parallel=distance,
            polarizability_perpendicular=distance,
            beta_parallel=distance,
        )


def test_packaged_h2plus_reference_data():
    surfaces = h2plus_aug_cc_pv5z_surfaces()

    assert surfaces.label == "H2+ aug-cc-pV5Z"
    assert surfaces.r_min == pytest.approx(0.1)
    assert surfaces.r_max == pytest.approx(12.0)
    assert len(surfaces.internuclear_distance) == 120
    np.testing.assert_allclose(np.diff(surfaces.internuclear_distance), 0.1)
    assert surfaces.dipole is None
    assert np.all(surfaces.polarizability_perpendicular > 0)
    assert surfaces.polarizability_perpendicular[0] == pytest.approx(
        0.2914818278236127
    )
