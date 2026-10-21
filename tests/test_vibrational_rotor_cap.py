import numpy as np
import pytest

from born_oppenheimer_dynamics.vibrational_rotor import CubicCAP


def test_cubic_cap_has_requested_onset_and_strength():
    radial_grid = np.array([0.0, 3.0, 4.0, 5.0])

    potential = CubicCAP(alpha=0.6, eta=0.25)(radial_grid, r_max=5.0)

    np.testing.assert_array_equal(
        potential,
        -0.25j * np.array([0.0, 0.0, 1.0, 8.0]),
    )
    assert CubicCAP(alpha=0.6).eta == 1.0


@pytest.mark.parametrize("alpha", [-0.1, 1.0, np.inf])
def test_cubic_cap_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        CubicCAP(alpha=alpha)


@pytest.mark.parametrize("eta", [0.0, -1.0, np.inf])
def test_cubic_cap_rejects_invalid_eta(eta):
    with pytest.raises(ValueError, match="eta"):
        CubicCAP(alpha=0.8, eta=eta)
