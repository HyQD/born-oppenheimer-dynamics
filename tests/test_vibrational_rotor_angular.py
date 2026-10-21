import numpy as np

from born_oppenheimer_dynamics.vibrational_rotor import rotor_angular_operators


def test_axisymmetric_angular_matrix_elements():
    operators = rotor_angular_operators(4)

    np.testing.assert_allclose(operators.cos_theta[0, 1], 1 / np.sqrt(3))
    np.testing.assert_allclose(operators.cos_theta_squared[0, 0], 1 / 3)
    np.testing.assert_allclose(
        operators.cos_two_theta,
        2 * operators.cos_theta_squared - operators.identity,
    )
    np.testing.assert_allclose(
        operators.sin_theta_squared_cos_theta,
        operators.cos_theta - operators.cos_theta_cubed,
    )


def test_even_sector_removes_odd_angular_operators():
    operators = rotor_angular_operators(6, np.arange(0, 7, 2))

    np.testing.assert_array_equal(operators.ell, [0, 2, 4, 6])
    np.testing.assert_allclose(operators.cos_theta, 0.0, atol=1e-15)
    np.testing.assert_allclose(operators.cos_theta_cubed, 0.0, atol=1e-15)
    assert np.any(np.abs(operators.cos_theta_squared) > 0)
