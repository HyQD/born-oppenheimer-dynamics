import numpy as np
import pytest

from born_oppenheimer_dynamics.vibrational_rotor.propagator import (
    SplitCrankNicolson,
)


@pytest.mark.parametrize(
    ("contraction_limit", "expected_method"),
    [(0.8, "fixed-point"), (1e-6, "bicgstab")],
)
def test_split_crank_nicolson_matches_dense_direct_solve(
    contraction_limit, expected_method
):
    h0 = np.array(
        [
            [[0.4, -0.1], [-0.1, 0.7]],
            [[0.8, -0.2], [-0.2, 1.1]],
        ]
    )
    coupling = np.array(
        [
            [0.03, 0.01, 0.08, 0.00],
            [0.01, -0.02, 0.00, 0.04],
            [0.08, 0.00, 0.01, -0.01],
            [0.00, 0.04, -0.01, -0.03],
        ]
    )
    field_value = 0.6

    def interaction(state, field):
        return (field * coupling @ state.ravel()).reshape(state.shape)

    propagator = SplitCrankNicolson(
        h0,
        time_step=0.2,
        field=lambda time: field_value,
        interaction=interaction,
        interaction_bound=lambda field: abs(field) * np.linalg.norm(coupling, 2),
        atol=1e-13,
        contraction_limit=contraction_limit,
    )
    state = np.array([[0.7 + 0.1j, -0.2], [0.1j, 0.5]])
    propagated, diagnostics = propagator.step(state, 0.0)

    field_free = np.zeros((4, 4))
    field_free[:2, :2] = h0[0]
    field_free[2:, 2:] = h0[1]
    hamiltonian = field_free + field_value * coupling
    identity = np.eye(4)
    c = propagator.time_step / 2
    expected = np.linalg.solve(
        identity + 1j * c * hamiltonian,
        (identity - 1j * c * hamiltonian) @ state.ravel(),
    ).reshape(state.shape)

    np.testing.assert_allclose(propagated, expected, atol=2e-13)
    assert diagnostics.method == expected_method
    assert diagnostics.residual_norm < 1e-12


def test_exact_field_free_path():
    h0 = np.array([[[0.2, -0.1], [-0.1, 0.5]]])
    propagator = SplitCrankNicolson(
        h0,
        time_step=0.1,
        field=lambda time: 0.0,
        interaction=lambda state, field: np.zeros_like(state),
        interaction_bound=lambda field: 0.0,
    )
    state = np.array([[1.0, 0.2j]])

    propagated, diagnostics = propagator.step(state, 0.0)

    c = propagator.time_step / 2
    expected = np.linalg.solve(
        np.eye(2) + 1j * c * h0[0],
        (np.eye(2) - 1j * c * h0[0]) @ state[0],
    )
    np.testing.assert_allclose(propagated[0], expected, atol=2e-14)
    assert diagnostics.method == "field-free"
    assert diagnostics.iterations == 0
