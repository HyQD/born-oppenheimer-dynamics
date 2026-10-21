"""Field-free-preconditioned Crank-Nicolson propagation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, bicgstab


@dataclass(frozen=True)
class StepDiagnostics:
    """Diagnostics for one accepted Crank-Nicolson step."""

    method: str
    iterations: int
    residual_norm: float


class SplitCrankNicolson:
    r"""Crank-Nicolson solver organized around a fixed field-free Hamiltonian.

    For ``H(t) = H0 + V(t)`` and ``c = dt/2``, this class caches LU factors
    for ``I + i*c*H0`` in every angular channel. No dense inverse is formed.
    """

    def __init__(
        self,
        field_free_blocks,
        *,
        time_step,
        field,
        interaction,
        interaction_bound,
        atol=1e-12,
        rtol=0.0,
        maxiter=None,
        contraction_limit=0.8,
        fixed_point_maxiter=50,
    ):
        blocks = np.asarray(field_free_blocks)
        if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
            raise ValueError("field_free_blocks must have shape (n_l, n_r, n_r).")
        if not np.all(np.isfinite(blocks)):
            raise ValueError("field_free_blocks must contain finite values.")
        dtype = np.complex128 if np.iscomplexobj(blocks) else float
        self.field_free_blocks = np.asarray(blocks, dtype=dtype)
        self.n_l, self.n_r, _ = self.field_free_blocks.shape
        self.time_step = float(time_step)
        self.c = self.time_step / 2
        self.field = field
        self.interaction = interaction
        self.interaction_bound = interaction_bound
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.maxiter = maxiter
        self.contraction_limit = float(contraction_limit)
        self.fixed_point_maxiter = int(fixed_point_maxiter)
        if self.atol < 0 or self.rtol < 0 or self.atol == self.rtol == 0:
            raise ValueError("At least one of atol and rtol must be positive.")
        if not 0 < self.contraction_limit < 1:
            raise ValueError(
                "contraction_limit must lie strictly between zero and one."
            )
        if self.fixed_point_maxiter < 1:
            raise ValueError("fixed_point_maxiter must be positive.")

        identity = np.eye(self.n_r)
        self._factors = tuple(
            lu_factor(identity + 1j * self.c * block)
            for block in self.field_free_blocks
        )

    def _validate_state(self, state):
        values = np.asarray(state)
        if values.shape != (self.n_l, self.n_r):
            raise ValueError(
                f"state must have shape {(self.n_l, self.n_r)}, not {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("state must contain finite values.")
        return np.asarray(values, dtype=np.complex128)

    def apply_field_free(self, state):
        return np.einsum("lab,lb->la", self.field_free_blocks, state)

    def apply_preconditioner(self, state):
        return np.array(
            [lu_solve(factor, channel) for factor, channel in zip(self._factors, state)]
        )

    def _tolerance(self, old_state, new_state):
        scale = max(1.0, np.linalg.norm(old_state), np.linalg.norm(new_state))
        return self.atol + self.rtol * scale

    def _true_residual(self, old_state, new_state, field_value):
        state_sum = new_state + old_state
        hamiltonian_sum = self.apply_field_free(state_sum) + self.interaction(
            state_sum, field_value
        )
        return new_state - old_state + 1j * self.c * hamiltonian_sum

    def step(self, state, time) -> tuple[np.ndarray, StepDiagnostics]:
        """Advance one step and return the state and solver diagnostics."""
        old_state = self._validate_state(state)
        midpoint = float(time) + self.c
        field_value = float(np.asarray(self.field(midpoint)))
        if not np.isfinite(field_value):
            raise ValueError("The electric field must return a finite scalar.")

        if field_value == 0.0:
            new_state = 2 * self.apply_preconditioner(old_state) - old_state
            method = "field-free"
            iterations = 0
        else:
            interaction_old = self.interaction(old_state, field_value)
            right_hand_side = (
                self.apply_preconditioner(
                    2 * old_state - 1j * self.c * interaction_old
                )
                - old_state
            )
            contraction = self.c * float(self.interaction_bound(field_value))
            new_state = old_state
            method = "bicgstab"
            iterations = 0

            if contraction < self.contraction_limit:
                method = "fixed-point"
                for iteration in range(1, self.fixed_point_maxiter + 1):
                    candidate = (
                        right_hand_side
                        - 1j
                        * self.c
                        * self.apply_preconditioner(
                            self.interaction(new_state, field_value)
                        )
                    )
                    iterations = iteration
                    if np.linalg.norm(candidate - new_state) <= self._tolerance(
                        new_state, candidate
                    ):
                        new_state = candidate
                        break
                    new_state = candidate
                else:
                    method = "bicgstab"

            if method == "bicgstab":
                shape = old_state.shape
                size = old_state.size

                def matvec(vector):
                    values = vector.reshape(shape)
                    result = values + 1j * self.c * self.apply_preconditioner(
                        self.interaction(values, field_value)
                    )
                    return result.ravel()

                matrix = LinearOperator(
                    (size, size), matvec=matvec, dtype=np.complex128
                )
                iterations = 0

                def callback(_):
                    nonlocal iterations
                    iterations += 1

                solution, info = bicgstab(
                    matrix,
                    right_hand_side.ravel(),
                    x0=new_state.ravel(),
                    atol=self.atol,
                    rtol=self.rtol,
                    maxiter=self.maxiter,
                    callback=callback,
                )
                if info != 0:
                    raise RuntimeError(
                        f"Crank-Nicolson BiCGSTAB failed with info={info}."
                    )
                new_state = solution.reshape(shape)

        residual = self._true_residual(old_state, new_state, field_value)
        residual_norm = float(np.linalg.norm(residual))
        tolerance = self._tolerance(old_state, new_state)
        if residual_norm > 20 * tolerance:
            raise RuntimeError(
                "Crank-Nicolson step failed the true-residual check: "
                f"{residual_norm:.3e} > {20 * tolerance:.3e}."
            )
        return new_state, StepDiagnostics(method, iterations, residual_norm)
