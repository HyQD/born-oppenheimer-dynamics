# Crank–Nicolson time propagation

Both rotor models use the internal `SplitCrankNicolson` propagator. The name
refers to its field-free block preconditioner; the accepted step solves the full
Crank–Nicolson equation and is not an operator-splitting approximation.

## Midpoint equation

Write the Hamiltonian as

$$H(t)=H_0+V(t),$$

where $H_0$ is time independent and block diagonal in the retained angular
channels. The interaction is evaluated at the midpoint of each step,
$H_{n+1/2}=H(t_n+\Delta t/2)$. The update is

$$\left(I+\frac{i\Delta t}{2}H_{n+1/2}\right)\Psi_{n+1}=\left(I-\frac{i\Delta t}{2}H_{n+1/2}\right)\Psi_n.$$

The midpoint Crank–Nicolson method is second-order accurate in time. For a
Hermitian Hamiltonian the update is norm preserving up to the linear-solver
tolerance. A complex absorbing potential makes $H_0$ non-Hermitian and causes
the intended norm loss; the state is never renormalized.

## Field-free preconditioner

Let $c=\Delta t/2$ and $A_0=I+icH_0$. Each angular block of $A_0$ is LU
factorized once and reused. The blocks are one-dimensional for the rigid rotor
and dense FEM-DVR radial matrices for the vibrational rotor.

At a midpoint where the electric field is exactly zero, the update reduces to

$$\Psi_{n+1}=2A_0^{-1}\Psi_n-\Psi_n.$$

No iteration is needed. With a nonzero field, the preconditioned implicit
equation is

$$\Psi_{n+1}=b-icA_0^{-1}V(t_{n+1/2})\Psi_{n+1},$$

where $b$ contains the old-state terms. The solver first estimates the
contraction factor $c\lVert V\rVert$. If it is below `contraction_limit`, a
fixed-point iteration is attempted. Otherwise, or if fixed-point iteration does
not converge within `fixed_point_maxiter`, the preconditioned equation is solved
with BiCGSTAB.

After either route, the implementation evaluates the residual of the original,
unpreconditioned Crank–Nicolson equation. A step is rejected with an exception if
that residual is inconsistent with the requested tolerance.

## Configuration and diagnostics

Both high-level interfaces pass `integrator_options` to the propagator. The
available options are:

| Option | Default | Meaning |
| --- | ---: | --- |
| `atol` | `1e-12` | Absolute iterative-solver tolerance |
| `rtol` | `0.0` | Relative iterative-solver tolerance |
| `maxiter` | `None` | Maximum BiCGSTAB iterations |
| `contraction_limit` | `0.8` | Largest estimated contraction for trying fixed point |
| `fixed_point_maxiter` | `50` | Maximum fixed-point iterations |

At least one of `atol` and `rtol` must be positive. For example:

```python
calculation = RigidRotorDiatomic(
    ...,
    time_step=10.0,
    integrator_options={
        "atol": 1e-12,
        "rtol": 0.0,
        "contraction_limit": 0.8,
    },
)
```

Every result contains `solver_methods`, `solver_iterations`, and
`residual_norms` at the sampled times. Method labels are `"initial"`,
`"field-free"`, `"fixed-point"`, and `"bicgstab"`.

The requested time step is treated as an upper bound. For duration $T$, the
interfaces select

$$N=\left\lceil\frac{T}{\Delta t_\mathrm{requested}}\right\rceil,\qquad \Delta t=\frac{T}{N},$$

so that the final time is reached exactly. The actual value is available as
`calculation.time_step` and `result.time_step`.

