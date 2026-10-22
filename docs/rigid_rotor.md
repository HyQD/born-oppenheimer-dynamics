# Time-dependent rigid rotor

`RigidRotorDiatomic` describes a diatomic molecule at a fixed internuclear
distance. It is a dedicated model rather than a frozen-grid special case of the
vibrational rotor, so it carries no radial discretization or complex absorbing
potential.

## Hamiltonian and basis

For rotational constant $B$, the field-free Hamiltonian and its eigenvalues are

$$H_0=B\mathbf{J}^2,\qquad E_J=BJ(J+1).$$

The axisymmetric angular wavefunction is expanded as

$$\Psi(\Omega,t)=\sum_{\ell\in\mathcal L}c_\ell(t)Y_{\ell0}(\Omega).$$

The retained set $\mathcal L$ is controlled by `l_max` and `angular_sector`.
The allowed sectors are `"full"`, `"even"`, `"odd"`, and `"auto"`. The automatic
choice selects the parity of the initial $J$ if the interaction preserves
parity, and the full basis if a permanent dipole, a first hyperpolarizability,
or a custom initial state requires it. This is the efficient even-ℓ reduction
normally available for a homonuclear molecule initially in $J=0$; the code
does not impose nuclear-spin statistical weights.

Angular matrix elements are constructed from exact Gaunt coefficients provided
by `grid_lib.spherical_coordinates`. No angular quadrature files are required by
this package. The [electric-field interaction](electric_fields.md) is shared with
the vibrational rotor.

## Rotational constant

Exactly one of `rotational_constant` and `bond_length` must be supplied. A direct
rotational constant is given in atomic units. With a bond length, the interface
uses isotope masses from `matter_field_params` and computes

$$B=\frac{1}{2M_\mathrm{red}R_0^2}.$$

For example:

```python
from born_oppenheimer_dynamics import DiatomicMolecule, RigidRotorDiatomic

molecule = DiatomicMolecule(("Li-7", "H-1"))

calculation = RigidRotorDiatomic(
    molecule,
    bond_length=3.015,
    dipole=0.0,
    polarizability_parallel=30.1,
    polarizability_perpendicular=24.0,
    l_max=20,
    fields=[pulse],
    final_time=final_time,
    time_step=10.0,
)
```

A spectroscopic rotational constant reported as a wavenumber is converted with

$$B\,[\mathrm{a.u.}]=B\,[\mathrm{cm}^{-1}]\frac{100hc}{E_\mathrm{h}}.$$

### Sampling response curves

`RigidRotorDiatomic.from_surfaces` evaluates a `DiatomicSurfaces` object at a
specified bond length:

```python
calculation = RigidRotorDiatomic.from_surfaces(
    molecule,
    surfaces,
    bond_length=2.0,
    l_max=12,
    fields=[pulse],
    time_step=0.25,
)
```

The bond length must lie inside the tabulated surface interval. The electronic
energy at a fixed geometry is omitted because it contributes only a global
phase.

## Initial state

`initial_state` may be any of the following:

- `"groundstate"`, equivalent to $J=0$;
- an integer $J$;
- `RotationalState(rotational=J)`;
- a one-dimensional complex coefficient array over the retained angular basis.

A custom state is normalized automatically. Because its angular parity cannot
be inferred in advance, `angular_sector="auto"` selects the full basis for a
custom array.

## Results

`calculation.run()` returns a `RigidRotorResult` containing:

- `times` and `electric_field`;
- `norms`;
- `expectation_cos_theta` and `expectation_cos_theta_squared`;
- `angular_momentum_populations`, where each column follows `result.ell`;
- the final coefficients in `final_state`;
- time-step and iterative-solver diagnostics.

Sampling is controlled by `sample_interval`; the initial and final states are
always sampled. Save the complete result to a compressed NumPy archive with
`result.save("result.npz")`.

See [`rigid_rotor_h2plus.py`](../examples/rigid_rotor_h2plus.py),
[`rigid_rotor_LiH_single_color.py`](../examples/rigid_rotor_LiH_single_color.py),
and [`rigid_rotor_LiH_two_color.py`](../examples/rigid_rotor_LiH_two_color.py)
for complete calculations.

