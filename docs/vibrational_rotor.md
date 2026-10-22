# Time-dependent vibrational rotor

`VibrationalRotorDiatomic` propagates the internuclear distance and molecular
orientation of a diatomic molecule on a single Born–Oppenheimer electronic
surface.

## Wavefunction and Hamiltonian

For an axisymmetric problem, the nuclear wavefunction is expanded in reduced
radial amplitudes:

$$\Psi(R,\Omega,t)=\frac{1}{R}\sum_{\ell\in\mathcal L}u_\ell(R,t)Y_{\ell0}(\Omega).$$

The field-free radial Hamiltonian in angular channel ℓ is

$$H_\ell=-\frac{1}{2M_\mathrm{red}}\frac{d^2}{dR^2}+\frac{\ell(\ell+1)}{2M_\mathrm{red}R^2}+V_\mathrm{BO}(R).$$

The isotope-dependent reduced mass is obtained from `DiatomicMolecule`. The
angular sector, exact angular operators, and
[electric-field interaction](electric_fields.md) follow the same conventions as
the [rigid rotor](rigid_rotor.md).

## Electronic curves

`DiatomicSurfaces` contains the clamped-nuclei quantities required by the
model:

- internuclear distance and electronic energy;
- parallel and perpendicular polarizabilities;
- an optional permanent dipole;
- optional parallel and perpendicular first hyperpolarizabilities.

All arrays use atomic units and must be sampled on one increasing internuclear
distance grid. The molecule is assumed to have been placed along the body-fixed
$z$ axis in the electronic-structure calculation. Curves are interpolated onto
the nuclear grid with cubic splines. Points above the largest tabulated distance
are rejected; points below the first tabulated distance are currently spline
extrapolated.

For homonuclear molecules, an omitted dipole curve is set to zero. A dipole curve
is required for heteronuclear molecules. Both hyperpolarizability curves must be
provided together or omitted together.

`DiatomicSurfaces.from_npz()` reads the established `R`, `E_R`, and `alpha`
archive layout, with optional `dipole`, `beta_parallel`, and `beta_perpendicular`
arrays. The package includes H2+ aug-cc-pV5Z reference curves from 0.1 to 12 bohr
at 0.1-bohr spacing:

```python
from born_oppenheimer_dynamics import h2plus_aug_cc_pv5z_surfaces

surfaces = h2plus_aug_cc_pv5z_surfaces()
```

## FEM-DVR radial grid

The radial coordinate uses the FEM-DVR implementation in `grid-lib`, with
Gauss–Legendre–Lobatto points in each finite element. The element boundaries can
be supplied in three ways:

- explicit `nodes` beginning at zero;
- `r_max` and `n_elements`;
- `r_max` and `element_size`.

When `element_size` is used, `r_max` must be an integer multiple of it. The
element boundaries are uniformly spaced and the final boundary is exactly
`r_max`. `n_points` may be one integer for all elements or one integer per
element, with at least three points per element.

The propagated radial samples exclude the two outer FEM-DVR endpoints, imposing
homogeneous Dirichlet boundary conditions. Consequently, `calculation.r` and
`result.radial_grid` contain the interior DVR nodes, not every element boundary.

```python
calculation = VibrationalRotorDiatomic(
    molecule,
    surfaces,
    r_max=12.0,
    element_size=2.0,
    n_points=11,
    l_max=12,
    fields=[pulse],
    time_step=0.25,
)
```

Internally the radial amplitudes are multiplied by the square root of the DVR
quadrature weights, so Euclidean inner products reproduce quadrature integrals.
Custom initial states and returned final states use physical, unweighted nodal
amplitudes.

## Initial state

`initial_state` may be:

- `"groundstate"`, equivalent to $(v,J)=(0,0)$;
- `RovibrationalState(vibrational=v, rotational=J)`;
- a tuple `(v, J)`;
- a complex array with shape `(len(calculation.ell), len(calculation.r))`.

Requested field-free eigenstates are obtained by diagonalizing the Hermitian
radial block for their $J$ channel. A custom state is normalized automatically.

## Complex absorbing potential

The `cap` argument accepts `None` or a callable with signature
`cap(radial_grid, r_max)`. The built-in cubic absorber is

$$W_\mathrm{CAP}(R)=\begin{cases}-i\eta\left(R-\alpha R_\max\right)^3,&R>\alpha R_\max,\\0,&R\leq\alpha R_\max.\end{cases}$$

It is configured with `CubicCAP(alpha, eta=1.0)`. A custom CAP must return one
finite value per interior radial node and may not have a positive imaginary
part. The CAP is excluded when constructing a field-free initial state and
included during real-time propagation. The state is not renormalized, so norm
loss represents absorbed probability. Crank–Nicolson remains applicable, but
the non-Hermitian Hamiltonian deliberately makes the propagation nonunitary.

## Observables and radial probability

`VibrationalRotorResult` stores the norm and the expectation values
$\langle R\rangle$, $\langle\cos\theta\rangle$,
$\langle\cos^2\theta\rangle$, $\langle R\cos\theta\rangle$, and
$\langle R\cos^2\theta\rangle$. With a CAP these are unnormalized moments of
the surviving wavefunction.

The sampled radial probability density is

$$P(R,t)=R^2\int|\Psi(R,\Omega,t)|^2d\Omega=\sum_\ell|u_\ell(R,t)|^2.$$

It is stored as `result.radial_probability`, with one row per sampled time, and
satisfies the quadrature relation

$$\sum_i w_iP(R_i,t)=\lVert\Psi(t)\rVert^2.$$

The result also contains the radial grid and weights, final wavefunction,
retained ℓ values, electric field, and iterative-solver diagnostics. Use
`sample_interval` to reduce the number of stored time samples and
`result.save("result.npz")` to write a compressed archive.

See [`vibrational_rotor_h2plus.py`](../examples/vibrational_rotor_h2plus.py) for
a complete calculation and plots.

