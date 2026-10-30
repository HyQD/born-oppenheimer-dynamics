# born-oppenheimer-dynamics

Tools for Born-Oppenheimer dynamics.

## Documentation

The detailed documentation covers the physical models and their shared
numerical components:

- [Documentation overview](docs/index.md)
- [Installation](docs/installation.md)
- [Rigid-rotor model](docs/rigid_rotor.md)
- [Vibrational-rotor model](docs/vibrational_rotor.md)
- [Electric-field interaction](docs/electric_fields.md)
- [Crank–Nicolson time propagation](docs/time_propagation.md)

## Time-dependent vibrational rotor

The vibrational-rotor interface propagates a diatomic nuclear wavefunction on
an FEM-DVR internuclear grid and an axisymmetric spherical-harmonic basis. It
uses length-gauge fields from `electromagnetic-fields`, isotope masses from
`matter_field_params`, and exact Gaunt coefficients from `grid-lib`.

```python
from electromagnetic_fields.dipole_fields import SineSquareLaser

from born_oppenheimer_dynamics import (
    CubicCAP,
    DiatomicMolecule,
    VibrationalRotorDiatomic,
    h2plus_aug_cc_pv5z_surfaces,
)

molecule = DiatomicMolecule(("H-1", "H-1"))
surfaces = h2plus_aug_cc_pv5z_surfaces()
pulse = SineSquareLaser(
    E0=0.01,
    omega=0.057,
    ncycles=1,
    gauge="length",
)

calculation = VibrationalRotorDiatomic(
    molecule,
    surfaces,
    r_max=12.0,
    element_size=2.0,
    n_points=11,
    l_max=12,
    angular_sector="auto",
    initial_state="groundstate",
    fields=[pulse],
    cap=CubicCAP(alpha=0.8),
    final_time=pulse.time_support[1] + 20.0,
    time_step=0.25,
)
result = calculation.run(progress=True)
```

When `element_size` is supplied, `r_max` must be an integer multiple of it.
The resulting element boundaries are uniformly spaced and the final boundary
is exactly `r_max`.

The sampled radial density is available as `result.radial_probability`, with
shape `(len(result.times), len(result.radial_grid))`. For the reduced radial
amplitudes $u_\ell(R)$ used by the implementation,

$$P(R) = R^2 \int |\Psi(R, \Omega)|^2\,d\Omega = \sum_\ell |u_\ell(R)|^2.$$

It integrates to the sampled norm using `result.radial_weights`.
The unnormalized mixed moment $\langle R\cos^2\theta\rangle$ is available as
`result.expectation_r_cos_theta_squared`.

`cap` accepts either `None` or a callable with signature
`cap(radial_grid, r_max)`. `CubicCAP(alpha, eta=1.0)` supplies the potential

$$W(R) = -i\eta\left(R-\alpha R_{\max}\right)^3$$

for $R>\alpha R_{\max}$ and zero elsewhere. The CAP is included only during
real-time propagation; initial rovibrational states are constructed from the
physical Hermitian Hamiltonian. Norm loss represents absorbed probability.
Expectation values remain unnormalized moments of the surviving wavefunction.

For a homonuclear molecule in its rotational ground state, `"auto"` retains
only even angular momenta when all odd-rank interactions vanish. A permanent
dipole or first hyperpolarizability activates the full angular space. The
electronic-surface data must cover the requested outer radial boundary;
long-range extrapolation is deliberately not performed.

The packaged H2+ reference data use aug-cc-pV5Z electronic curves from 0.1 to
12 bohr at 0.1-bohr spacing. `DiatomicSurfaces.from_npz()` loads the established
`R`, `E_R`, and `alpha` array layout for other datasets.

## Time-dependent rigid rotor

`RigidRotorDiatomic` provides a separate fixed-bond-length model while sharing
the vibrational rotor's angular operators, length-gauge field interaction, and
Crank-Nicolson propagator. Its field-free Hamiltonian is

$$H_0 = B\mathbf{J}^2, \qquad B = \frac{1}{2\mu R_0^2}.$$

The rotational constant may be supplied directly in atomic units. Alternatively,
the reduced mass is inferred from `DiatomicMolecule` and the rotational constant
is calculated from `bond_length`. Molecular-response values can be supplied
directly as scalars or interpolated from `DiatomicSurfaces` at the chosen bond
length. The electronic energy is omitted because it contributes only a global
phase when the bond length is fixed.

```python
from born_oppenheimer_dynamics import RigidRotorDiatomic

calculation = RigidRotorDiatomic.from_surfaces(
    molecule,
    surfaces,
    bond_length=2.0,
    l_max=12,
    initial_state="groundstate",
    fields=[pulse],
    final_time=pulse.time_support[1] + 20.0,
    time_step=0.25,
)
result = calculation.run(progress=True)
```

The result contains orientation, alignment, and the population of every
retained angular-momentum channel at each sampled time. As in the vibrational
model, `angular_sector="auto"` uses a parity-restricted basis when the initial
state and interaction permit it, and otherwise selects the full basis. A rigid
rotor has no radial grid or CAP.

## Installation

The package supports Python 3.12 through 3.14. A clean virtual environment is
recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the four HyQD dependencies directly from GitHub, followed by this
package:

```bash
python -m pip install \
    "git+https://github.com/HyQD/electromagnetic-fields.git" \
    "git+https://github.com/HyQD/grid-lib.git" \
    "git+https://github.com/HyQD/integrator-lib.git" \
    "git+https://github.com/HyQD/matter_field_params.git"

python -m pip install \
    "git+https://github.com/HyQD/born-oppenheimer-dynamics.git"
```

From a local checkout, install the package and its runtime dependencies with:

```bash
python -m pip install .
```

For development, install the package in editable mode with its test dependencies:

```bash
python -m pip install -e ".[dev]"
```

See the [installation guide](docs/installation.md) for editable installation
of all HyQD dependencies, revision pinning, and verification commands.
