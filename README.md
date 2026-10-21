# born-oppenheimer-dynamics

Tools for Born-Oppenheimer dynamics.

## Time-dependent vibrational rotor

The vibrational-rotor interface propagates a diatomic nuclear wavefunction on
an FEM-DVR internuclear grid and an axisymmetric spherical-harmonic basis. It
uses length-gauge fields from `electromagnetic-fields`, isotope masses from
`matter_field_params`, and exact Gaunt coefficients from `grid-lib`.

```python
from electromagnetic_fields.dipole_fields import SineSquareLaser

from born_oppenheimer_dynamics import (
    DiatomicMolecule,
    VibrationalRotor,
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

calculation = VibrationalRotor(
    molecule,
    surfaces,
    nodes=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    n_points=11,
    l_max=12,
    angular_sector="auto",
    initial_state="groundstate",
    fields=[pulse],
    final_time=pulse.time_support[1] + 20.0,
    time_step=0.25,
)
result = calculation.run(progress=True)
```

For a homonuclear molecule in its rotational ground state, `"auto"` retains
only even angular momenta when all odd-rank interactions vanish. A permanent
dipole or first hyperpolarizability activates the full angular space. The
electronic-surface data must cover the requested outer radial boundary;
long-range extrapolation is deliberately not performed.

The packaged H2+ reference data use aug-cc-pV5Z electronic curves from 0.1 to
12 bohr at 0.1-bohr spacing. `DiatomicSurfaces.from_npz()` loads the established
`R`, `E_R`, and `alpha` array layout for other datasets.

## Installation

Install the package and its runtime dependencies with:

```bash
python -m pip install .
```

For development, install the package in editable mode with its test dependencies:

```bash
python -m pip install -e ".[dev]"
```
