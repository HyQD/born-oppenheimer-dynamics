# H2+ electronic properties with PySCF

The optional Gaussian-type-orbital backend computes the electronic input used
by the rigid- and vibrational-rotor models for one-electron H2+. It provides the
Born--Oppenheimer potential and static or real, undamped dynamic
polarizabilities in atomic units.

Install the optional backend before using these functions:

```bash
python -m pip install -e ".[pyscf]"
```

PySCF supplies molecular integrals, while the installed Basis Set Exchange
package supplies the requested hydrogen basis. No network access is required
during a calculation.

## One-electron calculation

At each internuclear distance, the protons are placed on the body-fixed $z$
axis at $(0,0,-R/2)$ and $(0,0,R/2)$. The PySCF molecule has charge $+1$,
spin 1, and coordinates in bohr. The one-electron generalized eigenproblem is

$$H C = S C \varepsilon.$$

The total clamped-nuclei energy returned to the rotor models includes nuclear
repulsion:

$$V_{\mathrm{BO}}(R)=\varepsilon_0(R)+\frac{1}{R}.$$

Diffuse basis sets can make the AO overlap matrix nearly singular. The backend
diagonalizes $S$, removes eigenvectors below `linear_dependence_threshold`, and
solves the Hamiltonian in the retained orthonormal space. The default threshold
is $10^{-8}$. The smallest retained overlap eigenvalue, discarded dimension,
orthonormality error, and projected eigenproblem residual are stored as
diagnostics.

## Static and dynamic polarizability

For excitation energies $\Delta E_n=\varepsilon_n-\varepsilon_0$ and Cartesian
transition moments $\mu^i_{0n}$, the backend evaluates the finite-basis
sum-over-states expression

$$\alpha_{ij}(R;\omega)=\sum_{n>0}\frac{2\Delta E_n(R)\mu^i_{0n}(R)\mu^j_{n0}(R)}{\Delta E_n(R)^2-\omega^2}.$$

All retained electronic states contribute. Setting $\omega=0$ gives the static
polarizability. With the molecular axis on $z$, the rotor inputs are

$$\alpha_\parallel=\alpha_{zz},\qquad \alpha_\perp=\frac{\alpha_{xx}+\alpha_{yy}}{2}.$$

The complete Cartesian tensor is retained so that cylindrical symmetry and
off-diagonal components can be checked. Static sum-over-states results are
tested against four-point finite-field derivatives. Dynamic results are tested
against independent frequency-domain resolvent solves. Finite-field energy
derivatives alone do not provide frequency-dependent polarizabilities.

## Computing a radial data set

```python
from born_oppenheimer_dynamics import compute_h2plus_gto

data = compute_h2plus_gto(
    basis="aug-cc-pV5Z",
    radial_range=(0.1, 12.0, 0.1),
    frequencies=[0.0, 0.057],
    progress=True,
)
data.save("h2plus_aug_cc_pv5z.npz")
```

`radial_range=(r_min, r_max, dr)` is inclusive: the last point is exactly
`r_max`, and the interval must contain an integral number of steps. Frequencies
may be one non-negative scalar or a sequence and are interpreted as angular
frequencies in atomic units. The returned `DiatomicElectronicData` can be saved
and loaded without importing PySCF:

```python
from born_oppenheimer_dynamics import DiatomicElectronicData

data = DiatomicElectronicData.from_npz("h2plus_aug_cc_pv5z.npz")
```

The archive contains numerical diagnostics and metadata describing the basis,
backend, geometry, response method, and thresholds. It does not use pickled
Python objects.

## Resonance and continuum limitations

The dynamic tensor is real and undamped. A finite GTO basis discretizes
continuum-like states and cannot describe photoionization, linewidths, or the
imaginary absorptive response. The backend therefore rejects a requested
frequency at or above the vertical ionization threshold anywhere on the radial
grid.

It also rejects numerical poles when a requested frequency agrees, within the
configured absolute and relative tolerances, with a dipole-allowed transition.
Transitions below `transition_strength_threshold` do not trigger a pole for
that Cartesian component. Closest-transition energies, strengths, absolute
detunings, and relative detunings are retained for inspection.

Passing the numerical pole check does not prove that a laser pulse is
physically off resonance. That assessment also depends on the pulse bandwidth,
the field coupling $E_0|\mu_{0n}|$, and physical linewidths, none of which can
be inferred from the carrier frequency alone. Near a transition, use the
stored diagnostics and test basis, radial-grid, and pulse-parameter convergence.

At large $R$, low-lying parallel transitions can produce a strong physical
increase in $\alpha_\parallel$. Diffuse finite GTO bases can exaggerate or move
these features, especially close to dissociation or a finite-basis pole. Such
curves should not be called converged without comparison between suitably
diffuse basis sets.

## Converting data for the rotor models

The electronic result may contain several frequencies, while a real rotor
Hamiltonian uses one selected response curve. A single-frequency result is
selected automatically; a multi-frequency result requires an explicit value:

```python
surfaces = data.to_diatomic_surfaces(frequency=0.057)
```

Frequency matching uses tight numerical tolerances. Uncalculated frequencies
and complex response tensors are rejected rather than silently approximated.
The selected frequency and electronic-structure provenance are preserved in
the surface label and metadata. The H2+ permanent dipole is omitted and is
therefore treated as zero by both homonuclear rotor models.

```python
from born_oppenheimer_dynamics import (
    DiatomicMolecule,
    RigidRotorDiatomic,
    VibrationalRotorDiatomic,
)

molecule = DiatomicMolecule(("H-1", "H-1"))

rigid = RigidRotorDiatomic.from_surfaces(
    molecule,
    surfaces,
    bond_length=2.0,
    l_max=12,
    fields=[],
    final_time=1.0,
    time_step=0.25,
)

vibrational = VibrationalRotorDiatomic(
    molecule,
    surfaces,
    r_max=12.0,
    element_size=4.0,
    n_points=15,
    l_max=12,
    fields=[],
    final_time=1.0,
    time_step=0.25,
)
```

This selection is an effective single-carrier response. It is not a general
treatment of broadband or multicolor electronic response.

See
[`h2plus_gto_polarizability.py`](../examples/h2plus_gto_polarizability.py) for
the complete aug-cc-pV5Z calculation, the BO-energy plot, static and dynamic
polarizability plots, isolated-hydrogen reference lines, persistence, and both
rotor conversions.
