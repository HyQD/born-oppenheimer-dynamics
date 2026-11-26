# H2+ GTO electronic-properties implementation plan

## Status and decisions

This document specifies the agreed initial implementation. It is a plan only;
no implementation is included in this document.

Implementation status:

- Phases 1–5: implemented on the dedicated electronic-properties feature
  branch.

The first electronic-structure backend will:

- be specialized for one-electron H2+;
- use PySCF for Gaussian-basis molecular integrals;
- obtain basis sets through the Basis Set Exchange Python package;
- compute static and undamped, off-resonant dynamic polarizabilities by an
  explicit sum over states;
- accept one frequency or multiple frequencies;
- reject numerical resonances and frequencies at or above the ionization
  threshold;
- keep PySCF and Basis Set Exchange as optional dependencies;
- return backend-neutral data that a future grid-TDSE backend can also produce.

All distances, energies, frequencies, fields, and response properties use atomic
units.

## Goals

1. Generate the electronic input needed by `RigidRotorDiatomic` and
   `VibrationalRotorDiatomic` for H2+.
2. Compute the ground-state Born–Oppenheimer potential-energy curve.
3. Compute static or frequency-dependent parallel and perpendicular
   polarizabilities.
4. Preserve enough diagnostics and provenance to assess basis convergence and
   proximity to electronic resonances.
5. Convert one selected real-frequency response into the existing
   `DiatomicSurfaces` representation.
6. Establish a result format that can later be populated by a grid-TDSE
   calculation.

## Initial non-goals

- General many-electron electronic structure.
- Damped or complex polarizabilities.
- Response above the ionization threshold.
- Continuum photoabsorption.
- Frequency-dependent first hyperpolarizabilities.
- General broadband or multicolor response in the rotor Hamiltonian.
- The grid-TDSE backend itself.

## Scientific method

### Geometry and one-electron Hamiltonian

At every internuclear distance, place the nuclei on the body-fixed $z$ axis:

$$\mathbf R_A=(0,0,-R/2),\qquad \mathbf R_B=(0,0,R/2).$$

The PySCF molecule must explicitly use:

~~~python
charge = 1
spin = 1
unit = "Bohr"
~~~

For one-electron H2+, no SCF iteration is required. PySCF supplies the
one-electron Hamiltonian $H$, AO overlap $S$, position integrals, and nuclear
repulsion. Solve

$$H C = S C\varepsilon.$$

The ground-state Born–Oppenheimer energy is

$$V_\mathrm{BO}(R)=\varepsilon_0(R)+E_\mathrm{nuc}(R)=\varepsilon_0(R)+\frac{1}{R}.$$

### Linear-dependence handling

Diffuse basis sets and short bond lengths can make the AO overlap matrix nearly
singular. Diagonalize $S$, discard overlap eigenvectors below a configurable
threshold, and solve the Hamiltonian in the retained orthonormal basis.

Record at every $R$:

- the smallest retained overlap eigenvalue;
- the number of discarded AO combinations;
- the retained electronic dimension.

Use \(10^{-8}\) as the initial default. At the short-range aug-cc-pV5Z
endpoint \(R=0.1\), it removes one near-dependent combination without changing
the reference BO energy at the tested precision. Reassess convergence of the
polarizability with respect to this threshold in Phase 3.

### Dipole matrix elements

Transform all three Cartesian position operators into the electronic
eigenbasis. The sign of the electronic dipole does not affect diagonal
polarizabilities because transition moments occur in products.

Use the molecular midpoint as the coordinate origin. For symmetric H2+, the
permanent dipole is zero. Validate numerically that

$$\mu_{00,x}\simeq\mu_{00,y}\simeq\mu_{00,z}\simeq0.$$

### Static and dynamic polarizability

Define

$$\Delta E_n(R)=\varepsilon_n(R)-\varepsilon_0(R).$$

For real orbitals and an undamped real frequency, evaluate the symmetric
sum-over-states tensor

$$\alpha_{ij}(R;\omega)=\sum_{n>0}\frac{2\Delta E_n(R)\,\mu^i_{0n}(R)\mu^j_{n0}(R)}{\Delta E_n(R)^2-\omega^2}.$$

All retained finite-basis eigenstates must contribute. Do not truncate the
response to a small requested number of excited states.

Interpret $\omega=0$ as the static polarizability. For a diatomic aligned with
the body-fixed $z$ axis, expose

$$\alpha_\parallel(R;\omega)=\alpha_{zz}(R;\omega),\qquad \alpha_\perp(R;\omega)=\frac{\alpha_{xx}(R;\omega)+\alpha_{yy}(R;\omega)}{2}.$$

Store the full tensor as well as these reduced components so that cylindrical
symmetry and off-diagonal elements can be checked.

## Resonance and validity policy

### Dipole-allowed transitions

Only transitions with non-negligible strength in a requested Cartesian
component should trigger a resonance for that component. Define

$$s_{nq}(R)=|\mu^q_{0n}(R)|^2,\qquad q\in\{x,y,z\}.$$

Use a configurable transition-strength threshold and report separate
diagnostics for parallel and perpendicular response.

### Numerical pole detection

For every requested $\omega$, $R$, state, and active component, calculate

$$\delta_{nq}(R,\omega)=\left|\Delta E_n(R)-|\omega|\right|.$$

Reject the calculation when

$$\delta_{nq}\leq\epsilon_\mathrm{abs}+\epsilon_\mathrm{rel}\max\left(\Delta E_n,|\omega|\right).$$

Initial candidate defaults are:

~~~python
resonance_atol = 1e-8
resonance_rtol = 1e-6
~~~

These defaults prevent numerical poles; they do not establish that a physical
laser pulse is far off resonance.

The exception must identify:

- $R$;
- the requested frequency;
- the closest excitation energy;
- the detuning;
- the transition strength;
- the affected tensor component.

### Physical off-resonance diagnostics

Whether a pulse is physically off resonant also depends on its bandwidth,
field strength, and physical linewidth. The electronic calculation receives
only frequencies and must not claim a definitive pulse-level classification.

Store:

- the closest dipole-allowed excitation at every $R$ and frequency;
- the corresponding absolute and relative detuning;
- the minimum detuning over the requested radial interval.

A later pulse-aware helper may compare these data with the pulse bandwidth and
the characteristic coupling $E_0|\mu_{0n}|$.

### Ionization threshold

An undamped finite GTO calculation is not an adequate representation of
continuum absorption. For one-electron H2+, determine the vertical ionization
threshold from the ground-state electronic energy relative to the zero-energy
continuum.

Record the threshold at every $R$ and reject a requested frequency if it
reaches or exceeds that threshold anywhere on the radial grid. The error must
report the first offending $R$, frequency, and threshold.

Continuum and complex response are deferred to the future grid-TDSE backend.

## Proposed public API

### Calculation function

Provide a backend-specific convenience function:

~~~python
result = compute_h2plus_gto(
    basis="aug-cc-pV5Z",
    radial_range=(0.1, 12.0, 0.1),
    frequencies=[0.0, 0.057],
)
~~~

The initial options should include:

~~~python
compute_h2plus_gto(
    *,
    basis,
    radial_range,
    frequencies=0.0,
    linear_dependence_threshold=1e-8,
    transition_strength_threshold=...,
    resonance_atol=1e-8,
    resonance_rtol=1e-6,
    progress=False,
)
~~~

`frequencies` accepts one finite non-negative scalar or a finite
one-dimensional sequence. Normalize it internally to a one-dimensional array.
Negative frequencies are unnecessary for the present symmetric undamped
response.

### Radial range

Interpret

~~~python
radial_range = (r_min, r_max, dr)
~~~

as an inclusive grid. Require:

- $r_{\min}>0$;
- $r_{\max}\geq r_{\min}$;
- $dr>0$;
- $(r_{\max}-r_{\min})/dr$ to be integral within a tight tolerance.

Construct the grid using `linspace`, rather than accumulated `arange`, so
that the final point is exactly $r_{\max}$.

### Result object

Introduce an immutable backend-neutral result, provisionally named
`DiatomicElectronicData`. It should contain:

~~~text
internuclear_distance          (n_R,)
frequencies                    (n_frequency,)
electronic_energy              (n_R,)
polarizability                 (n_frequency, n_R, 3, 3)
polarizability_parallel        (n_frequency, n_R)
polarizability_perpendicular   (n_frequency, n_R)
ionization_threshold           (n_R,)
minimum_excitation_energy      (n_R,)
resonance/detuning diagnostics
linear-dependence diagnostics
metadata
~~~

Metadata should include:

- backend name and version;
- PySCF version;
- Basis Set Exchange version;
- requested and normalized basis-set name;
- basis-set version or identifying metadata when available;
- molecular charge, spin, geometry convention, and units;
- all numerical thresholds.

Provide compressed NPZ persistence without pickled Python objects:

~~~python
result.save("h2plus_aug-cc-pv5z.npz")
result = DiatomicElectronicData.from_npz(
    "h2plus_aug-cc-pv5z.npz"
)
~~~

Store structured metadata as JSON text.

## Conversion to rotor surfaces

The electronic result may contain multiple frequencies, whereas
`DiatomicSurfaces` contains one parallel and one perpendicular
polarizability curve.

When exactly one frequency is stored, allow:

~~~python
surfaces = result.to_diatomic_surfaces()
~~~

When multiple frequencies are stored, require:

~~~python
surfaces = result.to_diatomic_surfaces(frequency=0.057)
~~~

Rules:

1. With one stored frequency, omitting `frequency` selects it automatically.
2. With multiple stored frequencies, `frequency` is required.
3. Match a requested frequency using a documented tight numerical tolerance.
4. Reject frequencies that were not calculated.
5. Record the selected frequency in the surface label or metadata.
6. Do not silently discard an imaginary part if complex response is introduced
   later.
7. Follow the existing homonuclear convention for the zero H2+ permanent
   dipole.

For a single off-resonant carrier, the selected
$\alpha(R;\omega_\mathrm{laser})$ can be used as the effective response curve.
A selected curve is not a general treatment of broadband or multicolor
response; document this limitation.

## Package layout

Use a backend-oriented layout:

~~~text
born_oppenheimer_dynamics/
└── electronic_structure/
    ├── __init__.py
    ├── results.py
    └── pyscf_gto.py
~~~

`results.py` must not import PySCF. Result loading and surface conversion must
remain available when the optional electronic-structure dependencies are not
installed.

## Optional dependencies

Add an optional dependency group:

~~~toml
[project.optional-dependencies]
pyscf = [
    "pyscf>=2.14,<3",
    "basis-set-exchange",
]
~~~

Choose the exact Basis Set Exchange constraint after checking the supported
development environment. Users install the backend with:

~~~bash
python -m pip install -e ".[pyscf]"
~~~

Importing the base package and loading previously generated data must continue
to work without this extra.

## Basis Set Exchange handling

1. Resolve only the hydrogen basis through `basis_set_exchange`.
2. Resolve and parse the basis once, outside the radial loop.
3. Use a PySCF-compatible NWChem representation or PySCF's BSE conversion
   utility.
4. Preserve BSE basis metadata and references where practical.
5. Produce a clear error for an unknown basis or a basis without hydrogen.

The calculation must not require network access; the installed BSE Python data
package is the source.

## Implementation phases

### Phase 1: result and grid infrastructure

- Add the optional dependency group.
- Implement inclusive radial-range validation.
- Add the backend-neutral result dataclass and NPZ serialization.
- Add unit tests that do not require PySCF.

### Phase 2: H2+ GTO electronic solver

- Add lazy optional-dependency imports.
- Resolve and parse the BSE basis.
- Build centered H2+ geometries with the correct charge and spin.
- Implement overlap orthogonalization and linear-dependence removal.
- Compute the total ground-state Born–Oppenheimer energy.
- Transform Cartesian dipole matrices to the electronic eigenbasis.

### Phase 3: response and validity checks

- Implement static and multi-frequency sum-over-states polarizability.
- Reduce the tensor to parallel and perpendicular curves.
- Implement symmetry diagnostics.
- Implement dipole-strength filtering and numerical-pole rejection.
- Compute and enforce the ionization threshold.
- Store closest-transition and detuning diagnostics.
- Validate the static sum-over-states response against four-point finite-field
  dipole derivatives over a converged range of field strengths.
- Validate the dynamic sum-over-states response against an equivalent
  frequency-domain resolvent calculation.

### Phase 4: rotor integration

- Implement conversion to `DiatomicSurfaces`.
- Support automatic selection for one stored frequency.
- Require explicit selection for multi-frequency data.
- Preserve frequency and electronic-structure provenance.

### Phase 5: documentation and example

- Document the one-electron method and response formula.
- Explain resonance, pulse-bandwidth, and continuum limitations.
- Add an H2+ example using aug-cc-pV5Z.
- Plot $V_\mathrm{BO}(R)$, $\alpha_\parallel(R;\omega)$, and
  $\alpha_\perp(R;\omega)$.
- Demonstrate conversion to both rotor models.

## Validation and tests

### Unit tests

- The inclusive radial grid reaches $r_{\max}$ exactly.
- Invalid radial ranges and frequencies are rejected.
- Scalar and sequence frequencies produce consistent shapes.
- Unknown BSE bases fail clearly.
- Result serialization round-trips without pickle.
- Single- and multi-frequency surface selection obey the API rules.
- Base-package imports work without the optional dependencies.

### Electronic-structure tests

- The PySCF molecule has charge $+1$, spin $1$, and centered nuclei.
- $V_\mathrm{BO}(R)=\varepsilon_0+1/R$.
- The permanent dipole and off-diagonal polarizabilities vanish within
  tolerance.
- $\alpha_{xx}$ and $\alpha_{yy}$ agree.
- $\alpha(\omega=0)$ agrees with an independent four-point finite-field
  calculation at representative bond lengths.
- Static aug-cc-pV5Z curves agree with the packaged H2+ reference data within
  documented tolerances.
- Results are stable with respect to the linear-dependence threshold.
- Frequencies close to dipole-allowed transitions raise informative errors.
- Frequencies at or above the ionization threshold are rejected.
- Frequencies far below all allowed transitions produce finite real tensors.

### Convergence checks

The example or development tests should compare at least two diffuse
correlation-consistent basis sets. Dynamic response must not be presented as
converged solely because the numerical sum is finite.

## Acceptance criteria

The initial feature is complete when:

1. A user can request one or several static/off-resonant frequencies for an
   inclusive H2+ radial interval.
2. The calculation uses BSE-provided GTO data and PySCF integrals.
3. The ground-state energy and response pass the symmetry, finite-field, and
   packaged-reference checks.
4. Numerical resonances and continuum frequencies cannot silently produce
   rotor inputs.
5. The output can be saved, loaded without PySCF, and converted to
   `DiatomicSurfaces`.
6. The optional dependency boundary and physical limitations are documented.
