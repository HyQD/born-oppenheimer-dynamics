# H2+ reference curves

`h2+_aug-cc-pv5z_rmin=0.1_rmax=12.0.npz` contains clamped-nuclei H2+
electronic data with the protons at `(0, 0, -R/2)` and `(0, 0, R/2)`.

- Internuclear distances: 0.1 through 12.0 bohr in 0.1-bohr increments.
- Electronic basis: aug-cc-pV5Z.
- `E_R`: six one-electron Born-Oppenheimer energy curves including nuclear
  repulsion. The vibrational-rotor reference helper selects the first curve.
- `alpha`: finite-field polarizabilities. The x and z diagonal components are
  the perpendicular and parallel curves, respectively.

The polarizabilities were evaluated with a four-point finite-difference
formula using a field increment of 0.001 atomic units.
