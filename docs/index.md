# Born–Oppenheimer dynamics documentation

`born-oppenheimer-dynamics` provides time-dependent nuclear dynamics models for
diatomic molecules. All quantities are expressed in atomic units unless stated
otherwise.

The documentation is divided into the physical models and the numerical
components shared by them:

- [Installation](installation.md)
- [Rigid-rotor model](rigid_rotor.md)
- [Vibrational-rotor model](vibrational_rotor.md)
- [Electric-field interaction](electric_fields.md)
- [Crank–Nicolson time propagation](time_propagation.md)

The high-level interfaces are designed around complete calculations:
`RigidRotorDiatomic` propagates only the molecular orientation, whereas
`VibrationalRotorDiatomic` propagates the internuclear coordinate and the
orientation together. Both models currently assume an axisymmetric wavefunction
and a linearly polarized electric field along the laboratory-fixed $z$ axis.

Runnable calculations are available in the [`examples`](../examples) directory.

