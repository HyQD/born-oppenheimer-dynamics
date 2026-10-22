"""Propagate a rigid H2+ rotor in a z-polarized pulse."""

import matplotlib.pyplot as plt
from electromagnetic_fields.dipole_fields import TrigonometricGaussianLaser

from born_oppenheimer_dynamics import (
    DiatomicMolecule,
    RigidRotorDiatomic,
    h2plus_aug_cc_pv5z_surfaces,
)

molecule = DiatomicMolecule(("H-1", "H-1"))
surfaces = h2plus_aug_cc_pv5z_surfaces()

pulse = TrigonometricGaussianLaser(
    E0=0.058,
    omega=0.057,
    phase=0.0,
    sigma=745.0,
    n=2,
    gauge="length",
)

calculation = RigidRotorDiatomic.from_surfaces(
    molecule,
    surfaces,
    bond_length=2.0,
    l_max=12,
    angular_sector="auto",
    initial_state="groundstate",
    fields=[pulse],
    final_time=4 * pulse.t_end,
    time_step=0.25,
    sample_interval=1,
    integrator_options={"rtol": 0.0, "atol": 1e-12},
)

result = calculation.run(progress=True)

figure, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
axes[0].plot(result.times, result.electric_field)
axes[0].set_ylabel(r"$E(t)$")
axes[1].plot(result.times, result.expectation_cos_theta_squared)
axes[1].set_ylabel(r"$\langle\cos^2\theta\rangle$")
axes[2].plot(result.times, result.angular_momentum_populations)
axes[2].set_ylabel(r"$P_J(t)$")
axes[2].set_xlabel("time (a.u.)")
axes[2].legend([rf"$J={angular_momentum}$" for angular_momentum in result.ell])
for axis in axes:
    axis.axvline(pulse.t_end, color="0.35", linestyle="--", linewidth=1.0)
figure.tight_layout()
plt.show()
