"""Propagate the H2+ rovibrational ground state in a z-polarized pulse."""

import matplotlib.pyplot as plt
from electromagnetic_fields.dipole_fields import TrigonometricGaussianLaser

from born_oppenheimer_dynamics import (
    DiatomicMolecule,
    VibrationalRotor,
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

calculation = VibrationalRotor(
    molecule,
    surfaces,
    nodes=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    n_points=11,
    l_max=12,
    angular_sector="auto",
    initial_state="groundstate",
    fields=[pulse],
    final_time=pulse.time_support[1],
    time_step=0.25,
    integrator_options={"rtol": 0.0, "atol": 1e-12},
)
result = calculation.run(progress=True)
result.save("h2plus_vibrational_rotor.npz")

figure, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
axes[0].plot(result.times, result.electric_field)
axes[0].set_ylabel(r"$E(t)$")
axes[1].plot(result.times, result.expectation_cos_theta_squared)
axes[1].set_ylabel(r"$\langle\cos^2\theta\rangle$")
axes[2].plot(result.times, result.expectation_r)
axes[2].set_ylabel(r"$\langle R\rangle$")
axes[3].semilogy(result.times, abs(1.0 - result.norms))
axes[3].set_ylabel("norm error")
axes[3].set_xlabel("time (a.u.)")
figure.tight_layout()
plt.show()
