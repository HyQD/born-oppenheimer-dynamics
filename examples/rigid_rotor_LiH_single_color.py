r"""Reproduce the single-color rigid-rotor calculation in Fig. 2(d-f).

Reference
---------
Q.-Q. Hong et al., "Quantum control of field-free molecular orientation",
Phys. Chem. Chem. Phys. 25, 32763-32777 (2023).
https://doi.org/10.1039/D3CP03115B

The published parameters are converted to atomic units as follows:

* ``B [cm^-1] -> B [Eh]``: ``B * 100 * h * c / Eh``;
* ``E [V/m] -> E [a.u.]``: ``E / E_atomic``;
* ``tau [s] -> tau [a.u.]``: ``tau / t_atomic``;
* ``lambda [m] -> omega [a.u.]``: ``2*pi*c*t_atomic / lambda``.

The paper's electric-field amplitude, rather than its factor-of-two larger
quoted intensity, is used. Its Gaussian parameter ``tau = 30 fs`` is the
standard deviation of the field envelope. The paper does not report its
time step, angular cutoff, or Gaussian truncation; the converged numerical
choices below use ``dt = 2 a.u.``, ``J_max = 12``, and start at ``-5*tau``.
"""

import matplotlib.pyplot as plt
import numpy as np
from electromagnetic_fields.dipole_fields import GaussianLaser
from scipy.constants import c, h, physical_constants

from born_oppenheimer_dynamics import DiatomicMolecule, RigidRotorDiatomic


# Atomic-unit conversion factors in SI units.
HARTREE = physical_constants["Hartree energy"][0]
ATOMIC_TIME = physical_constants["atomic unit of time"][0]
ATOMIC_ELECTRIC_FIELD = physical_constants["atomic unit of electric field"][0]

# Parameters reported for LiH in Table 1 and the caption of Fig. 2.
rotational_constant = 7.51 * 100 * h * c / HARTREE
fundamental_field = 2.0e10 / ATOMIC_ELECTRIC_FIELD
pulse_width = 30.0e-15 / ATOMIC_TIME
fundamental_frequency = 2 * np.pi * c * ATOMIC_TIME / (800.0e-9)
final_time = 6.0e-12 / ATOMIC_TIME

# GaussianLaser uses sin(omega*t + phase), whereas Eq. (10) uses cos(omega*t).
fundamental_pulse = GaussianLaser(
    E0=fundamental_field,
    omega=fundamental_frequency,
    phase=np.pi / 2,
    sigma=pulse_width,
    tc=0.0,
    gauge="length",
)

calculation = RigidRotorDiatomic(
    DiatomicMolecule(("Li-7", "H-1")),
    rotational_constant=rotational_constant,
    dipole=0.0,  # Permanent-dipole coupling is neglected in Eq. (9).
    polarizability_parallel=25.4,
    polarizability_perpendicular=29.5,
    beta_parallel=676.0,
    beta_perpendicular=356.0,
    l_max=12,
    initial_state="groundstate",
    fields=[fundamental_pulse],
    initial_time=-5 * pulse_width,
    final_time=final_time,
    time_step=2.0,
    sample_interval=10,
    integrator_options={"rtol": 0.0, "atol": 1e-12},
)

result = calculation.run(progress=True)
times_ps = result.times * ATOMIC_TIME / 1.0e-12

figure, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
axes[0].plot(times_ps, result.expectation_cos_theta)
axes[0].set_ylabel(r"$\langle\cos\theta\rangle$")
# Do not set ylim=(-1, 1): that scale hides the small orientation oscillations.

axes[1].plot(times_ps, result.expectation_cos_theta_squared)
axes[1].set_ylabel(r"$\langle\cos^2\theta\rangle$")
axes[1].set_ylim(0.0, 0.9)

for angular_momentum in (0, 2, 4, 6):
    channel = np.flatnonzero(result.ell == angular_momentum).item()
    axes[2].plot(
        times_ps,
        result.angular_momentum_populations[:, channel],
        label=rf"$J={angular_momentum}$",
    )
axes[2].set_ylabel("population")
axes[2].set_xlabel("time (ps)")
axes[2].set_ylim(0.0, 1.0)
axes[2].legend(ncols=2)

for axis in axes:
    axis.axvline(0.0, color="0.5", linestyle="--", linewidth=1.0)
    axis.set_xlim(times_ps[0], 6.0)
figure.tight_layout()
plt.show()
