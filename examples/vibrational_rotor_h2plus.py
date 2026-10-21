"""Propagate the H2+ rovibrational ground state in a z-polarized pulse."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from electromagnetic_fields.dipole_fields import TrigonometricGaussianLaser

from born_oppenheimer_dynamics import (
    DiatomicMolecule,
    VibrationalRotorDiatomic,
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
t_end_pulse = pulse.t_end
t_final = t_end_pulse

r_max = 8.0
element_size = 0.2
n_points_pr_element = 3

calculation = VibrationalRotorDiatomic(
    molecule,
    surfaces,
    r_max=r_max,
    element_size=element_size,
    n_points=n_points_pr_element,
    l_max=12,
    angular_sector="auto",
    initial_state="groundstate",
    fields=[pulse],
    final_time=t_final,
    time_step=0.25,
    sample_interval=1,
    integrator_options={"rtol": 0.0, "atol": 1e-12},
)

print("FEM-DVR element boundaries:", calculation.grid.nodes)
print("Active radial grid (Dirichlet endpoints omitted):", calculation.r)
print(f"Number of radial grid points: {calculation.r.size}")

mebibyte = 1024**2
n_samples = calculation.n_steps + 1
radial_probability_bytes = n_samples * calculation.r.nbytes
float_bytes = np.dtype(np.float64).itemsize
integer_bytes = np.dtype(np.int64).itemsize
complex_bytes = np.dtype(np.complex128).itemsize
solver_method_bytes = np.dtype("<U11").itemsize
sampled_scalar_bytes = n_samples * (
    8 * float_bytes + integer_bytes + solver_method_bytes
)
final_state_bytes = calculation.ell.size * calculation.r.size * complex_bytes
metadata_bytes = (
    2 * calculation.r.nbytes
    + calculation.ell.nbytes
    + float_bytes
    + integer_bytes
)
estimated_result_bytes = (
    radial_probability_bytes
    + sampled_scalar_bytes
    + final_state_bytes
    + metadata_bytes
)
print(
    "Estimated uncompressed radial-probability history: "
    f"{radial_probability_bytes / mebibyte:.2f} MiB "
    f"({n_samples} time samples x {calculation.r.size} radial points)"
)
print(
    "Estimated uncompressed total result-array payload: "
    f"{estimated_result_bytes / mebibyte:.2f} MiB"
)

result = calculation.run(progress=True)
output_path = Path("h2plus_vibrational_rotor.npz")
result.save(output_path)
result_array_bytes = sum(
    value.nbytes for value in vars(result).values() if hasattr(value, "nbytes")
)
print(f"Uncompressed result-array payload: {result_array_bytes / mebibyte:.2f} MiB")
print(f"Compressed NPZ file: {output_path.stat().st_size / mebibyte:.2f} MiB")

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

radial_figure, radial_axis = plt.subplots(figsize=(6, 4))
radial_axis.plot(result.radial_grid, result.radial_probability[-1])
radial_axis.set_xlabel(r"$R$ (bohr)")
radial_axis.set_ylabel(r"$P(R)$")
radial_axis.set_title(f"Final radial probability, $t={result.times[-1]:.1f}$ a.u.")
radial_figure.tight_layout()
plt.show()
