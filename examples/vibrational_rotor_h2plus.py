"""Propagate the H2+ rovibrational ground state in a z-polarized pulse."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from electromagnetic_fields.dipole_fields import TrigonometricGaussianLaser

from born_oppenheimer_dynamics import (
    CubicCAP,
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
t_final = 4*t_end_pulse

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
    cap=CubicCAP(alpha=0.8, eta=1.0),
    final_time=t_final,
    time_step=1/4,
    sample_interval=1,
    integrator_options={"rtol": 0.0, "atol": 1e-12},
)

mebibyte = 1024**2
n_samples = calculation.n_steps + 1
radial_probability_bytes = n_samples * calculation.r.nbytes
float_bytes = np.dtype(np.float64).itemsize
integer_bytes = np.dtype(np.int64).itemsize
complex_bytes = np.dtype(np.complex128).itemsize
solver_method_bytes = np.dtype("<U11").itemsize
sampled_scalar_bytes = n_samples * (
    9 * float_bytes + integer_bytes + solver_method_bytes
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

figure, axes = plt.subplots(5, 1, sharex=True, figsize=(8, 10))
axes[0].plot(result.times, result.electric_field)
axes[0].set_ylabel(r"$E(t)$")
axes[1].plot(result.times, result.expectation_cos_theta_squared)
axes[1].set_ylabel(r"$\langle\cos^2\theta\rangle$")
axes[2].plot(result.times, result.expectation_r)
axes[2].set_ylabel(r"$\langle R\rangle$")
radial_alignment_correlation = (
    result.expectation_r_cos_theta_squared
    - result.expectation_r * result.expectation_cos_theta_squared
)
axes[3].plot(result.times, radial_alignment_correlation)
axes[3].set_ylabel(
    r"$\langle R\cos^2\theta\rangle$"
    "\n"
    r"$-\langle R\rangle\langle\cos^2\theta\rangle$"
)
axes[4].semilogy(result.times, abs(1.0 - result.norms))
axes[4].set_ylabel("absorbed probability")
axes[4].set_xlabel("time (a.u.)")
for axis in axes:
    axis.axvline(t_end_pulse, color="0.35", linestyle="--", linewidth=1.0)
figure.tight_layout()

radial_figure, radial_axis = plt.subplots(figsize=(6, 4))
legend_handles = []
if len(result.radial_probability) > 1:
    (initial_line,) = radial_axis.semilogy(
        result.radial_grid,
        result.radial_probability[0],
        linestyle="--",
        label=r"$P(R;t=0)$",
    )
    legend_handles.append(initial_line)
(final_line,) = radial_axis.semilogy(
    result.radial_grid,
    result.radial_probability[-1],
    label=r"$P(R;t=t_f)$",
)
legend_handles.append(final_line)
radial_axis.set_xlabel(r"$R$ (bohr)")
radial_axis.set_ylabel(r"$P(R)$")
radial_axis.set_ylim(bottom=1e-20)
radial_axis.set_title("Radial probability")
if calculation.cap_potential is not None:
    cap_profile = -calculation.cap_potential.imag
    cap_label = r"$W_{\mathrm{CAP}}(R)$"
    if isinstance(calculation.cap, CubicCAP):
        cap_label = (
            rf"$W_{{\mathrm{{CAP}}}}(R;\alpha={calculation.cap.alpha:g},"
            rf"\eta={calculation.cap.eta:g})$"
        )
    cap_axis = radial_axis.twinx()
    (cap_line,) = cap_axis.plot(
        result.radial_grid,
        cap_profile,
        color="0.35",
        label=cap_label,
    )
    legend_handles.append(cap_line)
    cap_axis.fill_between(
        result.radial_grid,
        0.0,
        cap_profile,
        color="0.7",
        alpha=0.35,
    )
    cap_axis.set_ylabel(r"CAP strength $-\operatorname{Im}W(R)$", color="0.35")
    cap_axis.tick_params(axis="y", colors="0.35")
    cap_axis.set_ylim(bottom=0.0)
radial_axis.legend(handles=legend_handles, loc="lower left")
radial_figure.tight_layout()
plt.show()
