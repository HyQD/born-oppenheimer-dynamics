"""Plot static and dynamic GTO polarizabilities of H2+."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, h, physical_constants

from born_oppenheimer_dynamics import (
    compute_h2plus_gto,
    compute_hydrogen_gto_polarizability,
)


basis = "aug-cc-pV5Z"
radial_range = (0.1, 12.0, 0.1)
frequency_800_nm = 0.057

# In atomic units, hbar = 1, so the angular frequency has the same numerical
# value as the photon energy. Hence omega [a.u.] = h*c/(lambda*E_h).
wavelength_1300_nm = 1300.0
wavelength_400_nm = 400.0
hartree_energy_joule = physical_constants["Hartree energy"][0]
frequency_1300_nm = (
    h * c / (wavelength_1300_nm * 1e-9) / hartree_energy_joule
)
# Treat 400 nm as the second harmonic of the nominal 800 nm frequency.
frequency_400_nm = 2 * frequency_800_nm
frequencies = [
    0.0,
    frequency_1300_nm,
    frequency_800_nm,
    frequency_400_nm,
]

result = compute_h2plus_gto(
    basis=basis,
    radial_range=radial_range,
    frequencies=frequencies,
    progress=True,
)
result.save("h2plus_gto_polarizability.npz")
hydrogen_polarizability = compute_hydrogen_gto_polarizability(
    basis=basis,
    frequencies=frequencies[1:],
)
hydrogen_isotropic = np.trace(
    hydrogen_polarizability,
    axis1=-2,
    axis2=-1,
) / 3

parallel = result.polarizability_parallel
perpendicular = result.polarizability_perpendicular
anisotropic = parallel - perpendicular

figure, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))
quantities = (
    (parallel, r"$\alpha_\parallel(R;\omega)$"),
    (perpendicular, r"$\alpha_\perp(R;\omega)$"),
    (
        anisotropic,
        r"$\Delta\alpha(R;\omega)"
        r"=\alpha_\parallel-\alpha_\perp$",
    ),
)
for axis, (values, ylabel) in zip(axes, quantities):
    axis.plot(
        result.internuclear_distance,
        values[0],
        linestyle="--",
        label="Static",
    )
    line_1300, = axis.plot(
        result.internuclear_distance,
        values[1],
        label=rf"${wavelength_1300_nm:g}$ nm",
    )
    line_800, = axis.plot(
        result.internuclear_distance,
        values[2],
        label=r"$800$ nm",
    )
    line_400, = axis.plot(
        result.internuclear_distance,
        values[3],
        label=rf"${wavelength_400_nm:g}$ nm",
    )
    if axis is axes[0]:
        for line, atomic_value in zip(
            (line_1300, line_800, line_400),
            hydrogen_isotropic,
        ):
            axis.axhline(
                atomic_value,
                color=line.get_color(),
                linestyle="--",
                linewidth=1.5,
            )
    axis.set_ylabel(ylabel + " (a.u.)")
    axis.grid(alpha=0.25)

# The parallel response grows strongly near the H2+ dissociation limit, while
# the dynamic response changes sign near resonances. A symmetric logarithmic
# scale keeps both features visible.
axes[0].set_yscale("symlog", linthresh=1.0)
axes[2].set_yscale("symlog", linthresh=1.0)
axes[-1].set_xlabel(r"$R$ (bohr)")
handles, labels = axes[0].get_legend_handles_labels()
figure.legend(
    handles,
    labels,
    loc="center left",
    bbox_to_anchor=(0.79, 0.5),
)
figure.text(
    0.79,
    0.39,
    "Dashed horizontal lines:\nisolated H",
    ha="left",
    va="top",
)
figure.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
figure.savefig(
    Path("h2plus_gto_polarizability.png"),
    dpi=200,
)
plt.show()
