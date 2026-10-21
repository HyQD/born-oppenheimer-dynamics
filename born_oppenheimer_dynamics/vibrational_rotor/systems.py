"""Diatomic systems used by vibrational-rotor calculations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from numbers import Real

import numpy as np
from matter_field_params import isotope_mass, reduced_mass


_ISOTOPE_PATTERN = re.compile(
    r"^(?:(?P<mass_first>\d+)-?)?(?P<symbol>[A-Za-z]{1,3})"
    r"(?:-(?P<mass_last>\d+))?$"
)


def _element_symbol(isotope: str) -> str:
    if isotope in {"D", "T"}:
        return "H"
    match = _ISOTOPE_PATTERN.fullmatch(isotope.strip())
    if match is None or (match.group("mass_first") is None) == (
        match.group("mass_last") is None
    ):
        raise ValueError(
            f"Invalid isotope specification '{isotope}'. Use, for example, "
            "'H-1' or 'Li-7'."
        )
    return match.group("symbol").capitalize()


@dataclass(frozen=True)
class DiatomicMolecule:
    """Two explicitly specified isotopes and their nuclear masses.

    Parameters
    ----------
    isotopes
        Two isotope strings, such as ``("Li-7", "H-1")``.
    mass_kind
        Mass convention forwarded to :mod:`matter_field_params`. The default
        ``"nuclear"`` subtracts electron rest masses from neutral-atom
        isotope masses and neglects electronic binding energies.
    codata_set
        CODATA set used for conversion to electron-mass atomic units.
    reduced_mass_override
        Optional positive reduced mass in electron-mass atomic units.
    """

    isotopes: tuple[str, str]
    mass_kind: str = "nuclear"
    codata_set: str = "current"
    reduced_mass_override: float | None = None

    def __post_init__(self):
        isotopes = tuple(self.isotopes)
        if len(isotopes) != 2 or not all(isinstance(item, str) for item in isotopes):
            raise ValueError("isotopes must contain exactly two isotope strings.")
        if self.mass_kind not in {"atomic", "nuclear"}:
            raise ValueError("mass_kind must be 'atomic' or 'nuclear'.")
        for isotope in isotopes:
            isotope_mass(
                isotope,
                units="electron_mass",
                mass_kind=self.mass_kind,
                codata_set=self.codata_set,
            )
        object.__setattr__(self, "isotopes", isotopes)

        if self.reduced_mass_override is not None:
            value = self.reduced_mass_override
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError("reduced_mass_override must be a positive scalar.")
            if not np.isfinite(value) or value <= 0:
                raise ValueError("reduced_mass_override must be positive and finite.")
            object.__setattr__(self, "reduced_mass_override", float(value))

    @property
    def element_symbols(self) -> tuple[str, str]:
        """Element symbols corresponding to the two isotope specifications."""
        return tuple(_element_symbol(isotope) for isotope in self.isotopes)

    @property
    def homonuclear(self) -> bool:
        """Whether the nuclei have the same charge/electronic element."""
        first, second = self.element_symbols
        return first == second

    @property
    def masses(self) -> tuple[float, float]:
        """Individual masses in electron-mass atomic units."""
        return tuple(
            isotope_mass(
                isotope,
                units="electron_mass",
                mass_kind=self.mass_kind,
                codata_set=self.codata_set,
            )
            for isotope in self.isotopes
        )

    @property
    def reduced_mass(self) -> float:
        """Diatomic reduced mass in electron-mass atomic units."""
        if self.reduced_mass_override is not None:
            return self.reduced_mass_override
        return reduced_mass(
            self.isotopes,
            units="electron_mass",
            mass_kind=self.mass_kind,
            codata_set=self.codata_set,
        )

    @property
    def label(self) -> str:
        """Human-readable isotope label."""
        return "--".join(self.isotopes)
