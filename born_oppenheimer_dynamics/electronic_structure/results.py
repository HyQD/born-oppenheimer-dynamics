"""Backend-neutral electronic-structure results for diatomic molecules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np


_ARCHIVE_FORMAT = "born-oppenheimer-dynamics.diatomic-electronic-data"
_ARCHIVE_VERSION = 1
_FREQUENCY_MATCH_RTOL = 1e-12
_FREQUENCY_MATCH_ATOL = 1e-14


def _readonly_real_array(values, name, shape=None):
    array = np.asarray(values)
    if np.iscomplexobj(array) or array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values.")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, not {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _readonly_numeric_array(values, name, shape=None):
    array = np.asarray(values)
    if array.dtype.kind not in "biufc":
        raise ValueError(f"{name} must contain numeric values.")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, not {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    dtype = np.complex128 if np.iscomplexobj(array) else np.float64
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _freeze_json(value):
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value):
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _metadata(metadata):
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping.")
    try:
        encoded = json.dumps(
            _thaw_json(metadata),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "metadata must contain only JSON-compatible values."
        ) from error
    decoded = json.loads(encoded)
    return _freeze_json(decoded)


def _diagnostics(diagnostics):
    if not isinstance(diagnostics, Mapping):
        raise TypeError("diagnostics must be a mapping.")
    result = {}
    for name, values in diagnostics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Every diagnostic name must be a non-empty string.")
        array = np.asarray(values)
        if array.dtype.kind not in "biufc":
            raise ValueError(
                f"diagnostics[{name!r}] must contain numeric values."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"diagnostics[{name!r}] must contain only finite values."
            )
        values_copy = np.array(array, copy=True)
        values_copy.setflags(write=False)
        result[name] = values_copy
    return MappingProxyType(result)


@dataclass(frozen=True)
class DiatomicElectronicData:
    """Electronic energies and response tensors sampled for a diatomic.

    The polarizability tensor has shape (n_frequency, n_R, 3, 3).
    Backend-specific numerical information is stored as named, numeric arrays
    in diagnostics. Human-readable provenance and calculation settings are
    stored as JSON-compatible metadata.
    """

    internuclear_distance: np.ndarray
    frequencies: np.ndarray
    electronic_energy: np.ndarray
    polarizability: np.ndarray
    diagnostics: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        distance = _readonly_real_array(
            self.internuclear_distance, "internuclear_distance"
        )
        if distance.ndim != 1 or len(distance) == 0:
            raise ValueError(
                "internuclear_distance must be a non-empty one-dimensional array."
            )
        if np.any(distance <= 0) or np.any(np.diff(distance) <= 0):
            raise ValueError(
                "internuclear_distance must be positive and strictly increasing."
            )

        frequencies = _readonly_real_array(self.frequencies, "frequencies")
        if frequencies.ndim != 1 or len(frequencies) == 0:
            raise ValueError("frequencies must be a non-empty one-dimensional array.")
        if np.any(frequencies < 0) or np.any(np.diff(frequencies) <= 0):
            raise ValueError(
                "frequencies must be non-negative and strictly increasing."
            )

        energy = _readonly_real_array(
            self.electronic_energy,
            "electronic_energy",
            shape=(len(distance),),
        )
        polarizability = _readonly_numeric_array(
            self.polarizability,
            "polarizability",
            shape=(len(frequencies), len(distance), 3, 3),
        )

        object.__setattr__(self, "internuclear_distance", distance)
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "electronic_energy", energy)
        object.__setattr__(self, "polarizability", polarizability)
        object.__setattr__(self, "diagnostics", _diagnostics(self.diagnostics))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def polarizability_parallel(self) -> np.ndarray:
        """Return the body-fixed parallel polarizability."""
        return self.polarizability[..., 2, 2]

    @property
    def polarizability_perpendicular(self) -> np.ndarray:
        """Return the average of the two body-fixed perpendicular components."""
        return 0.5 * (
            self.polarizability[..., 0, 0]
            + self.polarizability[..., 1, 1]
        )

    def to_diatomic_surfaces(self, *, frequency=None):
        """Select one response frequency and return rotor-ready surfaces.

        If the result stores one frequency, omitting ``frequency`` selects it.
        Results with multiple frequencies require an explicit selection. A
        requested value is matched with relative tolerance ``1e-12`` and
        absolute tolerance ``1e-14`` atomic units.

        Complex response tensors cannot be represented by the current real
        rotor Hamiltonians and are rejected rather than silently truncated.
        """
        from ..vibrational_rotor.surfaces import DiatomicSurfaces

        if frequency is None:
            if len(self.frequencies) != 1:
                raise ValueError(
                    "frequency is required when electronic data contain "
                    "multiple response frequencies."
                )
            frequency_index = 0
        else:
            if isinstance(frequency, (bool, np.bool_)):
                raise TypeError("frequency must be a finite non-negative scalar.")
            frequency_array = np.asarray(frequency)
            if frequency_array.ndim != 0 or np.iscomplexobj(frequency_array):
                raise TypeError("frequency must be a finite non-negative scalar.")
            try:
                frequency_value = float(frequency_array)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "frequency must be a finite non-negative scalar."
                ) from error
            if not np.isfinite(frequency_value) or frequency_value < 0:
                raise ValueError(
                    "frequency must be a finite non-negative scalar."
                )
            matches = np.flatnonzero(
                np.isclose(
                    self.frequencies,
                    frequency_value,
                    rtol=_FREQUENCY_MATCH_RTOL,
                    atol=_FREQUENCY_MATCH_ATOL,
                )
            )
            if len(matches) == 0:
                available = ", ".join(f"{value:.12g}" for value in self.frequencies)
                raise ValueError(
                    f"frequency={frequency_value:.12g} was not calculated; "
                    f"available frequencies are [{available}] atomic units."
                )
            if len(matches) > 1:
                raise ValueError(
                    f"frequency={frequency_value:.12g} matches multiple stored "
                    "frequencies within the selection tolerance."
                )
            frequency_index = int(matches[0])

        selected_frequency = float(self.frequencies[frequency_index])
        selected_tensor = self.polarizability[frequency_index]
        if np.iscomplexobj(selected_tensor):
            raise ValueError(
                "DiatomicSurfaces supports only real response curves; the "
                f"selected polarizability at frequency={selected_frequency:.12g} "
                "atomic units is complex."
            )

        system = str(self.metadata.get("system", "diatomic"))
        backend = self.metadata.get("backend")
        basis_metadata = self.metadata.get("basis")
        if isinstance(basis_metadata, Mapping):
            basis = basis_metadata.get("name")
        else:
            basis = basis_metadata
        if not basis:
            settings = self.metadata.get("settings")
            if isinstance(settings, Mapping):
                basis = settings.get("basis")
        label_parts = [system]
        if basis:
            label_parts.append(str(basis))
        label_parts.append(f"omega={selected_frequency:.12g} a.u.")
        if backend:
            label_parts.append(str(backend))

        surface_metadata = {
            "selected_frequency_au": selected_frequency,
            "electronic_structure": self.metadata,
        }
        return DiatomicSurfaces(
            self.internuclear_distance,
            self.electronic_energy,
            polarizability_parallel=selected_tensor[:, 2, 2],
            polarizability_perpendicular=0.5
            * (selected_tensor[:, 0, 0] + selected_tensor[:, 1, 1]),
            dipole=None,
            label=", ".join(label_parts),
            metadata=surface_metadata,
        )

    def save(self, path):
        """Save the result to a compressed, non-pickled NPZ archive."""
        names = list(self.diagnostics)
        payload = {
            "archive_format": np.asarray(_ARCHIVE_FORMAT),
            "archive_version": np.asarray(_ARCHIVE_VERSION),
            "internuclear_distance": self.internuclear_distance,
            "frequencies": self.frequencies,
            "electronic_energy": self.electronic_energy,
            "polarizability": self.polarizability,
            "metadata_json": np.asarray(
                json.dumps(
                    _thaw_json(self.metadata),
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "diagnostic_names_json": np.asarray(json.dumps(names)),
        }
        payload.update(
            {
                f"diagnostic_{index}": self.diagnostics[name]
                for index, name in enumerate(names)
            }
        )
        np.savez_compressed(Path(path), **payload)

    @classmethod
    def from_npz(cls, path):
        """Load a result written by save without enabling pickle."""
        with np.load(Path(path), allow_pickle=False) as archive:
            required = {
                "archive_format",
                "archive_version",
                "internuclear_distance",
                "frequencies",
                "electronic_energy",
                "polarizability",
                "metadata_json",
                "diagnostic_names_json",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(
                    "Electronic-data archive is missing: "
                    + ", ".join(sorted(missing))
                )
            archive_format = str(np.asarray(archive["archive_format"]).item())
            version = int(np.asarray(archive["archive_version"]).item())
            if archive_format != _ARCHIVE_FORMAT:
                raise ValueError(
                    f"Unsupported electronic-data archive format {archive_format!r}."
                )
            if version != _ARCHIVE_VERSION:
                raise ValueError(
                    f"Unsupported electronic-data archive version {version}."
                )

            try:
                metadata = json.loads(
                    str(np.asarray(archive["metadata_json"]).item())
                )
                names = json.loads(
                    str(np.asarray(archive["diagnostic_names_json"]).item())
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    "Electronic-data archive contains invalid JSON metadata."
                ) from error
            if (
                not isinstance(names, list)
                or not all(isinstance(name, str) and name for name in names)
                or len(names) != len(set(names))
            ):
                raise ValueError(
                    "Electronic-data archive has invalid diagnostic names."
                )

            diagnostic_keys = {
                f"diagnostic_{index}" for index in range(len(names))
            }
            missing_diagnostics = diagnostic_keys.difference(archive.files)
            if missing_diagnostics:
                raise ValueError(
                    "Electronic-data archive is missing: "
                    + ", ".join(sorted(missing_diagnostics))
                )
            diagnostics = {
                name: np.array(archive[f"diagnostic_{index}"], copy=True)
                for index, name in enumerate(names)
            }
            return cls(
                internuclear_distance=np.array(
                    archive["internuclear_distance"], copy=True
                ),
                frequencies=np.array(archive["frequencies"], copy=True),
                electronic_energy=np.array(
                    archive["electronic_energy"], copy=True
                ),
                polarizability=np.array(archive["polarizability"], copy=True),
                diagnostics=diagnostics,
                metadata=metadata,
            )
