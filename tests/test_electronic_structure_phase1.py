import subprocess
import sys

import numpy as np
import pytest

from born_oppenheimer_dynamics import (
    DiatomicElectronicData,
    radial_grid_from_range,
)


def sample_electronic_data():
    distance = np.array([0.5, 1.0, 1.5])
    frequencies = np.array([0.0, 0.057])
    energy = np.array([-0.2, -0.6, -0.5])
    polarizability = np.zeros((2, 3, 3, 3))
    polarizability[..., 0, 0] = 2.0
    polarizability[..., 1, 1] = 4.0
    polarizability[..., 2, 2] = 8.0
    return DiatomicElectronicData(
        internuclear_distance=distance,
        frequencies=frequencies,
        electronic_energy=energy,
        polarizability=polarizability,
        diagnostics={
            "minimum_overlap_eigenvalue": np.array([0.1, 0.2, 0.3]),
            "discarded_overlap_modes": np.array([1, 0, 0]),
        },
        metadata={
            "backend": "test",
            "settings": {"basis": "test-basis", "thresholds": [1e-10, 1e-8]},
        },
    )


def test_radial_grid_from_range_is_inclusive_and_uniform():
    grid = radial_grid_from_range((0.1, 0.4, 0.1))

    np.testing.assert_allclose(grid, [0.1, 0.2, 0.3, 0.4])
    assert grid[-1] == 0.4
    assert not grid.flags.writeable


def test_radial_grid_from_range_accepts_one_point():
    np.testing.assert_array_equal(
        radial_grid_from_range((2.0, 2.0, 0.25)),
        [2.0],
    )


@pytest.mark.parametrize(
    ("radial_range", "error", "message"),
    [
        ((0.0, 1.0, 0.1), ValueError, "r_min"),
        ((1.0, 0.5, 0.1), ValueError, "r_max"),
        ((0.1, 1.0, 0.0), ValueError, "dr"),
        ((0.1, 1.0, 0.25), ValueError, "integral"),
        ((0.1, 1.0), ValueError, "exactly"),
        ((True, 1.0, 0.1), TypeError, "r_min"),
    ],
)
def test_radial_grid_from_range_rejects_invalid_input(
    radial_range, error, message
):
    with pytest.raises(error, match=message):
        radial_grid_from_range(radial_range)


def test_electronic_data_validates_and_derives_components():
    data = sample_electronic_data()

    assert data.polarizability.shape == (2, 3, 3, 3)
    np.testing.assert_allclose(data.polarizability_parallel, 8.0)
    np.testing.assert_allclose(data.polarizability_perpendicular, 3.0)
    assert data.metadata["settings"]["thresholds"] == (1e-10, 1e-8)
    assert data.diagnostics["discarded_overlap_modes"].dtype.kind in "iu"

    with pytest.raises(ValueError, match="read-only"):
        data.internuclear_distance[0] = 1.0
    with pytest.raises(TypeError):
        data.metadata["backend"] = "changed"
    with pytest.raises(ValueError, match="read-only"):
        data.diagnostics["discarded_overlap_modes"][0] = 2


def test_electronic_data_copies_input_arrays():
    distance = np.array([1.0, 2.0])
    polarizability = np.zeros((1, 2, 3, 3))
    data = DiatomicElectronicData(
        internuclear_distance=distance,
        frequencies=[0.0],
        electronic_energy=[-0.5, -0.4],
        polarizability=polarizability,
    )

    distance[0] = 9.0
    polarizability[0, 0, 2, 2] = 9.0
    assert data.internuclear_distance[0] == 1.0
    assert data.polarizability[0, 0, 2, 2] == 0.0


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"internuclear_distance": []}, "non-empty"),
        ({"internuclear_distance": [1.0, 1.0, 2.0]}, "strictly increasing"),
        ({"internuclear_distance": [-1.0, 1.0, 2.0]}, "positive"),
        ({"frequencies": []}, "non-empty"),
        ({"frequencies": [-0.1, 0.0]}, "non-negative"),
        ({"frequencies": [0.0, 0.0]}, "strictly increasing"),
        ({"electronic_energy": [-0.5]}, "shape"),
        ({"polarizability": np.zeros((1, 3, 3, 3))}, "shape"),
    ],
)
def test_electronic_data_rejects_invalid_core_arrays(options, message):
    arguments = {
        "internuclear_distance": [0.5, 1.0, 1.5],
        "frequencies": [0.0, 0.057],
        "electronic_energy": [-0.2, -0.6, -0.5],
        "polarizability": np.zeros((2, 3, 3, 3)),
    }
    arguments.update(options)
    with pytest.raises(ValueError, match=message):
        DiatomicElectronicData(**arguments)


def test_electronic_data_rejects_non_json_metadata_and_object_diagnostics():
    with pytest.raises(ValueError, match="JSON-compatible"):
        DiatomicElectronicData(
            internuclear_distance=[1.0],
            frequencies=[0.0],
            electronic_energy=[-0.5],
            polarizability=np.zeros((1, 1, 3, 3)),
            metadata={"invalid": object()},
        )

    with pytest.raises(ValueError, match="numeric"):
        DiatomicElectronicData(
            internuclear_distance=[1.0],
            frequencies=[0.0],
            electronic_energy=[-0.5],
            polarizability=np.zeros((1, 1, 3, 3)),
            diagnostics={"invalid": np.array([object()], dtype=object)},
        )


def test_electronic_data_npz_round_trip_without_pickle(tmp_path):
    data = sample_electronic_data()
    path = tmp_path / "electronic_data.npz"

    data.save(path)
    loaded = DiatomicElectronicData.from_npz(path)

    np.testing.assert_array_equal(
        loaded.internuclear_distance, data.internuclear_distance
    )
    np.testing.assert_array_equal(loaded.frequencies, data.frequencies)
    np.testing.assert_array_equal(
        loaded.electronic_energy, data.electronic_energy
    )
    np.testing.assert_array_equal(loaded.polarizability, data.polarizability)
    assert tuple(loaded.diagnostics) == tuple(data.diagnostics)
    for name in data.diagnostics:
        np.testing.assert_array_equal(
            loaded.diagnostics[name], data.diagnostics[name]
        )
    assert loaded.metadata == data.metadata

    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)


def test_single_frequency_converts_to_surfaces_automatically():
    data = sample_electronic_data()
    single_frequency = DiatomicElectronicData(
        internuclear_distance=data.internuclear_distance,
        frequencies=[data.frequencies[1]],
        electronic_energy=data.electronic_energy,
        polarizability=data.polarizability[1:2],
        metadata=data.metadata,
    )

    surfaces = single_frequency.to_diatomic_surfaces()

    np.testing.assert_array_equal(
        surfaces.internuclear_distance,
        single_frequency.internuclear_distance,
    )
    np.testing.assert_array_equal(
        surfaces.electronic_energy,
        single_frequency.electronic_energy,
    )
    np.testing.assert_allclose(surfaces.polarizability_parallel, 8.0)
    np.testing.assert_allclose(surfaces.polarizability_perpendicular, 3.0)
    assert surfaces.dipole is None
    assert "omega=0.057 a.u." in surfaces.label
    assert "test-basis" in surfaces.label
    assert surfaces.metadata["selected_frequency_au"] == 0.057
    assert surfaces.metadata["electronic_structure"] is single_frequency.metadata


def test_multi_frequency_conversion_requires_explicit_selection():
    data = sample_electronic_data()

    with pytest.raises(ValueError, match="frequency is required"):
        data.to_diatomic_surfaces()

    surfaces = data.to_diatomic_surfaces(frequency=0.057 + 1e-15)
    np.testing.assert_allclose(surfaces.polarizability_parallel, 8.0)
    assert surfaces.metadata["selected_frequency_au"] == 0.057


def test_surface_conversion_rejects_unavailable_or_invalid_frequency():
    data = sample_electronic_data()

    with pytest.raises(ValueError, match="was not calculated"):
        data.to_diatomic_surfaces(frequency=0.1)
    with pytest.raises(ValueError, match="non-negative"):
        data.to_diatomic_surfaces(frequency=-0.1)
    with pytest.raises(TypeError, match="scalar"):
        data.to_diatomic_surfaces(frequency=[0.057])


def test_surface_conversion_rejects_complex_response():
    polarizability = np.zeros((1, 2, 3, 3), dtype=complex)
    polarizability[..., 2, 2] = 1.0 + 0.1j
    data = DiatomicElectronicData(
        internuclear_distance=[1.0, 2.0],
        frequencies=[0.057],
        electronic_energy=[-0.5, -0.4],
        polarizability=polarizability,
    )

    with pytest.raises(ValueError, match="complex"):
        data.to_diatomic_surfaces()


def test_base_import_does_not_load_optional_electronic_dependencies():
    script = """
import builtins

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in {"pyscf", "basis_set_exchange"}:
        raise AssertionError(f"unexpected optional import: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import born_oppenheimer_dynamics
"""
    subprocess.run([sys.executable, "-c", script], check=True)
