# Installation

`born-oppenheimer-dynamics` supports Python 3.12 through 3.14. Install it in a
dedicated virtual environment so that its numerical dependencies do not alter
an unrelated Python installation.

## Create an environment

Using the standard-library environment tool:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

An existing Conda environment may be used instead. In either case, verify that
`python` and pip refer to the same environment:

```bash
command -v python
python -m pip --version
```

Always invoke pip as `python -m pip`; this prevents a `pip` executable from a
different environment from receiving the installation.

## Install directly from GitHub

The package depends on four other HyQD repositories. Install those first and
then install `born-oppenheimer-dynamics`:

```bash
python -m pip install \
    "git+https://github.com/HyQD/electromagnetic-fields.git" \
    "git+https://github.com/HyQD/grid-lib.git" \
    "git+https://github.com/HyQD/integrator-lib.git" \
    "git+https://github.com/HyQD/matter_field_params.git"

python -m pip install \
    "git+https://github.com/HyQD/born-oppenheimer-dynamics.git"
```

The general form is:

```bash
python -m pip install "git+https://github.com/<owner>/<repository>.git"
```

To make an installation reproducible, append a Git tag or full commit hash:

```bash
python -m pip install \
    "git+https://github.com/HyQD/born-oppenheimer-dynamics.git@<tag-or-commit>"
```

Replace `<tag-or-commit>` with a real revision; do not type the angle brackets.
Pin the four HyQD dependencies as well when the complete environment must be
reproducible.

## Install from local checkouts

For development across the HyQD repositories, place the checkouts below a
common directory. For example:

```text
/path/to/HyQD/
├── born-oppenheimer-dynamics/
├── electromagnetic-fields/
├── grid-lib/
├── integrator-lib/
└── matter_field_params/
```

Install the dependencies before their consumer:

```bash
hyqd_root=/path/to/HyQD

python -m pip install --editable "${hyqd_root}/electromagnetic-fields"
python -m pip install --editable "${hyqd_root}/grid-lib"
python -m pip install --editable "${hyqd_root}/integrator-lib"
python -m pip install --editable "${hyqd_root}/matter_field_params"
python -m pip install --editable \
    "${hyqd_root}/born-oppenheimer-dynamics[dev]"
```

An editable installation uses the source checkout directly, so subsequent
Python processes see code changes without rebuilding a wheel. Re-run the
editable-install command when dependency declarations or other packaging
metadata change.

For a non-editable installation from an existing checkout, run this in the
repository root:

```bash
python -m pip install .
```

## Verify the installation

Check dependency metadata and the imported package location:

```bash
python -m pip check

python -c 'import born_oppenheimer_dynamics as bod; print(bod.__version__); print(bod.__file__)'
```

For a development installation, run the test suite from the repository root:

```bash
python -m pytest -q
```

The current dependency baseline includes grid-lib 1.2.0, integrator-lib 0.3.0,
NumPy 2.5.2, SciPy 1.18.0, opt-einsum 3.4, Numba 0.67, and SymPy 1.14. The
bounds in `pyproject.toml` are the authoritative installation requirements.
