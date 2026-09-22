# Repository Guidelines

## Project Structure & Module Organization

Augplot is a Python package using a `src/` layout. Product code lives in
`src/augplot/`; `core.py` owns the public notebook workflow, while profiling,
providers, execution, history, and exporting are separated into focused modules.
Tests live in `tests/` and generally mirror those responsibilities. Documentation is
under `docs/`, the runnable walkthrough is `examples/quickstart.ipynb`, and README
images belong in `docs/assets/`.

## Build, Test, and Development Commands

Create an isolated environment and install development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run the same core checks used by CI:

```bash
python -m ruff check .
MPLBACKEND=Agg python -m pytest
python -m build
```

The first command checks style and imports, the second runs the offline suite with a
headless plotting backend, and the third validates source and wheel packaging.

## Coding Style & Naming Conventions

Use four-space indentation, Python 3.11-compatible syntax, and a 100-character line
limit. Ruff enforces `E`, `F`, `I`, `B`, and `UP` rules. Use `snake_case` for functions,
variables, and modules; `PascalCase` for classes; and descriptive exception types in
`errors.py`. Keep public workflow methods small and move provider, persistence, or code
execution concerns into their existing modules.

## Testing Guidelines

Pytest discovers `tests/test_*.py` and `test_*` functions. Add focused regression tests
for every behavior change; there is no numeric coverage threshold. Tests must not depend
on real credentials or shared `.augplot` history. Live provider tests are explicitly
opt-in and may incur costs:

```bash
AUGPLOT_LIVE_TEST=1 python -m pytest -m live
```

## Commit & Pull Request Guidelines

Use short, imperative, sentence-case commit subjects, for example `Improve README quick
start` or `Add trusted PyPI release workflow`. `main` is protected: develop on a branch
and merge through a pull request; do not bypass protections with direct pushes. Keep PRs
focused, explain user-visible behavior, link relevant issues, and include screenshots for
notebook or plotting changes. Update tests and documentation when APIs, configuration,
persistence, or security boundaries change.

Every PR runs the required checks: Ruff, pytest, package build, and clean wheel import on
Python 3.11–3.14; a minimum-dependency test on Python 3.11; and CodeQL analysis. All
required checks must pass before merge.

## Security & Configuration

Never commit API keys, private datasets, generated environments, or notebook history
containing local paths. Generated Python is validated but is not OS-sandboxed; preserve
the execution guardrails and report vulnerabilities through `SECURITY.md`.
