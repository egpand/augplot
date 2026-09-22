# Contributing to Augplot

Thanks for helping improve Augplot. Bug reports, focused feature proposals, documentation
improvements, and tested code changes are welcome.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before opening an issue

- Search existing issues to avoid duplicates.
- Use a minimal, reproducible example for bugs.
- Include the Augplot and Python versions, operating system, model provider, and complete
  traceback where relevant.
- Remove credentials, personal data, and proprietary prompts or datasets.
- Report vulnerabilities privately according to the [security policy](SECURITY.md).

## Development setup

Augplot supports Python 3.11 through 3.14. Create an isolated environment, then install the
project with its development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run the local checks before submitting a pull request:

```bash
python -m ruff check .
MPLBACKEND=Agg python -m pytest
python -m build
```

Live model tests are opt-in because they use provider credentials and may incur costs:

```bash
AUGPLOT_LIVE_TEST=1 python -m pytest -m live
```

## Pull requests

- Keep changes focused and explain the user-visible behavior.
- Add or update tests for behavior changes.
- Update documentation when public APIs, configuration, or security boundaries change.
- Do not commit generated environments, API credentials, notebook history containing local
  paths, or private data.
- Confirm that tests, linting, and package builds pass.

Maintainers may ask for changes to keep the API small, preserve notebook usability, or
maintain the generated-code security boundary.
