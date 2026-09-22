import tomllib
from pathlib import Path

import augplot


def test_project_version_matches_package_version():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        project_version = tomllib.load(pyproject_file)["project"]["version"]

    assert project_version == augplot.__version__
