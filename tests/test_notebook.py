import base64
import copy
import json
import struct
import sys
from pathlib import Path

import nbformat
import pandas as pd
import pytest
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
from notebook_charts import (
    FLIGHTS_HEATMAP,
    FLIGHTS_LINES,
    PENGUIN_FACETS,
    PENGUIN_SCATTER,
)


def test_hr_notebook_preserves_case_in_white_dwarf_labels():
    notebook = nbformat.read(
        Path(__file__).parents[1] / "examples" / "hertzsprung-russell-one-shot.ipynb",
        as_version=4,
    )
    data_cell = next(cell for cell in notebook.cells if cell.id == "hr-data")
    catalog = pd.DataFrame(
        {
            "dist": [10.0] * 3,
            "mag": [5.0] * 3,
            "absmag": [12.0, 12.0, 4.83],
            "ci": [1.5, 0.2, 0.6],
            "spect": ["dM4", "DA6", "G2V"],
        }
    )
    namespace = {"catalog": catalog, "pd": pd}

    exec(data_cell.source, namespace)

    stars = namespace["stars"]
    assert stars["stellar_class"].tolist() == [
        "Unclassified",
        "White dwarfs",
        "Main sequence",
    ]
    assert stars.loc[2, "relative_V_band_luminosity_solar"] == 1.0


def test_example_in_real_kernel_with_mocked_inference(tmp_path):
    """Execute the actual example offline, including its generated Python output."""
    notebook = nbformat.read(
        Path(__file__).parents[1] / "examples" / "quickstart.ipynb", as_version=4
    )
    refined = PENGUIN_FACETS
    setup = f"""
import os, json
import numpy as np
import pandas as pd
get_ipython().run_line_magic("matplotlib", "inline")
from matplotlib_inline.config import InlineBackend
InlineBackend.instance().figure_formats = {{"svg"}}
from augplot import provider
from augplot import datasets
import augplot as ap
from augplot import auth
auth._directory = lambda: __import__("pathlib").Path("chatgpt-test-config")
offline_signins = 0
def offline_sign_in(*args, **kwargs):
    global offline_signins
    offline_signins += 1
    state = {{
        "version": 1, "host_id": "urn:uuid:offline", "active": "offline",
        "profiles": {{"offline": {{
            "label": "Offline test", "model": "offline-model",
            "connected": True, "plan_enabled": True,
        }}}},
    }}
    auth._write_state(state)
    return ap.accounts()[0]
ap.sign_in = offline_sign_in
penguin_rows = []
for species, island, base_length, base_depth in [
    ("Adelie", "Biscoe", 38, 18),
    ("Gentoo", "Biscoe", 48, 15),
    ("Chinstrap", "Dream", 47, 18),
]:
    for offset, sex in enumerate(["Female", "Male", "Female", "Male"]):
        penguin_rows.append({{
            "species": species, "island": island, "bill_length_mm": base_length + offset,
            "bill_depth_mm": base_depth + offset / 4, "flipper_length_mm": 190 + offset,
            "body_mass_g": 3500 + offset * 100, "sex": sex,
        }})
penguin_fixture = pd.DataFrame(penguin_rows)
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
flight_fixture = pd.DataFrame([
    {{"year": year, "month": month, "passengers": 100 + (year - 1949) * 20 + i * 5}}
    for year in range(1949, 1961) for i, month in enumerate(months)
])
def offline_dataset(name, **kwargs):
    if name == "penguins":
        return penguin_fixture.copy()
    if name == "flights":
        return flight_fixture.copy()
    raise AssertionError(name)
datasets.sns.load_dataset = offline_dataset
responses = iter({[PENGUIN_SCATTER, refined, FLIGHTS_LINES, FLIGHTS_HEATMAP]!r})
offline_calls = 0
def offline_complete(**kwargs):
    global offline_calls
    offline_calls += 1
    return json.dumps({{
        "status": "ok", "code": next(responses), "explanation": "Offline test fixture."
    }})
provider.complete = offline_complete
"""
    notebook.cells.insert(0, nbformat.v4.new_code_cell(setup))
    for cell in notebook.cells:
        if cell.cell_type == "code":
            if cell.id == "augplot-02":
                cell.source += (
                    '\nassert ap.accounts()[0]["model"] == "offline-model"\n'
                    'assert ap.accounts()[0]["connected"]'
                )
            if cell.id == "augplot-10":
                cell.source += "\nassert set(biscoe_penguins['island']) == {'Biscoe'}"
    notebook.cells.append(
        nbformat.v4.new_code_cell("assert InlineBackend.instance().figure_formats == {'svg'}")
    )

    # Use this test environment's Python, never a user's default notebook kernel.
    kernel_root = tmp_path / "kernels"
    kernel_path = kernel_root / "augplot-test"
    kernel_path.mkdir(parents=True)
    (kernel_path / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Augplot tests",
                "language": "python",
                "env": {
                    "IPYTHONDIR": str(tmp_path / "ipython"),
                    "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
                },
            }
        )
    )
    # The second pass starts a fresh kernel and must replay all steps from disk.
    for expected_calls in (4, 0):
        manager = KernelManager(
            kernel_name="augplot-test",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]),
            connection_file=str(tmp_path / "connection.json"),
        )
        replay = copy.deepcopy(notebook)
        replay.cells.append(
            nbformat.v4.new_code_cell(
                f"assert offline_calls == {expected_calls}\n"
                f"assert offline_signins == {int(expected_calls > 0)}\n"
                f"assert plt.code == {refined!r}\n"
                f"assert flight_viz.cache_hit is {expected_calls == 0}\n"
                f"assert flight_viz.code == {FLIGHTS_HEATMAP!r}\n"
                "assert flights.shape == (144, 3)\n"
                "assert len(flight_viz.figure.axes[0].texts) == 144\n"
                "assert len(flight_viz.figure.axes[0].patches) == 1\n"
                "assert len(plt.figure.axes) == 2\n"
                "assert set(biscoe_penguins['species']) == {'Adelie', 'Gentoo'}\n"
                "plt.render(biscoe_penguins, show=False)\n"
                "assert len(plt.figure.axes) == 2\n"
                f"assert offline_calls == {expected_calls}"
            )
        )
        try:
            executed = NotebookClient(
                replay, km=manager, timeout=90, resources={"metadata": {"path": str(tmp_path)}}
            ).execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            manager.cleanup_resources()

    for cell in executed.cells:
        if cell.cell_type != "code":
            continue
        assert not any(output.output_type == "error" for output in cell.outputs)
        if cell.id in {"augplot-04", "augplot-07", "augplot-10", "augplot-14", "augplot-forecast"}:
            images = [output for output in cell.outputs if "image/png" in output.get("data", {})]
            assert len(images) == 1, f"Expected exactly one inline chart: {cell.source}"
            output = images[0]
            png = base64.b64decode(output.data["image/png"])
            width, height = struct.unpack(">II", png[16:24])
            dimensions = output.metadata["image/png"]
            assert width // 2 == dimensions["width"]
            assert height // 2 == dimensions["height"]
    exported = tmp_path / "augplot_utils.py"
    assert exported.exists()
    assert "def plot_penguin_bills" in exported.read_text()


@pytest.mark.parametrize(
    "filename",
    [
        "cross-validation-model-comparison.ipynb",
        "hertzsprung-russell-one-shot.ipynb",
    ],
)
def test_other_examples_execute_offline_with_chatgpt_setup(tmp_path, filename):
    """Run each complete notebook, including authentication setup and exported code."""
    notebook = nbformat.read(
        Path(__file__).parents[1] / "examples" / filename,
        as_version=4,
    )
    code = """def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize or (8, 4))
    ax.set_title(title or "Offline notebook check")
    return fig
"""
    setup = f"""
import json
from pathlib import Path
import pandas as pd
import augplot as ap
from augplot import auth, provider
get_ipython().run_line_magic("matplotlib", "inline")
auth._directory = lambda: Path("chatgpt-test-config")
signin_calls = 0
def offline_signin(*args, **kwargs):
    global signin_calls
    signin_calls += 1
    auth._write_state({{
        "version": 1, "host_id": "urn:uuid:offline", "active": "offline",
        "profiles": {{"offline": {{
            "label": "Offline test", "model": "offline-model",
            "connected": True, "plan_enabled": True,
        }}}},
    }})
    return ap.accounts()[0]
ap.sign_in = offline_signin
inference_calls = 0
def offline_complete(**kwargs):
    global inference_calls
    inference_calls += 1
    assert kwargs["auth"] == "chatgpt"
    assert kwargs["model"] == "offline-model"
    assert kwargs["profile_id"] == "offline"
    if "apple pie" in kwargs["messages"][1]["content"]:
        return json.dumps({{
            "status": "out_of_scope", "code": "", "explanation": "Plotting requests only."
        }})
    return json.dumps({{
        "status": "ok", "code": {code!r}, "explanation": "Offline notebook fixture."
    }})
provider.complete = offline_complete
real_read_csv = pd.read_csv
def offline_read_csv(source, **kwargs):
    if isinstance(source, str) and source.startswith("https://raw.githubusercontent.com/"):
        return pd.DataFrame({{
            "id": [1, 2, 3], "dist": [10.0, 10.0, 10.0], "mag": [5.0, 4.0, 2.0],
            "absmag": [4.83, 10.0, -2.0], "ci": [0.6, 0.2, 1.5], "spect": ["G2V", "DA6", "K2III"],
        }})
    raise AssertionError("Unexpected dataset download")
pd.read_csv = offline_read_csv
"""
    notebook.cells.insert(0, nbformat.v4.new_code_cell(setup))
    notebook.cells.append(
        nbformat.v4.new_code_cell("assert signin_calls == 1\nassert inference_calls > 0")
    )
    kernel_root = tmp_path / "kernels"
    kernel_path = kernel_root / "augplot-test"
    kernel_path.mkdir(parents=True)
    (kernel_path / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Augplot tests",
                "language": "python",
                "env": {
                    "IPYTHONDIR": str(tmp_path / "ipython"),
                    "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
                },
            }
        )
    )
    manager = KernelManager(
        kernel_name="augplot-test",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]),
        connection_file=str(tmp_path / "connection.json"),
    )
    try:
        executed = NotebookClient(
            notebook,
            km=manager,
            timeout=90,
            resources={"metadata": {"path": str(tmp_path)}},
        ).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    assert not any(
        output.output_type == "error"
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )
    chart_name = "chart" if filename.startswith("hertzsprung") else "vis"
    assert any(
        "image/png" in output.get("data", {})
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    ), f"Expected inline figure from {chart_name}"
