"""Execute the recorded HR example and its exported source entirely offline."""

import json
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


def test_hr_notebook_executes_recorded_source_in_real_kernel(tmp_path):
    notebook = nbformat.read(
        Path(__file__).parents[1] / "examples" / "hertzsprung-russell-one-shot.ipynb",
        as_version=4,
    )
    source_cell = next(cell for cell in notebook.cells if cell.id == "hr-code")
    sources = [
        output.get("data", {}).get("text/plain", "") for output in source_cell.outputs
        if output.get("data", {}).get("text/plain", "").startswith("def plot_data(")
    ]
    assert len(sources) == 1, "Record the evaluated generated function in the notebook Code output"
    source = sources[0]
    setup = f'''
import json
from pathlib import Path
import numpy as np
import pandas as pd
import augplot as ap
from augplot import auth, provider
get_ipython().run_line_magic("matplotlib", "inline")
auth._directory = lambda: Path("chatgpt-test-config")
auth._write_state({{
    "version": 1, "host_id": "urn:uuid:offline", "active": "offline",
    "profiles": {{"offline": {{
        "label": "Offline test", "model": "offline-model",
        "connected": True, "plan_enabled": True,
    }}}},
}})
def forbidden_signin(*args, **kwargs):
    raise AssertionError("The notebook must use the existing offline sign-in")
ap.sign_in = forbidden_signin
rows = []
for spectrum, intercept, slope in [
    ("DA6", 12, 2), ("G2V", 2, 3), ("G2IV", 1, 1.8),
    ("G2III", -1, 1.2), ("G2II", -3, 0.8), ("G2Ia", -6, 0.4),
]:
    for color in np.linspace(-0.1, 1.8, 40):
        rows.append({{
            "id": len(rows), "dist": 10.0, "mag": 5.0,
            "absmag": intercept + slope * color, "ci": color, "spect": spectrum,
        }})
rows.extend([
    {{"id": 240, "dist": 10., "mag": 5., "absmag": 12., "ci": 1.5, "spect": "dM4"}},
    {{"id": 241, "dist": 100000., "mag": 5., "absmag": 1., "ci": .5, "spect": "G2V"}},
    {{"id": 242, "dist": 10., "mag": 5., "absmag": 1., "ci": 3., "spect": "G2V"}},
])
fixture = pd.DataFrame(rows)
download_calls = 0
def offline_catalog(url, **kwargs):
    global download_calls
    download_calls += 1
    assert url.endswith("/hyg/CURRENT/hygdata_v41.csv")
    assert kwargs == {{"usecols": ["id", "dist", "mag", "absmag", "ci", "spect"]}}
    return fixture.copy(deep=True)
pd.read_csv = offline_catalog
inference_calls = 0
def offline_complete(**kwargs):
    global inference_calls
    inference_calls += 1
    assert inference_calls == 1
    assert kwargs["auth"] == "chatgpt"
    assert kwargs["model"] == "offline-model"
    assert kwargs["profile_id"] == "offline"
    return json.dumps({{
        "status": "ok", "code": {source!r}, "explanation": "Recorded notebook source."
    }})
provider.complete = offline_complete
'''
    notebook.cells.insert(0, nbformat.v4.new_code_cell(setup))
    notebook.cells.append(nbformat.v4.new_code_cell(f'''
assert download_calls == inference_calls == 1
assert not chart.cache_hit
assert chart.code == {source.strip() + chr(10)!r}
assert len(stars) == 241
assert stars["stellar_class"].value_counts()["Unclassified"] == 1
assert len(fig.axes[0].collections[0].get_offsets()) == len(stars)
expected = {{"White dwarfs", "Main sequence", "Subgiants", "Giants",
            "Bright giants", "Supergiants"}}
assert {{text.get_text() for text in fig.axes[0].texts}} == expected
assert len(fig.axes[0].lines) == 6
left, right = fig.axes[:2]
assert left.get_yscale() == "log"
assert right.yaxis_inverted()
np.testing.assert_allclose(
    right.get_ylim(), 4.83 - 2.5 * np.log10(left.get_ylim()), rtol=0, atol=1e-10,
)
export_source = Path("augplot_utils.py").read_text()
assert "def h_r_diagram" in export_source
assert "import augplot" not in export_source and "from augplot" not in export_source
assert inference_calls == 1
'''))
    kernel_root = tmp_path / "kernels"
    kernel_path = kernel_root / "augplot-hr-test"
    kernel_path.mkdir(parents=True)
    (kernel_path / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Augplot HR tests", "language": "python",
        "env": {
            "IPYTHONDIR": str(tmp_path / "ipython"),
            "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
            "MPLBACKEND": "Agg",
        },
    }))
    manager = KernelManager(
        kernel_name="augplot-hr-test",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]),
        connection_file=str(tmp_path / "connection.json"),
    )
    try:
        executed = NotebookClient(
            notebook, km=manager, timeout=90, resources={"metadata": {"path": str(tmp_path)}}
        ).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    for cell in executed.cells:
        if cell.cell_type == "code":
            assert not any(output.output_type == "error" for output in cell.outputs)
    plot_cell = next(cell for cell in executed.cells if cell.id == "hr-plot")
    assert any("image/png" in output.get("data", {}) for output in plot_cell.outputs)
    code_cell = next(cell for cell in executed.cells if cell.id == "hr-code")
    assert any(
        output.get("data", {}).get("text/plain") == source for output in code_cell.outputs
    )
    assert any(
        "Model calls: 1" in output.get("text", "") for output in code_cell.outputs
    )
    assert (tmp_path / "augplot_utils.py").exists()
