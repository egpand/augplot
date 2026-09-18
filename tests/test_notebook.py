import base64
import copy
import json
import struct
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
from notebook_charts import CV_BARS, CV_HIGHLIGHT, ORDERS_DOTS, ORDERS_FORECAST, ORDERS_PLOTLY


def test_example_in_real_kernel_with_mocked_inference(tmp_path):
    """Execute the actual example offline, including export and interactive output."""
    notebook = nbformat.read(
        Path(__file__).parents[1] / "examples" / "quickstart.ipynb", as_version=4
    )
    refined = CV_HIGHLIGHT
    setup = f"""
import os, json
def offline_key(prompt):
    assert prompt == "OpenAI API key: "
    return "offline-test-key"
get_ipython().kernel.getpass = offline_key
get_ipython().run_line_magic("matplotlib", "inline")
from matplotlib_inline.config import InlineBackend
InlineBackend.instance().figure_formats = {{"svg"}}
from augplot import provider
responses = iter({[CV_BARS, refined, ORDERS_DOTS, ORDERS_FORECAST, ORDERS_PLOTLY]!r})
offline_calls = 0
def offline_complete(**kwargs):
    global offline_calls
    offline_calls += 1
    return json.dumps({{"code": next(responses), "explanation": "Offline test fixture."}})
provider.complete = offline_complete
"""
    notebook.cells.insert(0, nbformat.v4.new_code_cell(setup))
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace("RUN_PLOTLY = False", "RUN_PLOTLY = True")
            if cell.id == "augplot-02":
                cell.source += (
                    '\nassert os.environ["AUGPLOT_MODEL"] == "openai/gpt-5.6-terra"\n'
                    'assert os.environ["OPENAI_API_KEY"] == "offline-test-key"'
                )
            if cell.id in {"augplot-07", "augplot-10"}:
                # Winners must be computed from the full data, including on local render.
                accuracy_winner = 2 if cell.id == "augplot-07" else 4
                cell.source += (
                    "\nassert [i for i, bar in enumerate(viz.figure.axes[0].patches) "
                    f"if bar.get_hatch()] == [{accuracy_winner}, 6]"
                )
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
    for expected_calls in (5, 0):
        manager = KernelManager(
            kernel_name="augplot-test",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]),
            connection_file=str(tmp_path / "connection.json"),
        )
        replay = copy.deepcopy(notebook)
        replay.cells.append(
            nbformat.v4.new_code_cell(
                f"assert offline_calls == {expected_calls}\n"
                f"assert viz.code == {refined!r}\n"
                f"assert orders_viz.cache_hit is {expected_calls == 0}\n"
                f"assert orders_viz.code == {ORDERS_FORECAST!r}\n"
                "forecast = next(line for line in orders_viz.figure.axes[0].lines "
                "if line.get_label() == 'Supplied forecast')\n"
                "np.testing.assert_array_equal(forecast.get_xdata(), forecast_results.week)\n"
                "np.testing.assert_allclose(forecast.get_ydata(), forecast_results.forecast)\n"
                "assert len(orders_viz.figure.axes[0].collections[0].get_offsets()) == 30\n"
                "outside = next(c for c in orders_viz.figure.axes[0].collections "
                "if c.get_label() == 'Outside supplied interval')\n"
                "np.testing.assert_allclose(outside.get_offsets()[:, 1], [620, 810])\n"
                "band = next(c for c in orders_viz.figure.axes[0].collections "
                "if c.get_label() == 'Supplied interval (synthetic)')\n"
                "np.testing.assert_allclose(np.unique(band.get_paths()[0].vertices[:, 1]), "
                "np.unique(forecast_window[['lower', 'upper']].to_numpy()))\n"
                "changed = forecast_results.copy()\n"
                "changed['forecast'] = changed['forecast'] + 13\n"
                "orders_viz.render(changed, show=False)\n"
                "shifted = next(line for line in orders_viz.figure.axes[0].lines "
                "if line.get_label() == 'Supplied forecast')\n"
                "np.testing.assert_allclose(shifted.get_ydata(), changed.forecast)\n"
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
        if cell.id in {
            "augplot-04", "augplot-07", "augplot-10", "augplot-14", "augplot-forecast"
        }:
            images = [output for output in cell.outputs if "image/png" in output.get("data", {})]
            assert len(images) == 1, f"Expected exactly one inline chart: {cell.source}"
            output = images[0]
            png = base64.b64decode(output.data["image/png"])
            width, height = struct.unpack(">II", png[16:24])
            dimensions = output.metadata["image/png"]
            assert width // 2 == dimensions["width"]
            assert height // 2 == dimensions["height"]
        if "RUN_PLOTLY = True" in cell.source:
            figures = [
                output
                for output in cell.outputs
                if "application/vnd.plotly.v1+json" in output.get("data", {})
            ]
            assert len(figures) == 1
    assert (tmp_path / "vis_utils.py").exists()
