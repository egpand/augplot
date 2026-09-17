import json
import sys
from pathlib import Path

import nbformat
from conftest import CV_CODE, DF_CODE, PLOTLY_CODE
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


def test_example_in_real_kernel_with_mocked_inference(tmp_path):
    """Execute the actual example offline, including export and interactive output."""
    notebook = nbformat.read(
        Path(__file__).parents[1] / "examples" / "quickstart.ipynb", as_version=4
    )
    refined = CV_CODE.replace("ax.bar(names, means)", "ax.barh(names, means)")
    setup = f"""
import os, json
os.environ["HIMALIA_MODEL"] = "test/offline"
get_ipython().run_line_magic("matplotlib", "inline")
from himalia import provider
responses = iter({[CV_CODE, refined, DF_CODE, PLOTLY_CODE]!r})
def offline_complete(**kwargs):
    return json.dumps({{"code": next(responses), "explanation": "Offline test fixture."}})
provider.complete = offline_complete
"""
    notebook.cells.insert(0, nbformat.v4.new_code_cell(setup))
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace("RUN_PLOTLY = False", "RUN_PLOTLY = True")

    # Use this test environment's Python, never a user's default notebook kernel.
    kernel_root = tmp_path / "kernels"
    kernel_path = kernel_root / "himalia-test"
    kernel_path.mkdir(parents=True)
    (kernel_path / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Himalia tests",
                "language": "python",
                "env": {
                    "IPYTHONDIR": str(tmp_path / "ipython"),
                    "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
                },
            }
        )
    )
    manager = KernelManager(
        kernel_name="himalia-test",
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
        if cell.cell_type != "code":
            continue
        assert not any(output.output_type == "error" for output in cell.outputs)
        if any(
            marker in cell.source
            for marker in (
                "viz = plot(results_dict",
                "viz.refine(",
                "viz.render(updated_results",
                "campaign_viz.fit(",
            )
        ):
            images = [output for output in cell.outputs if "image/png" in output.get("data", {})]
            assert len(images) == 1, f"Expected exactly one inline chart: {cell.source}"
        if "RUN_PLOTLY = True" in cell.source:
            figures = [
                output
                for output in cell.outputs
                if "application/vnd.plotly.v1+json" in output.get("data", {})
            ]
            assert len(figures) == 1
    assert (tmp_path / "vis_utils.py").exists()
