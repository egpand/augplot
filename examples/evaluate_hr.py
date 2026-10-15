"""Explicitly opt-in, repeated one-call evaluation of the notebook's HR request.

Example (uses the existing ChatGPT sign-in):
    python examples/evaluate_hr.py --runs 3 --output /tmp/hr-evaluation \\
        --catalog /tmp/hygdata_v41.csv

Each run uses the same frozen request. Inspect PNGs and source yourself; numerical
observations alone do not establish visual quality. No repair or prompt search runs.
"""

import argparse
import ast
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg

import augplot as ap
from augplot import provider
from augplot.execution import VALIDATOR_VERSION
from augplot.plotting import PLOTTING_VERSION
from augplot.prompts import PROMPT_VERSION

NOTEBOOK = Path(__file__).resolve().with_name("hertzsprung-russell-one-shot.ipynb")
COLUMNS = ["id", "dist", "mag", "absmag", "ci", "spect"]


def _assignment(cells, name):
    for source in cells:
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets
            ):
                return source, node
    raise ValueError(f"Notebook assignment {name!r} was not found")


def load_case(catalog_path=None, request_path=None):
    """Reuse notebook data preparation and request without running auth/plot cells."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    _, url_node = _assignment(cells, "catalog_url")
    catalog_source = catalog_path or ast.literal_eval(url_node.value)
    catalog = pd.read_csv(catalog_source, usecols=COLUMNS)
    preparation, _ = _assignment(cells, "stars")
    namespace = {"pd": pd, "catalog": catalog}
    exec(compile(preparation, str(NOTEBOOK), "exec"), namespace)
    _, request_node = _assignment(cells, "request")
    request = (
        request_path.read_text(encoding="utf-8").strip()
        if request_path else ast.literal_eval(request_node.value)
    )
    if not isinstance(request, str) or not request.strip() or len(request) > 20_000:
        raise ValueError("The plotting request must contain between 1 and 20,000 characters")
    return namespace["stars"], request


def observations(figure, row_count):
    """Bounded evidence for review, deliberately not a visual-quality verdict."""
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    all_axes = [*figure.axes, *(child for ax in figure.axes for child in ax.child_axes)]
    label_boxes = [
        text.get_window_extent(renderer) for ax in all_axes for text in ax.texts[:100]
        if text.get_visible() and text.get_text().strip()
    ]
    axes = []
    point_count = 0
    finite_curves = True
    for ax in all_axes[:12]:
        for collection in ax.collections:
            offsets = collection.get_offsets()
            point_count += len(offsets)
        for line in ax.lines:
            try:
                xy = np.asarray(line.get_xydata(), dtype=float)
                finite_curves = finite_curves and bool(np.isfinite(xy).all())
            except (TypeError, ValueError):
                finite_curves = False
        axes.append({
            "x_scale": ax.get_xscale(), "y_scale": ax.get_yscale(),
            "x_limits": [float(value) for value in ax.get_xlim()],
            "y_limits": [float(value) for value in ax.get_ylim()],
            "lines": len(ax.lines), "collections": len(ax.collections),
            "annotations": len(ax.texts),
            "x_label": ax.get_xlabel()[:120], "y_label": ax.get_ylabel()[:120],
        })
    return {
        "axes": axes, "input_rows": row_count, "collection_offset_rows": point_count,
        "annotation_box_overlaps": int(sum(
            box.overlaps(other) for i, box in enumerate(label_boxes)
            for other in label_boxes[i + 1:]
        )),
        "annotation_boxes_outside_figure": int(sum(
            not (figure.bbox.contains(box.x0, box.y0)
                 and figure.bbox.contains(box.x1, box.y1)) for box in label_boxes
        )),
        "checks": {
            "collection_rows_cover_input": point_count >= row_count,
            "finite_line_coordinates": finite_curves,
            "has_multiple_y_axes": len(all_axes) >= 2,
        },
        "native_log_y_axis": any(ax.get_yscale() == "log" for ax in all_axes),
    }


def evaluate(args):
    stars, request = load_case(args.catalog, args.request_file)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "request.txt").write_text(request + "\n", encoding="utf-8")
    request_hash = hashlib.sha256(request.encode()).hexdigest()
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(stars, index=True).values.tobytes())
    complete = provider.complete
    for number in range(1, args.runs + 1):
        directory = args.output / f"attempt-{number:02d}"
        directory.mkdir(exist_ok=True)
        result = {
            "attempt": number, "auth": args.auth, "model": args.model,
            "request": request, "request_sha256": request_hash,
            "data_sha256": data_hash.hexdigest(), "input_rows": len(stars),
            "max_model_calls": 1, "max_repairs": 0, "model_calls": 0,
            "human_visual_review_required": True, "automatic_visual_pass": None,
            "status": "error",
            "versions": {
                "prompt": PROMPT_VERSION, "plotting": PLOTTING_VERSION,
                "validator": VALIDATOR_VERSION,
                **{name: version(name) for name in ("numpy", "pandas", "matplotlib", "scipy")},
            },
        }
        captured_source = None

        def counted_complete(_result=result, **kwargs):
            nonlocal captured_source
            _result["model_calls"] += 1
            _result["model"] = kwargs.get("model", _result["model"])
            reply = complete(**kwargs)
            try:
                parsed = json.loads(reply)
                code = parsed.get("code")
                if isinstance(code, str):
                    captured_source = code[:100_000]
            except (TypeError, ValueError, AttributeError):
                pass
            return reply

        started = time.monotonic()
        chart = None
        try:
            with patch.object(provider, "complete", counted_complete):
                chart = ap.plot(
                    stars, prompt=request, auth=args.auth, model=args.model,
                    backend="matplotlib", show=False, cache_dir=None, regenerate=True,
                    max_repairs=0, max_model_calls=1, timeout=args.timeout,
                )
            captured_source = chart.code
            result["model"] = chart.model
            result["model_calls"] = chart.model_calls
            result["explanation"] = chart.explanation
            result["versions"]["profile"] = chart.profile["profile_version"]
            result["observations"] = observations(chart.figure, len(stars))
            result["checks_pass"] = all(result["observations"]["checks"].values())
            chart.figure.savefig(directory / "figure.png", dpi=150)
            result["status"] = "rendered"
        except Exception as exc:
            # Exception text may contain provider identities, local paths, or secrets.
            result["error"] = {
                "type": type(exc).__name__,
                "message": "Attempt failed; inspect local configuration or saved generated source.",
            }
            result["checks_pass"] = False
        finally:
            result["seconds"] = round(time.monotonic() - started, 3)
            if captured_source is not None:
                result["source_sha256"] = hashlib.sha256(captured_source.encode()).hexdigest()
                (directory / "generated.py").write_text(captured_source, encoding="utf-8")
            (directory / "result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            if chart is not None:
                plt.close(chart.figure)
        print(f"Attempt {number}: {result['status']}; calls={result['model_calls']}; "
              f"seconds={result['seconds']}; human visual review required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth", choices=["chatgpt", "api_key", "auto"], default="chatgpt")
    parser.add_argument("--model", help="Optional model ID accepted by the selected auth transport")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, help="Existing public HYG v4.1 CSV; else download")
    parser.add_argument("--request-file", type=Path, help="Freeze this caller-written request")
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    if not 1 <= args.runs <= 20:
        parser.error("--runs must be between 1 and 20")
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty; use separate directories for before/after runs")
    evaluate(args)


if __name__ == "__main__":
    main()
