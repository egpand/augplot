# Augplot

Turn notebook data into a chart, refine it in plain English, and export reusable Python.
Supports Pandas, NumPy, lists, and nested dictionaries—including cross-validation results.

## Scope

**You produce the data and model results; Augplot helps communicate them.**
It computes chart summaries (means, variation, rankings, residuals) and visualizes
supplied predictions and intervals. Training, fine-tuning, model selection, and
generating new predictions belong upstream. Augplot can highlight the highest observed
CV score; it does not establish which model you should deploy.
The generation prompt sets this boundary. When the model flags a request
as out of scope, Augplot raises `ScopeError` without executing code or retrying it.

## Quick start

Requires **Python 3.11+** and an LLM provider's model ID and API key.
Not published to PyPI yet. On macOS/Linux:

```bash
git clone https://github.com/egpand/augplot.git
cd augplot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[plotly]" jupyterlab ipykernel
python -m ipykernel install --user --name augplot --display-name "Python (Augplot)"

python -m jupyterlab examples/quickstart.ipynb
```

Select **Python (Augplot)** and run cells with **Shift+Enter**. The first cell selects
`openai/gpt-5.6-terra` and prompts for your API key with hidden input; no terminal exports
are needed. In VS Code, open the notebook and select the same kernel.

Next time, activate `.venv` and launch Jupyter again. No reinstall needed.
The [example notebook](examples/quickstart.ipynb) shows CV winner highlighting and
visualizing supplied forecasts, intervals, and missed observations.

## Plot, refine, reuse

```python
import augplot as ap

results = {
    "ridge": {"r2": [0.71, 0.75, 0.73]},
    "forest": {"r2": [0.80, 0.82, 0.81]},
}

viz = ap.plot(results)  # automatically chooses a chart and displays it
viz.refine("Highlight the model with the highest mean R² and label its score")
viz.render(results, title="Model comparison")  # reuse code; no LLM call

print(viz.code)
viz.save("vis_utils.py", function_name="plot_cv_results")
```

The export runs without Augplot or an LLM, using the plotting libraries it imports:

```python
from vis_utils import plot_cv_results

fig = plot_cv_results(results)
fig
```

Exports refuse to overwrite existing function names. Refinement uses the fitted data
snapshot; `render(new_data)` keeps the current code and leaves that snapshot unchanged.

## Options

Pass these to `ap.plot()`:

| Option | Default | Purpose |
| --- | --- | --- |
| `backend` | `"auto"` | Seaborn/Matplotlib automatically; or `"matplotlib"`, `"seaborn"`, `"plotly"`. |
| `model` | `AUGPLOT_MODEL` | LLM provider/model identifier. |
| `api_base` | `AUGPLOT_API_BASE` | Custom or local inference endpoint. |
| `display_format` | `"retina"` | Static output: `"retina"`, `"svg"`, or `"png"`. |
| `cache_dir` | `".augplot/plots"` | Persistent history location; `None` disables it. |

Use `prompt="..."` to describe a chart and `show=False` to suppress display.
Retina output is automatic; notebook settings stay unchanged. Save static images with
`viz.figure.savefig("chart.png", dpi=300)`.

## Persistent history and notebook reruns

Successful generations and refinements are saved automatically. Identical data and
settings replay the original plot and its refinement sequence without LLM calls,
even after a kernel restart. Changed inputs may generate new code.

**Keep `.augplot/` with your notebook—the `.ipynb` alone does not contain the history.**
Use `regenerate=True` on `ap.plot()`, `fit()`, or `refine()` to explicitly request new code.

See [How visualization history works](docs/visualization-history.md) for examples.

## LLM providers

Inference uses LiteLLM. Set `AUGPLOT_MODEL` and the corresponding credentials:

| Provider | Model identifier | Configuration |
| --- | --- | --- |
| OpenAI | `openai/YOUR_MODEL_ID` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/YOUR_MODEL_ID` | `ANTHROPIC_API_KEY` |
| Local Ollama | `ollama_chat/YOUR_INSTALLED_MODEL` | `AUGPLOT_API_BASE=http://localhost:11434` |

Local models need a running server and must generate the required JSON/Python.
New generations use your provider account and may incur charges.

## Data and execution

The model receives a bounded profile with samples, field names, statistics, and your
prompt. Defaults: `sample_rows=5`, `max_profile_chars=20000`. This is not anonymization;
`sample_rows=0` still includes names, statistics, and scalar dictionary values.
Generated code executes locally against a copy of the full data.

**Execution checks are not a security sandbox.** Review generated code and charts.
Saved code and explanations may contain data details; review before sharing.
New generations allow one repair request (`max_repairs=0` disables it).
`timeout=60` limits each model request, not local execution. Saved-code failures raise
an error without silently calling the LLM.

## Development

```bash
python -m pip install -e '.[dev,plotly]'
python -m pytest
python -m ruff check .
python -m build
```

Tests mock inference and include notebook replay in fresh kernels. Optional live tests
require `AUGPLOT_LIVE_TEST=1` and provider configuration, and incur API usage.
