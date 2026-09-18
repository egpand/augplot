# Augplot

Augplot generates plots from data already in a Jupyter notebook. Describe the chart,
refine it in place, and continue with the analysis instead of stopping to write plotting
code.

It accepts Pandas objects, NumPy arrays, lists, and nested dictionaries. Generated plots
use Matplotlib, Seaborn, or Plotly and include the Python source that produced them.

## Install

Augplot requires Python 3.11+ and is not yet published to PyPI.

```bash
git clone https://github.com/egpand/augplot.git
cd augplot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[plotly]" jupyterlab ipykernel
python -m ipykernel install --user --name augplot --display-name "Python (Augplot)"
```

Open [the example notebook](examples/quickstart.ipynb):

```bash
python -m jupyterlab examples/quickstart.ipynb
```

## Usage

Configure a model and its provider credentials:

```bash
export AUGPLOT_MODEL="openai/YOUR_MODEL_ID"
export OPENAI_API_KEY="..."
```

Then work directly with notebook data:

```python
import augplot as ap

penguins = ap.load_sns_dataset("penguins")
plt = ap.plot(penguins, prompt="Compare bill length and depth across penguin species")
plt.refine(
    "Use one panel per species, add a Seaborn regression trend as a visual summary, "
    "and keep sex visible"
)
```

`ap.plot()` returns the visualization, so it can be reused without another model call:

```python
plt.render(penguins[penguins["island"] == "Biscoe"])
plt.figure.savefig("penguin_bills.png", dpi=300)
plt.to_python(function_name="plot_penguin_bills")
```

With no path, `to_python()` writes `augplot_utils.py` in the notebook's working directory
and returns its path. New function names are appended; rerunning the same function safely
updates its Augplot-generated definition. Inspect the generated source directly with
`plt.code`.

`ap.load_sns_dataset()` is a thin wrapper around `seaborn.load_dataset()`, so examples do
not need a separate Seaborn import. The first load requires internet access; Seaborn caches
the CSV locally by default. Dataset documentation and provenance belong to the
[Seaborn example-data catalog](https://github.com/mwaskom/seaborn-data). These datasets
are intended for examples, not production data.

## Configuration

Pass options directly to `ap.plot()`:

```python
plt = ap.plot(
    data,
    backend="seaborn",       # auto, matplotlib, seaborn, or plotly
    display_format="retina", # retina, png, or svg
    show=True,
)
```

Use `AUGPLOT_MODEL` for the model and `AUGPLOT_API_BASE` for a custom or local endpoint.
Anthropic and other providers use the credentials expected by LiteLLM. For local Ollama,
use a model such as `ollama_chat/YOUR_MODEL` and set `AUGPLOT_API_BASE`.

## History

Augplot stores generated and refined plots in `.augplot/plots`. Rerunning the same input
replays saved code without a model call. Keep `.augplot/` with the notebook if you want
that history to persist. Pass `regenerate=True` to request new code or `cache_dir=None`
to disable persistence.

See [visualization history](docs/visualization-history.md) for the replay rules.

## Data and generated code

Augplot sends the configured model a bounded profile of the data, including samples,
field names, and statistics. This is not anonymization. Set `sample_rows=0` to omit sample
rows, but names, statistics, and scalar dictionary values may still be included.

Generated Python runs locally against a copy of the full data. The validation checks are
not a security sandbox; review generated code before using it with sensitive data.

The scope boundary is the selected visualization backend. If Matplotlib, Seaborn, or
Plotly can compute something from the supplied data while rendering the figure, it is in
scope. That includes aggregation, histogram bins, density estimates, regression or
smoothing trends, descriptive error bars, and confidence intervals.

The boundary is the figure: Augplot does not use a separate modeling system or return a
fitted model, transformed dataset, predictions, or other analytical artifacts. Future
predictions, forecasts, and their intervals must be supplied upstream, although the
selected backend can visualize them.

## Development

```bash
python -m pip install -e '.[dev,plotly]'
python -m pytest
python -m ruff check .
python -m build
```
