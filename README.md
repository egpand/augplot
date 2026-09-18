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

training_history = ap.load_dataset("training_history")
viz = ap.plot(training_history, prompt="Plot training and validation loss by epoch")
viz.refine("Use a log scale and mark the epoch with the lowest validation loss")
```

`ap.plot()` returns the visualization, so it can be reused without another model call:

```python
viz.render(training_history)
viz.figure.savefig("loss.png", dpi=300)
viz.save("plots.py", function_name="plot_training_loss")
```

Inspect the generated source with `viz.code`.

`ap.load_dataset()` includes `training_history`, `cv_results`, and `forecast_results`.
The datasets are packaged with Augplot and load without a network connection.

## Configuration

Pass options directly to `ap.plot()`:

```python
viz = ap.plot(
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

## Development

```bash
python -m pip install -e '.[dev,plotly]'
python -m pytest
python -m ruff check .
python -m build
```
