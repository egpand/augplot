# Himalia

Turn data in a Jupyter notebook into a visualization, refine it in plain English,
and save the resulting Python function for reuse without an LLM.

Himalia is an early MVP. It generates and executes Python locally after conservative
checks. **Those checks are not a security sandbox.** Use it in a notebook environment
where you are comfortable running model-generated code. The provider receives a
bounded data profile, including sampled values, field names, statistics, and your
prompt. Profiling is not anonymization. Generated code runs against the full local data.

## Quick start: your first notebook

You need **Git, Python 3.11 or newer, and an LLM provider's API key and model ID**.
Himalia is not published to PyPI yet, so install it from this repository. The commands
below are for macOS/Linux and use OpenAI as the example provider.

### 1. Download Himalia

```bash
git clone https://github.com/egpand/himalia.git
cd himalia
```

### 2. Create a virtual environment and install

A virtual environment keeps Himalia's dependencies separate from your other projects.
Create it once:

```bash
python3 --version  # Must be 3.11 or newer; otherwise use a newer Python installation.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[plotly]" jupyterlab ipykernel
python -m ipykernel install --user --name himalia --display-name "Python (Himalia)"
```

If your newer Python is named `python3.12`, use `python3.12 -m venv .venv` above.
The install includes the optional Plotly backend and JupyterLab; developer/test tools
are not needed to use Himalia.

### 3. Set two configuration values

Replace both placeholders below. `YOUR_MODEL_ID` must be a model available through
your provider account; keep the `openai/` prefix.

```bash
export HIMALIA_MODEL="openai/YOUR_MODEL_ID"
export OPENAI_API_KEY="your-api-key"
```

That's all the configuration required for this example. No config file is needed.
Other [LLM providers](#llm-providers) use their own model prefix and credentials.

### 4. Open and run the example

Launch JupyterLab **from the same terminal** so it inherits your configuration:

```bash
python -m jupyterlab examples/quickstart.ipynb
```

In the browser, select the **Python (Himalia)** kernel and run cells from top to bottom
with **Shift+Enter**. The first `plot(...)` call should display a chart beneath the cell.
The notebook then demonstrates refinement, exporting a reusable function, and plotting
a DataFrame. The Plotly example is optional; enable its `RUN_PLOTLY` toggle to try it.
Model calls use your provider account and may incur API charges.

For your own notebook, the minimal usage is:

```python
from himalia import plot

results_dict = {
    "ridge": {"r2": [0.71, 0.75, 0.73]},
    "forest": {"r2": [0.80, 0.82, 0.81]},
}
viz = plot(results_dict)
```

`backend="auto"` is the default. Add `backend="matplotlib"`, `"seaborn"`, or
`"plotly"` to choose the plotting library explicitly.

**Next time:** return to the repository, activate `.venv`, set the two environment
variables in your new terminal, and launch JupyterLab again. You do not need to
recreate the environment or reinstall packages.

**Using VS Code instead?** After installation, open `examples/quickstart.ipynb` and
select `.venv/bin/python` or **Python (Himalia)** as the notebook kernel. If VS Code
was not launched from the configured terminal, set the values in a new first cell:

```python
import os
from getpass import getpass

os.environ["HIMALIA_MODEL"] = "openai/YOUR_MODEL_ID"
os.environ["OPENAI_API_KEY"] = getpass("API key: ")
```

If `import himalia` fails, check that the selected kernel uses the environment where
you installed Himalia. If the notebook reports a missing model or credentials, set
the values in that kernel using the cell above, then rerun the failed cell.

## Notebook workflow

```python
from himalia import plot, Visualizer

results = {
    "ridge": {"r2": [0.71, 0.75, 0.73, 0.70, 0.74]},
    "forest": {"r2": [0.80, 0.82, 0.79, 0.84, 0.81]},
}

viz = plot(results)  # displays immediately, even when assigned
viz.refine("Show each fold and sort models by mean R²")

# You can specify the plotting library independently of the LLM provider:
viz = plot(results, prompt="Compare fold scores", backend="seaborn")

print(viz.explanation)
print(viz.code)
viz.figure.savefig("cv_results.png", dpi=160, bbox_inches="tight")
viz.save("vis_utils.py", function_name="plot_cv_results")
```

The exported function accepts data with the same structure and required fields but
different values or row counts. It requires only the libraries it imports, without
Himalia, API credentials, or additional inference:

```python
from vis_utils import plot_cv_results

fig = plot_cv_results(results, title="Cross-validation results", figsize=(9, 5))
fig  # displays in Jupyter
```

An existing module can receive multiple differently named functions. Himalia refuses
to overwrite an existing name or modify invalid Python. If you have already imported
the module in your notebook, reload it with `importlib.reload(vis_utils)` after adding
another function. Export does not write your dataset or credentials.

The same workflow is available through a class:

```python
viz = Visualizer(backend="auto", timeout=60, max_repairs=1)
viz.fit(results, prompt="Compare models across folds")
viz.render(results, title="Updated title")  # local execution, no LLM call
```

`fit()` snapshots the input. `refine()` uses that snapshot and the latest successful
code. A failed fit or refinement preserves the previous successful state. `render()`
reuses the function on supplied data without changing the fitted snapshot. Call
`fit(new_data)` to change the data used for subsequent refinement.

Set `show=False` to suppress display. In scripts, inspect or save `viz.figure`;
Himalia does not open a browser or GUI window automatically.

See [the example notebook](examples/quickstart.ipynb) for the complete workflow with
CV results, a DataFrame, refinement, export, and an optional interactive plot.

## Plotting backends

| `backend` | Output | Behavior |
| --- | --- | --- |
| `auto` (default) | Matplotlib Figure | Model selects a chart using Seaborn/Matplotlib. |
| `matplotlib` | Matplotlib Figure | Matplotlib alone for plotting. |
| `seaborn` | Matplotlib Figure | Seaborn with Matplotlib for layout and customization. |
| `plotly` | Plotly Figure | Interactive chart; install the `plotly` extra. |

`auto` deliberately uses the static stack in v0.1. Seaborn draws on Matplotlib, so
both static options have the same figure type and image-export API. For Plotly use
`viz.figure.write_html('chart.html')`; static Plotly image export needs additional
Plotly/Kaleido setup. `figsize` is in inches; Plotly code converts it to pixels at 100 dpi.

## LLM providers

Himalia uses the [LiteLLM Python SDK](https://docs.litellm.ai/docs/) behind an internal
adapter. Choose a LiteLLM model identifier and set that provider's credentials:

| Provider | Model identifier | Environment |
| --- | --- | --- |
| OpenAI | `openai/YOUR_MODEL_ID` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/YOUR_MODEL_ID` | `ANTHROPIC_API_KEY` |
| Local Ollama | `ollama_chat/YOUR_INSTALLED_MODEL` | `HIMALIA_API_BASE=http://localhost:11434` |

Other LiteLLM completion providers can use the same adapter. Local inference requires
a separately running server and a model capable of returning the requested JSON and
Python. Himalia does not download, serve, or train models. Provider support through
the adapter is not a guarantee of visualization quality for every model.

Pass `model=` and `api_base=` to override `HIMALIA_MODEL` and `HIMALIA_API_BASE`.
Credentials are resolved at request time; importing Himalia works without them.
There is no silently selected paid model and no automatic provider fallback.

## Data, errors, and limits

- Inputs: Pandas DataFrames/Series, 1D/2D NumPy arrays, lists/tuples, and nested
  dictionaries of ordinary data. CV dictionaries need no sklearn dependency.
- Unsupported custom objects, cyclic containers, and empty top-level inputs fail
  before inference. Convert specialized objects to one of the supported formats.
- Profiles default to five sampled rows/values per collection and 20,000 serialized
  characters, with structural truncation markers. Statistics use at most 1,000
  evenly spaced observations per series and identify when they are sampled.
- Customize `sample_rows` (0–100) and `max_profile_chars` (500–100,000). Setting
  `sample_rows=0` suppresses collection samples, **not** scalar dictionary values,
  names, or statistical summaries. Inspect `viz.profile` after fitting; for a
  preflight preview use `himalia.profiling.profile_data(data)` locally.
- Generated code gets a copy of the full input, not just the sample. Large inputs
  therefore incur copying and plotting costs. Fields omitted from the bounded
  profile may need a more specific prompt or a narrowed dataset.
- A fit/refinement makes one model request, plus at most one repair request for
  invalid code or execution failure. `max_repairs=0` disables repair. The timeout
  applies per inference request, not to local Python execution; interrupt the
  notebook kernel if generated plotting code runs too long.
- On failure, `GenerationError.code` contains the last parsed candidate. Provider
  errors and repair diagnostics omit raw SDK/runtime messages and local variables.
- Allowed imports and AST checks reject common I/O, dynamic execution, and
  introspection patterns. They cannot make arbitrary Python safe. Exported source
  runs as ordinary Python and does not carry the runtime checks or defensive copy.
- Review charts for statistical correctness. A working figure does not establish
  that the model chose a suitable visualization or interpreted the data correctly.

## Development

```bash
python -m pip install -e '.[dev,plotly]'
python -m pytest
python -m ruff check .
python -m build
```

The build creates a wheel that can be installed in another Python 3.11+ environment:

```bash
python -m pip install dist/himalia-0.1.0-py3-none-any.whl
```

The default tests mock inference and need no credentials. They cover profiling,
execution checks, repair, all plotting backends, standalone export, and real Jupyter
kernel display. The notebook integration test writes exports only inside a temporary
directory. Optional live smoke tests run only with `HIMALIA_LIVE_TEST=1` and a configured
model/API environment. The live test sends synthetic data and incurs provider usage.

Prompt templates are versioned in `src/himalia/prompts.py` to support future benchmarks.
