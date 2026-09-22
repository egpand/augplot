# Augplot

Augplot generates plots from data already in a Jupyter notebook. Describe the chart,
refine it in place, and continue with the analysis instead of stopping to write plotting
code.

It accepts Pandas objects, NumPy arrays, lists, and nested dictionaries. Generated plots
use Matplotlib or Seaborn and include the Python source that produced them.

## Install

Augplot supports Python 3.11 through 3.14.

```bash
pip install augplot
```

Then open or download [the example notebook](examples/quickstart.ipynb) to try the
workflow.

## Usage

Configure a model and its provider credentials:

```bash
export AUGPLOT_MODEL="openai/YOUR_MODEL_ID"
export OPENAI_API_KEY="..."
```

For the MVP, the configured model must support strict JSON Schema responses and be
recognized as such by LiteLLM. Augplot checks this capability before inference and
raises `ConfigurationError` for unsupported model/provider combinations rather than
falling back to unconstrained text output.

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

### Workflow

The typical Augplot workflow is:

```text
plot → refine as needed → to_python
          └── render anytime
```

- `plot()` generates the initial visualization.
- `refine()` revises the current version and can be repeated.
- `render()` applies the current visualization to compatible data without a model call.
- `to_python()` exports the current version as standalone Matplotlib or Seaborn code.

Only `plot()` and `refine()` may call the configured model.

Generated and refined versions are saved in `.augplot/plots`, allowing matching
workflow steps to replay without another model call. Keep `.augplot/` with the
notebook when you want its visualization history to persist. See [History](#history).

The returned visualization object exposes the model-free reuse and export steps:

```python
plt.render(penguins[penguins["island"] == "Biscoe"])
plt.figure.savefig("penguin_bills.png", dpi=300)
plt.to_python(function_name="plot_penguin_bills")
```

With no path, `to_python()` writes `augplot_utils.py` in the notebook's working directory
and returns its path. New function names are appended; rerunning the same function safely
updates its Augplot-generated definition. Inspect the generated source directly with
`plt.code`.

See the [API and workflow reference](docs/api.md) for every command, option, and
inspectable attribute.

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
    backend="seaborn",       # auto, matplotlib, or seaborn
    display_format="retina", # retina, png, or svg
    show=True,
)
```

Use `AUGPLOT_MODEL` for the model and `AUGPLOT_API_BASE` for a custom endpoint. Providers
use the credentials expected by LiteLLM, but only model/provider combinations for which
LiteLLM reports strict response-schema support are accepted in the MVP.

## History

Augplot stores generated and refined plots in `.augplot/plots`. Rerunning the same input
replays saved code without a model call. Keep `.augplot/` with the notebook if you want
that history to persist. Pass `regenerate=True` to request new code or `cache_dir=None`
to disable persistence.

See [visualization history](docs/visualization-history.md) for the replay rules.

## Data and generated code

Augplot sends the configured model a bounded profile of your data, including samples,
field names, and statistics. This is not anonymization; `sample_rows=0` omits samples but
not all schema or summary information.

Generated Python runs locally against a copy of the data after validation. Invalid fresh
output gets at most one repair attempt; rejected code never runs or saves. This is defense
in depth, not an OS sandbox. See [generated-code guardrails](docs/generated-code-guardrails.md).

Augplot visualizes supplied data only. It can compute plot-related summaries and trends,
but does not train models or return predictions, forecasts, or other analytical artifacts.

## Beta status and limitations

Augplot 0.1.0 is a beta release. APIs and saved-history formats may change before 1.0,
and plotting requires a configured model with strict JSON Schema support. Generated code
is validated before local execution, but this validation is defense in depth rather than
an OS sandbox. Review generated code before using Augplot with sensitive data or in
security-critical environments.

## Security

Report vulnerabilities privately as described in the [security policy](SECURITY.md). See
[generated-code guardrails](docs/generated-code-guardrails.md) for Augplot's validation
and execution boundaries.

## License

Augplot is licensed under the [Apache License 2.0](LICENSE).

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check .
python -m build
```
