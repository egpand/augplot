# API and workflow reference

The README is the shortest path to a first chart. This page covers Augplot's complete
public notebook workflow.

## Create a chart

```python
import augplot as ap

plt = ap.plot(data, prompt="Plot revenue by channel")
```

`ap.plot()` accepts Pandas objects, NumPy arrays, lists, and nested dictionaries. It
returns the current Augplot chart object and displays its figure by default.

```python
plt = ap.plot(
    data,
    prompt="auto",
    model=None,
    backend="auto",
    display_format="retina",
    api_base=None,
    sample_rows=5,
    max_profile_chars=20_000,
    timeout=60,
    max_repairs=1,
    cache_dir=".augplot/plots",
    show=True,
    regenerate=False,
)
```

| Option | Meaning |
| --- | --- |
| `prompt` | Visualization instruction; `"auto"` asks Augplot to choose. |
| `model` | LiteLLM model identifier. Defaults to `AUGPLOT_MODEL`. |
| `backend` | `"auto"`, `"matplotlib"`, or `"seaborn"`. |
| `display_format` | Static notebook output: `"retina"`, `"png"`, or `"svg"`. |
| `api_base` | Custom provider endpoint. Defaults to `AUGPLOT_API_BASE`. |
| `sample_rows` | Rows included in the model-facing profile, from 0 to 100. |
| `max_profile_chars` | Maximum serialized profile size, from 500 to 100,000. |
| `timeout` | Provider timeout in seconds. |
| `max_repairs` | Whether invalid generated code gets zero or one repair request. |
| `cache_dir` | Visualization-history directory; `None` disables persistence. |
| `show` | Display the resulting figure in the notebook. |
| `regenerate` | Request new code instead of replaying the matching saved step. |

## Refine the current chart

```python
plt.refine("Use one panel per region and label the largest value")
```

`refine(prompt, *, show=True, regenerate=False)` updates the current chart. It returns
the same object, so calls can be chained. Each refinement uses the current generated
function as its parent.

## Reuse code with other data

```python
plt.render(new_data, title="Latest results", figsize=(10, 5), show=True)
```

`render(data=None, *, title=None, figsize=None, show=True)` runs the current function
locally without a model call. Omit `data` to render the original fitted data. Compatible
column names and structure are required.

Use `fit(data, prompt="auto", *, show=True, regenerate=False)` when new data should
become the basis for later refinements rather than merely being rendered once.

## Inspect the chart and its source

```python
plt.figure       # Matplotlib figure
plt.code         # Complete generated Python function
plt.explanation  # Model's short chart explanation
plt.profile      # Bounded data profile sent to the model
```

For syntax-highlighted source in Jupyter:

```python
from IPython.display import Code

Code(plt.code, language="python")
```

History metadata is also inspectable:

```python
plt.cache_hit        # Whether this step replayed saved code
plt.history_path     # Saved source path, or None when history is disabled
plt.data_fingerprint # Fingerprint of the full fitted data
```

## Write reusable Python

```python
python_path = plt.to_python(function_name="plot_results")
```

`to_python(path=None, *, function_name="plot_visualization")` writes a standalone
function without a model call. The default module is `augplot_utils.py` in the notebook's
working directory. New names are appended. Reusing a name updates only a function marked
as Augplot-generated; user-authored definitions are never overwritten.

Pass a path when another module is preferred:

```python
plt.to_python("analysis/plots.py", function_name="plot_results")
```

The returned `Path` identifies the file that was written. The generated function accepts
`data` and optional `title` and `figsize` keyword arguments and does not require Augplot or
provider credentials at runtime.

## Load Seaborn example data

```python
penguins = ap.load_sns_dataset("penguins")
```

`load_sns_dataset(name, *, cache=True, data_home=None, **read_csv_options)` delegates to
Seaborn without requiring a separate import. `ap.SNS_DATASETS` contains the names accepted
by this Augplot release. The first uncached load requires internet access.

## Errors

All public exceptions inherit from `ap.AugplotError`:

| Exception | Meaning |
| --- | --- |
| `ap.ConfigurationError` | Invalid options or missing model configuration. |
| `ap.DataError` | Unsupported or unsafe-to-profile data. |
| `ap.ProviderError` | Provider authentication, connectivity, or request failure. |
| `ap.GenerationError` | Generated source or saved source failed validation/execution. |
| `ap.ScopeError` | The request requires work outside the selected visualization backend. |

See [visualization history](visualization-history.md) for cache keys, branching,
regeneration, and replay behavior.
