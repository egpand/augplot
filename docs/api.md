# API and workflow reference

The README is the shortest path to a first chart. This page covers Augplot's complete
public notebook workflow.

## Sign in and choose a model

ChatGPT sign-in is included in the standard `pip install augplot`. An eligible Plus or
Pro account can authorize access to its plan's Codex / ChatGPT Work allowance without
entering an API key. Extra credits are optional; manage whether other apps may use them
after included usage runs out in ChatGPT's Usage settings.
See [OpenAI's user guide](https://learn.chatgpt.com/docs/sign-in-with-chatgpt).

```python
import augplot as ap

ap.sign_in()               # Continue with ChatGPT in your system browser.
ap.models()                # DataFrame: id, name, selected.
ap.set_model("model-id")    # Save a listed ID as this account's default.
```

`sign_in(profile_id=None, *, new_account=False, timeout=180)` remembers the connection
through the operating system's secure credential store. It requires a secure storage
backend and returns safe account information, never tokens. Sign-in and permission to
use the plan are separate: an account without plan-use permission cannot generate charts.
Importing Augplot does not sign in or contact a provider. Later sessions can reuse the
saved connection; plotting never opens the sign-in browser automatically.

`accounts()` returns dictionaries with `profile_id`, `label`, `connected`, `plan_enabled`,
`model`, and `active`, reading saved metadata without contacting OpenAI or accessing
credentials. Use `select_account(profile_id)` to change the active account,
`sign_in(new_account=True)` to connect another, and `sign_out(profile_id=None)` to revoke
and remove the selected connection's local credentials.

`models()` lists the selected account's available models, which can differ from the
ChatGPT picker and API catalog. Augplot chooses the first visible catalog model when no
default is saved. `set_model(model_id)` validates and remembers a catalog ID per account.
It affects new charts; an existing chart keeps its original account and model for later refinements.
An explicit `model=` overrides the default for a single chart.

## Create a chart

```python
import augplot as ap

plt = ap.plot(data, prompt="Plot revenue by channel")
```

`ap.plot()` accepts Pandas objects, one- or two-dimensional NumPy arrays, lists, tuples,
and nested dictionaries. It returns the current Augplot chart object and displays its
figure by default.

```python
plt = ap.plot(
    data,
    prompt="auto",
    auth="auto",
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
| `auth` | `"auto"` preserves explicit API-provider configuration, otherwise uses the saved ChatGPT connection. `"chatgpt"` or `"api_key"` selects a route explicitly. |
| `model` | With `auth="chatgpt"`, a catalog ID, defaulting to the saved account choice. For API providers, a LiteLLM model identifier with strict response-schema support, defaulting to `AUGPLOT_MODEL`. |
| `backend` | `"auto"`, `"matplotlib"`, or `"seaborn"`. |
| `display_format` | Static notebook output: `"retina"`, `"png"`, or `"svg"`. |
| `api_base` | Custom provider endpoint. Defaults to `AUGPLOT_API_BASE`. |
| `sample_rows` | Rows included in the model-facing profile, from 0 to 100. |
| `max_profile_chars` | Maximum serialized profile size, from 500 to 100,000. |
| `timeout` | Provider timeout in seconds. |
| `max_repairs` | Number of repair requests after the initial attempt: 0, 1 (default), or 2. |
| `cache_dir` | Visualization-history directory; `None` disables persistence. |
| `show` | Display the resulting figure in the notebook. |
| `regenerate` | Request new code instead of replaying the matching saved step. |

For ChatGPT generation, prefer `auth="chatgpt"` when an old `AUGPLOT_MODEL` environment
variable is still set. Custom `api_base` endpoints are unsupported on this route.
Use [API-provider setup](../README.md#api-provider-setup) for an API-key workflow.
Authentication, permission, or usage failures never switch to another billing route.

The profile includes bounded category counts from the full input. See
[model data and privacy](model-data-and-privacy.md) for the model-facing payload and
the limits of `sample_rows` and `max_profile_chars`.

Fresh generations print the current attempt (`plotting` or `repairing`) and when returned
code is checked. A request still waiting after 20 seconds prints another status line.
ChatGPT requests identify the account and model and include a link to manage usage.
Saved-history replays do not make model requests or print these progress messages.

## Refine the current chart

```python
plt.refine("Use one panel per region and label the largest value")
```

`refine(prompt, *, show=True, regenerate=False)` updates the current chart. It returns
the same object, so calls can be chained. Each refinement uses the current generated
function as its parent.

## Answer a plotting question

If a chart needs a choice that the prompt and data do not settle, Augplot asks for it
automatically when running in a Jupyter notebook kernel:

```python
viz = ap.plot(data, "Plot the selected metric")
viz.refine("Use another metric")
```

Jupyter displays the standard `input()` field beneath the running cell. The answer is
sent with the original request, and Augplot retries up to twice. A blank answer or an
unresolved question raises `ap.InputRequiredError`. In a regular Python script or
terminal, Augplot raises that error immediately without waiting for input. `fit()`
uses the same behavior. Only a successful chart updates the current visualization or
its history. The answer becomes part of that request's cache key, so a different answer
creates a different saved chart. Rerunning the original prompt may ask again; giving
the same answer to the same question replays the saved chart without another generation
request.

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
| `ap.AuthenticationError` | ChatGPT sign-in, credential storage, session, or plan permission failure. |
| `ap.DataError` | Unsupported or unsafe-to-profile data. |
| `ap.ProviderError` | Provider authentication, connectivity, or request failure. |
| `ap.GenerationError` | Generated source or saved source failed validation/execution. |
| `ap.InputRequiredError` | A supported chart needs an essential input or choice. |
| `ap.ScopeError` | The request requires work outside the selected visualization backend. |

When all generation attempts fail, `GenerationError.diagnostics_path` points to a private
directory under the matching visualization history root with rejected source and an
`errors.json` summary. The exception message includes the path. If `cache_dir=None` or
the directory cannot be written, the error still raises and `diagnostics_path` is `None`.

See [visualization history](visualization-history.md) for cache keys, branching,
regeneration, and replay behavior. See [generated-code guardrails](generated-code-guardrails.md)
for validation, repair, and rejected-code behavior.

## Manually check the sign-in workflow

Most authentication, model selection, streaming, and replay behavior is covered by
offline tests with fake services. A real eligible account is needed to check the browser
experience, account-specific model catalog, plan allowance, and chart quality:

1. Run `ap.sign_in()`, grant plan-use permission, and inspect `ap.accounts()` and
   `ap.models()`.
2. Choose a listed ID with `ap.set_model()` and run the three example notebooks in order.
   Their setup cells skip browser sign-in when an active connection already exists.
3. Add `regenerate=True` to the initial `ap.plot()` call and selected `refine()` calls
   to exercise fresh inference; verify `cache_hit` is false and inspect the figure,
   explanation, and exported Python. Remove the flag after checking.
4. Restart the kernel and confirm the connection and model choice persist. Change the
   default and check that a new chart uses it while an existing chart retains its model.
5. Sign out with `ap.sign_out()`. The saved default remains available for cached replay;
   fresh generation should explain how to reconnect.

Live checks consume plan usage and remain opt-in. Dataset downloads can require internet
independently of model generation; the cross-validation notebook uses inline data.

After signing in locally, you can also run an opt-in automated smoke test with real
ChatGPT generation and refinement. It reuses your connection without opening a browser
and consumes plan usage:

```bash
AUGPLOT_LIVE_CHATGPT_TEST=1 python -m pytest -m live
```
