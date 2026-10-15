# Generated-code guardrails

Augplot validates model-generated Python before running it in the notebook process.
The 0.2.0 validator focuses on security-sensitive operations and obvious resource
bombs. It is a defense-in-depth check, **not an operating-system sandbox**. Python
libraries can have side effects that static checks cannot reliably identify.

## Generated-code contract

Code defines one undecorated `plot_data(data, *, title=None, figsize=None)` function,
imports approved plotting/data libraries inside it, and returns a Matplotlib Figure.
The returned value is checked at runtime. Augplot passes a copy of the input data
and closes figures opened during execution. Generated source may use ordinary Pandas
and NumPy transformations, assignment to local data, loops, comprehensions, and
Matplotlib or Seaborn plotting APIs. There is no chart-type or artist-method manifest.

Execution exposes a limited set of builtins, including `object` for NumPy annotation
arrays. Pandas and Python date formatting can import their internal `time` helper;
generated source still cannot import `time` directly. The builtin `format(value, spec)`
is accepted only with a literal specification of at most 40 characters and bounded
numeric widths/precision (at most 1,000, as for f-strings). It cannot be aliased,
reassigned, or used as a callback.

## Security checks

The validator rejects:

- Imports outside NumPy, Pandas, Matplotlib pyplot/ticker/dates, Seaborn, and the shared
  `augplot.plotting` helpers; Seaborn
  imports also require a compatible backend.
- File, network, subprocess, environment, serialization, and plot-export APIs,
  including read/write/save methods and module-internal traversal.
- Dynamic execution, indirect calls, private/dunder access, nested definitions,
  reflection routes, and active rendering options such as external URLs or `usetex`.
- Pandas string dispatch to arbitrary method names and explicit plotting-backend
  selection. Simple aggregation names and direct safe NumPy functions are accepted.
- Oversized source, ASTs, literals, numeric ranges, static array allocations,
  subplot grids, figure dimensions, and format widths.

For example, these are rejected before compilation:

```python
pd.read_csv("https://example.invalid/data.csv")
frame.to_pickle("chart.pkl")
fig.savefig("chart.png")
```

For named Pandas `.agg`, use literal source-column strings and simple aggregation names
in each tuple. If the column name is stored in a variable, aggregate the selected Series
directly. For empirical smooth guides, the generation prompt recommends the shared
`smooth_trend` helper instead of hand-written global polynomial fits.

`import augplot.plotting as apx` exposes only direct calls to `apx.smooth_trend` and
`apx.label_curves`. Generated code cannot alias or pass this module as data, access other
members, or change its attributes. The trusted label helper can measure its renderer;
generated code still cannot access `fig.canvas`. Exports bundle the used trusted helper
definitions; generated code still cannot define additional functions itself. SciPy is
used internally by the smoother and is not a permitted generated-code import.
The complete constructor call `plt.cm.colors.LinearSegmentedColormap.from_list(...)`
is accepted for custom palettes. This compatibility case does not expose the intermediate
modules or constructor as stored values, and does not permit arbitrary module traversal.
A local colormap object may be called when its only binding is one simple assignment
from `plt.get_cmap(...)` or that complete palette constructor. The pyplot alias must
come from its approved import. Rebinding, parameter/loop/comprehension shadowing,
colormap aliases, and arbitrary stored callables remain rejected.

The checks are intentionally modest. Data-sized loops and allocations are allowed,
so a large input or expensive plotting operation can still use substantial CPU or
memory. The AST rules reduce obvious side effects but cannot guarantee that every
public third-party method is side-effect free. Do not use Augplot with sensitive data
where executing model-produced Python is unacceptable.

## Validation failures and saved plots

Fresh model output receives at most the configured number of repair attempts
(`max_repairs=1` by default, or 2 for three total attempts). Attempts
also respect `max_model_calls` when set; that cap includes clarification requests.
`max_model_calls=1` makes a single attempt, with call counts available on the chart or
raised Augplot exception. Each repair request
contains a sanitized validation diagnostic, not data values or a runtime exception
message. For an undefined name, the diagnostic identifies it only when a bounded
identifier appears at the failing reference in generated source; library-internal
names remain excluded. If repair fails, `ap.plot()` raises
`GenerationError`. Rejected source is not executed, displayed, exported, or saved as
visualization history. With history enabled, Augplot saves an `errors.json` summary and
each rejected candidate under `.augplot/plots/<request-root>/failures/failure-*/` by
default. The request root includes the full-data fingerprint and generation settings.
The exception message and `exc.diagnostics_path` give the directory. A malformed model
response is saved as `response-N.txt` when no source can be parsed. `cache_dir=None`
disables these files along with visualization history.

```python
try:
    viz = ap.plot(data, prompt="...")
except ap.GenerationError as exc:
    print(exc)
    print(exc.violations)
    print(exc.code)
    print(exc.diagnostics_path)
```

These files can contain generated source or response text with literal values from the
request. Review them before sharing. Augplot does not save the input data, profile,
credentials, or the full model request in the diagnostic directory.

Saved source is revalidated on every replay. The validator version is part of the
cache identity and metadata, so code accepted under an older policy is not silently
reused. A cache checksum establishes file integrity, not trust. If saved code is
rejected, Augplot does not make an automatic model request; use `regenerate=True`
when you want new code.
