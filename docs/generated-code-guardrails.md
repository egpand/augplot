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

## Security checks

The validator rejects:

- Imports outside NumPy, Pandas, Matplotlib pyplot/ticker/dates, and Seaborn; Seaborn
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

The checks are intentionally modest. Data-sized loops and allocations are allowed,
so a large input or expensive plotting operation can still use substantial CPU or
memory. The AST rules reduce obvious side effects but cannot guarantee that every
public third-party method is side-effect free. Do not use Augplot with sensitive data
where executing model-produced Python is unacceptable.

## Validation failures and saved plots

Fresh model output receives at most the configured repair attempt (`max_repairs=1` by
default). The repair request contains a sanitized validation diagnostic, not data
values or a runtime exception message. If repair fails, `ap.plot()` raises
`GenerationError`; rejected source is not executed, displayed, exported, or saved.

```python
try:
    viz = ap.plot(data, prompt="...")
except ap.GenerationError as exc:
    print(exc)
    print(exc.violations)
    print(exc.code)
```

Saved source is revalidated on every replay. The validator version is part of the
cache identity and metadata, so code accepted under an older policy is not silently
reused. A cache checksum establishes file integrity, not trust. If saved code is
rejected, Augplot does not make an automatic model request; use `regenerate=True`
when you want new code.
