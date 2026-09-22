# Generated-code guardrails

Augplot executes generated plotting code locally only after validating it against a
default-deny capability manifest. This is defense in depth against unsafe provider output
and prompt injection in data values or column names; it is not an operating-system sandbox.

## What is permitted

Generated code must define exactly `plot_data(data, *, title=None, figsize=None)` and
return a figure it created. It may use the approved, in-memory parts of NumPy, Pandas,
Matplotlib, and Seaborn. This includes common chart types, subplot layouts, axes, legends,
artists, ticks, date formatters, and supported data transformations.

Imports use fixed aliases: `np`, `pd`, `plt`, `ticker`, `dates`, and `sns`.
Every call and attribute path is checked, and figures, axes, artists, and data-derived
values are tracked so that a valid object cannot be substituted with an arbitrary callable.

Loops are limited to approved Axes collections, small static sequences, or columns selected
from data explicitly bounded to at most 200 rows with `head` or `tail`. This supports panel
styling and per-record chart annotations without permitting unbounded or nested generated
loops.

The generation prompt summarizes the validator's main constraints so a provider is less
likely to emit code that needs repair. This is compatibility guidance only: the prompt is
not trusted or relied upon for enforcement, and generated source must still pass the
independent validator before execution.

## What is rejected

The validator rejects unknown or indirect call targets, module traversal, private or dunder
attributes, alias shadowing, and `*args` or `**kwargs`. It also rejects filesystem,
network, environment, subprocess, serialization, dynamic-execution, introspection, plot
export, and backend-changing operations. Source, AST, literals, and static loop bounds are
limited to keep validation and execution predictable.

For example, these are rejected before compilation:

```python
pd.io.common.os.environ
pd.io.common.urlopen("https://example.invalid")
plt.imsave("chart.png", data)
```

## If validation fails

Fresh model output receives at most the configured repair attempt (`max_repairs=1` by
default). The repair request contains a sanitized validation diagnostic, never a runtime
exception message or data values. If it still fails, `ap.plot()` raises
`GenerationError`; rejected source is never executed, displayed, exported, or saved.

```python
try:
    viz = ap.plot(data, prompt="...")
except ap.GenerationError as exc:
    print(exc)             # concise reason
    print(exc.violations)  # structured validation failures
    print(exc.code)        # inspect only when appropriate
```

Saved source is revalidated every time it is replayed. The capability-manifest version is
part of both the cache identity and its metadata, so changing the approved capabilities
cannot silently reuse code accepted under an older manifest. A cache checksum establishes
file integrity, not trust. If saved code is rejected, Augplot does not execute it or make
an automatic model request; use `regenerate=True` to explicitly request new code.

## Remaining risk

Validation substantially narrows what generated code can do, but does not isolate the
Python process or make untrusted code generally safe. Do not use Augplot with sensitive
data where executing model-produced Python is unacceptable.
