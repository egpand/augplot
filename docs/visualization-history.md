# How visualization history works

Each value returned by `ap.plot()` tracks its current version. Saved history is shared
across instances and kernel restarts. Python variable names don't identify saved plots.

```python
import augplot as ap

campaign_viz = ap.plot(data)                  # original A
campaign_viz.refine("Horizontal bars")     # A → B
campaign_viz.refine("Add labels")          # B → C

plt = ap.plot(data)                           # loads A, not C
plt.refine("Horizontal bars")              # loads B
plt.refine("Add labels")                   # loads C
```

With identical inputs and the same cache directory, the second sequence makes no LLM
calls. Each instance has its own current state. A different instruction creates a branch:

```text
A: original
├── B: horizontal bars
│   └── C: add labels
└── D: line chart
```

The original lookup combines a fingerprint of the full data—values, order, names,
and types—with the prompt and generation settings. Each refinement also includes its
exact parent version and instruction.

ChatGPT history also distinguishes its transport from API-provider requests. It includes
the selected model, never credentials or account identity. `ap.set_model()` affects new
charts; an existing chart keeps its original model for subsequent refinements. Choosing
a different model for a new chart creates a different lookup.

- **Rerun from `ap.plot(data)`:** replay the original and its refinements from disk.
- **Repeat only a refinement cell:** edit the current version again; this may call the LLM.
- **`plt.render(new_data)`:** reuse current code without inference or a new version.
- **`regenerate=True`:** explicitly replace a step's lookup; earlier source files remain.
  Remove the flag afterward to resume reuse.
- **`plt.to_python(function_name="plot_results")`:** write the current version as reusable
  Python source in `augplot_utils.py`, appending new functions or updating the same
  Augplot-generated function on reruns.

Inspect `plt.history_path`, `plt.data_fingerprint`, and `plt.cache_hit` for the current
source file, fitted-data hash, and whether the last fit/refine reused saved code.

**Keep `.augplot/` alongside your notebook.** History defaults to `.augplot/plots/`
relative to the kernel's working directory; saving the `.ipynb` alone doesn't include it.
Use `cache_dir=` for a consistent location or `cache_dir=None` to disable persistence.
Replay needs the same model configuration but no active connection or API key. ChatGPT
sign-out retains the saved default for offline replay. If that default has changed, pass
`auth="chatgpt", model="original-model-id"` to replay the original chart. Fresh generation
and refinement without a matching saved step require a valid connection. Replay preserves code;
identical appearance also depends on data, dependencies, and plotting randomness.

[Back to README](../README.md)
