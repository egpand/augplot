# How visualization history works

Each `Visualizer` tracks its current version. Saved history is shared across instances
and kernel restarts. Python variable names don't identify saved plots.

```python
from himalia import plot

campaign_viz = plot(data)                  # original A
campaign_viz.refine("Horizontal bars")     # A → B
campaign_viz.refine("Add labels")          # B → C

viz = plot(data)                           # loads A, not C
viz.refine("Horizontal bars")              # loads B
viz.refine("Add labels")                   # loads C
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

- **Rerun from `plot(data)`:** replay the original and its refinements from disk.
- **Repeat only a refinement cell:** edit the current version again; this may call the LLM.
- **`viz.render(new_data)`:** reuse current code without inference or a new version.
- **`regenerate=True`:** explicitly replace a step's lookup; earlier source files remain.
  Remove the flag afterward to resume reuse.
- **`viz.save("vis_utils.py", function_name="plot_results")`:** export the current version.

Inspect `viz.history_path`, `viz.data_fingerprint`, and `viz.cache_hit` for the current
source file, fitted-data hash, and whether the last fit/refine reused saved code.

**Keep `.himalia/` alongside your notebook.** History defaults to `.himalia/plots/`
relative to the kernel's working directory; saving the `.ipynb` alone doesn't include it.
Use `cache_dir=` for a consistent location or `cache_dir=None` to disable persistence.
Replay needs the same model configuration but no API key. It preserves code;
identical appearance also depends on data, dependencies, and plotting randomness.

[Back to README](../README.md)
