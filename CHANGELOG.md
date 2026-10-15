# Changelog

All notable changes to Augplot are documented in this file.

## Unreleased

### Added

- Reusable count-weighted empirical smoothing and renderer-aware direct curve labels,
  with SciPy-backed helpers bundled into standalone Python exports.
- Bounded grouped relationship profiles, numeric quantiles, and explicit coverage,
  sampling, log-transform, and truncation metadata.
- Optional `max_model_calls` budget shared across generation, repairs, and clarification,
  and inspectable attempted-call counts on charts and Augplot failures.
- An opt-in H–R visual-evaluation harness and unrelated complex-plot regressions.

### Changed

- Guide generation toward statistical smoothing and consistent external-colorbar layout.
- Version helper/profile changes in history identity while preserving existing workflow,
  live Figures, provider setup, and execution guardrails.
- Refresh the H–R notebook with a short appearance-focused request and strict fresh
  one-call generation. Document sparse and contaminated class-label limits.

## [0.2.0] - 2026-09-23

### Changed

- Replace the plotting capability manifest with a smaller security-focused validator.
  Common Pandas/NumPy transformations, local assignments, and data-sized loops now work
  without method-by-method approval.
- Keep checks for imports, file and network access, subprocess and dynamic execution,
  reflection, active rendering options, and obvious resource bombs.
- Shorten generation guidance and invalidate cached code accepted under the old policy.
- Report generation and repair progress, including a notice after 20 seconds of waiting.
- Improve chart guidance for titles, legends, labels, uncertainty notes, and spacing.
- Refresh the quickstart notebook and add an anonymized cross-validation walkthrough.

## [0.1.1] - 2026-09-23

### Changed

- Broaden guarded plotting compatibility for timeline annotations, axis styling,
  data-sized positions, and additional Pandas and NumPy data shapes.

## [0.1.0] - 2026-09-22

Initial beta release.

### Added

- Notebook-first `plot`, `refine`, `render`, and `to_python` workflow.
- Support for Pandas objects, NumPy arrays, lists, and nested dictionaries.
- Persistent visualization history for replaying and revising generated plots.
- Validated local execution of generated Matplotlib and Seaborn code.
- Strict JSON Schema model responses with explicit provider capability checks.
- Seaborn example-dataset loading and a quickstart notebook.
