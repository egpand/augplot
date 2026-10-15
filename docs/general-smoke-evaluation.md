# Five live smoke tests beyond H–R

On 2026-10-15, five fresh `gpt-6.1-sol` ChatGPT calls checked the other example use
cases. Every operation used `max_model_calls=1`, `max_repairs=0`, and disabled history
replay. Exactly five provider calls were made; none were added to repair failures.

| Case | Original live execution (guard 12) | Unchanged-source local replay (guard 13) |
| --- | --- | --- |
| Quickstart penguin scatter by species | Rejected: stored colormap call | Rendered and exported |
| Penguin species facets, shared axes, sex markers, regression trends | Rendered and exported | Rendered and exported |
| Quickstart airline lines, one per year | Rendered and exported | Rendered and exported |
| Airline heatmap refinement, calendar order, annotations, peak outline | Rejected: builtin `format()` | Rendered and exported |
| Cross-validation F1/ROC AUC grouped bars, mean labels, all tied winners | Rendered and exported | Rendered and exported |

The first scatter failure meant the penguin facets used the combined initial and
refinement sentences as an independent plot request. The heatmap was an actual
refinement of the freshly generated airline line chart. The cross-validation case
combined the comparison notebook's successive presentation requirements into one
request using its original anonymized data.

Two generic compatibility fixes address the rejected programs. Callable colormap
locals require one binding from an approved pyplot factory; arbitrary function aliases
and rebinding remain blocked. Direct builtin `format(value, literal_spec)` accepts only
bounded literal specs, with Unicode-aware width/precision limits. These rules have
focused positive and adversarial regression tests. See [guardrails](generated-code-guardrails.md).

All five figures were visually inspected for readable axes, legends, annotations and
layout. Additional local numerical checks verified:

- Both penguin programs retain all 342 finite bill pairs, including unknown sex in the
  faceted plot; its three species panels share axis limits.
- The 12 airline lines agree with all 144 observations in calendar order.
- All 144 heatmap cells and annotations agree with the input; July 1960's maximum of
  622 is outlined, with a separate unobstructed colorbar.
- All 16 cross-validation means and two-decimal labels agree with the input; both
  configurations F and G receive the tied-winner highlight.

Each standalone export executed on its corresponding data. Local revalidation made
zero model calls and did not edit generated code. Original first-attempt success was
**3/5**, while all five frozen responses work after the compatibility fixes. This small,
adaptive smoke sample does not establish prospective one-call reliability.

The [manifest](assets/general-smoke-results.json) records requests, source hashes,
original outcomes and revalidation checks. Inputs are the public Seaborn `penguins` and
`flights` datasets plus the comparison notebook's inline anonymized table. Figures and
raw responses were kept as local evaluation artifacts rather than replacing notebook
outputs or adding further example-specific code to the package.
