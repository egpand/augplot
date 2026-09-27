# Model data and privacy

Augplot sends a JSON **profile**, not the full input object, to the configured model.
The profile is not anonymized. Inspect or sanitize sensitive data before calling
`ap.plot()` or `fit()`.

## Profile contents

| Input | Information sent |
| --- | --- |
| DataFrame | Shape, index type and names, up to 100 column names and dtypes, sampled statistics, eligible category counts, and spread-out sampled rows (including index values). |
| Series | Length, name, dtype, sampled statistics, eligible category counts, and sampled index/value pairs. |
| NumPy array or sequence | Shape or length, dtype where available, spread-out samples, and sampled statistics for supported numeric forms. A list or tuple of records also gets eligible field category counts. |
| Dictionary | Length, up to 100 keys, and recursively described values; scalar values may be included. |

`sample_rows` defaults to 5. DataFrame row samples include up to 100 values each.
Numeric statistics use at most 1,000 spread-out observations: sampled missing count,
min, max, mean, and standard deviation where applicable. Augplot does not send full
`pandas.DataFrame.describe()` output or a row-level missingness mask.

For text, boolean, and Pandas categorical fields in DataFrames, Series, and record
tables, Augplot counts all values locally. The profile reports exact missing and
distinct counts, plus the real labels and counts for all categories up to 20 distinct
values. Above 20, it includes the 10 most and 10 least frequent and the number omitted.
Ordinary numeric fields do not receive category counts.

## Requests and local work

Generation sends the plotting request, backend, and profile with system instructions.
Refinement also sends the previous generated code and original request. A repair sends
the prior model response (capped at 60,000 characters), a sanitized diagnostic
(capped at 500 characters), and safe replacement guidance for recognized validator
errors. Notebook clarification answers are appended to the request.
Matching history replays, `render()`, and `to_python()` make no model request.

The full input stays local for plotting. Augplot validates generated Python, then runs
it locally on a copy. History saves generated source, explanation, and a full-data
fingerprint, not the input or profile; generated source may contain literal values.
After all generation attempts fail, a private directory under the matching history root
holds rejected source and validation errors. If a response cannot be parsed, its first
60,000 characters are saved instead. These local diagnostic files may contain literal
values and should be reviewed before sharing; they do not contain the full input or
profile. `cache_dir=None` disables diagnostic persistence.
Augplot disables LiteLLM caching. Provider retention depends on the configured provider.

## Disclosure limits

`sample_rows=0` removes row and sequence samples, but **not** field names, dictionary
keys or scalar values, statistics, or category labels and counts. `max_profile_chars`
(default 20,000) caps only the serialized profile, not the whole request. When needed,
Augplot trims samples before category counts and marks omissions. Even a bounded profile
can disclose a small or distinctive dataset; remove identifying fields or values before
passing sensitive data to Augplot.
