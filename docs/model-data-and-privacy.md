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
Numeric profiles also include finite/positive sample counts and five-number quantiles;
positive values spanning at least three decades receive labeled log10 quantiles.

DataFrames can also send up to two numeric relationship summaries, prioritized by column
mentions in the request and then schema order. Each uses at most 100,000 spread-out rows
and can group by one eligible field with at most 12 observed categories. Groups include
finite-pair counts, quantiles, and up to 12 occupied x-bin ranges with counts and y
quartiles. Positive responses spanning several decades may use explicitly labeled log10
bin summaries. Counts say whether they refer to the exact input or sampled rows. These
summaries disclose relationships and small groups even when `sample_rows=0`.

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

## ChatGPT connection and storage

ChatGPT sign-in opens your system browser for authorization and stores access and refresh
credentials in the operating system's secure credential store.
Registration metadata and model preferences live in Augplot's user configuration
directory, separate from notebook files and `.augplot/` plot history. Tokens and account
identifiers are not recorded in plot cache keys, exported Python, or generation
diagnostics. Account commands expose only safe labels and connection status.

ChatGPT generation uses the public Responses API with `store=false` and streamed output.
This flag does not promise zero provider retention; provider policies still apply.
The full data remains local, but the profile, prompt, generated code during refinement,
and clarification answers cross the same disclosure boundary described above. Augplot
does not import your ChatGPT conversation history. Model responses must complete
successfully before their code is validated or executed.

`ap.sign_out()` attempts to revoke the session and removes local credentials; it does not
delete saved chart code or notebook output. Review those artifacts before sharing them.
Permission to use the ChatGPT plan is distinct from signing in. Requests consume the
eligible account's plan allowance; optional additional-credit use is controlled in
ChatGPT's Usage settings. See
[OpenAI's sign-in guide](https://learn.chatgpt.com/docs/sign-in-with-chatgpt).

## Disclosure limits

`sample_rows=0` removes row and sequence samples, but **not** field names, dictionary
keys or scalar values, statistics, or category labels and counts. `max_profile_chars`
(default 20,000) caps only the serialized profile, not the whole request. When needed,
Augplot trims samples and relationship-bin detail before category counts and marks
omissions; complete quantile arrays and bin sets are omitted rather than cut midway.
Even a bounded profile
can disclose a small or distinctive dataset; remove identifying fields or values before
passing sensitive data to Augplot.
