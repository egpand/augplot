"""Restricted generated-source validation and local execution (not an OS sandbox)."""

import ast
import builtins
import json
from contextlib import ExitStack

from .errors import GenerationError, ScopeError
from .profiling import copy_data

API_MANIFEST_VERSION = 8
MAX_SOURCE_CHARS = 50_000
MAX_AST_NODES = 4_000
MAX_LITERAL_ITEMS = 2_000
MAX_STATIC_RANGE = 10_000
MAX_STATIC_INTEGER = 10_000_000
MAX_LOOPS = 10
MAX_BOUNDED_ITERATION = 200
_IMPORT_ALIASES = {
    "numpy": "np",
    "pandas": "pd",
    "matplotlib.pyplot": "plt",
    "matplotlib.ticker": "ticker",
    "matplotlib.dates": "dates",
    "seaborn": "sns",
}
_BUILTINS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "filter",
    "float",
    "int",
    "isinstance",
    "issubclass",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "range",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
}

# Default-deny, versioned capability manifest. Deliberately absent: IO, serialization,
# environment, subprocess, configuration/backend APIs, dynamic execution and reflection.
_NUMPY = {
    "abs",
    "all",
    "any",
    "argmax",
    "argmin",
    "argsort",
    "array",
    "asarray",
    "average",
    "ceil",
    "clip",
    "corrcoef",
    "count_nonzero",
    "cumsum",
    "diff",
    "digitize",
    "divide",
    "exp",
    "floor",
    "isfinite",
    "isinf",
    "isnan",
    "log",
    "log10",
    "max",
    "mean",
    "median",
    "min",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanpercentile",
    "nanstd",
    "nansum",
    "percentile",
    "quantile",
    "ravel",
    "reshape",
    "round",
    "sort",
    "sqrt",
    "std",
    "sum",
    "unique",
    "var",
    "where",
}
_RESOURCE_NUMPY = {
    "arange",
    "column_stack",
    "concatenate",
    "full",
    "histogram",
    "hstack",
    "linspace",
    "ones",
    "repeat",
    "vstack",
    "zeros",
}
_PANDAS = {
    "DataFrame",
    "Series",
    "crosstab",
    "cut",
    "qcut",
    "get_dummies",
    "infer_freq",
    "isna",
    "notna",
    "pivot_table",
    "to_datetime",
    "to_numeric",
}
_RESOURCE_PANDAS = {"concat"}
_DYNAMIC_DISPATCH_METHODS = {"agg", "aggregate", "apply", "map", "transform"}
_DYNAMIC_BACKEND_METHODS = {"hist", "plot"}
_DATA_METHODS = {
    "abs",
    "all",
    "any",
    "append",
    "astype",
    "between",
    "clip",
    "combine_first",
    "copy",
    "corr",
    "count",
    "cov",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "describe",
    "diff",
    "drop",
    "drop_duplicates",
    "dropna",
    "duplicated",
    "eq",
    "explode",
    "extend",
    "ffill",
    "fillna",
    "first_valid_index",
    "ge",
    "groupby",
    "gt",
    "head",
    "idxmax",
    "idxmin",
    "infer_objects",
    "interpolate",
    "iterrows",
    "isin",
    "isna",
    "items",
    "keys",
    "kurt",
    "last",
    "le",
    "lt",
    "max",
    "mean",
    "median",
    "melt",
    "min",
    "mode",
    "nlargest",
    "notna",
    "nsmallest",
    "nunique",
    "pct_change",
    "pivot",
    "quantile",
    "rank",
    "reindex",
    "rename",
    "reset_index",
    "resample",
    "reverse",
    "rolling",
    "round",
    "sample",
    "select_dtypes",
    "sem",
    "set_index",
    "shift",
    "skew",
    "sort",
    "sort_index",
    "sort_values",
    "squeeze",
    "std",
    "sum",
    "tail",
    "to_dict",
    "to_frame",
    "to_list",
    "to_numpy",
    "tolist",
    "transpose",
    "truediv",
    "unique",
    "unstack",
    "value_counts",
    "var",
    "where",
    "xs",
}
_DATA_PROPERTIES = {
    "T",
    "at",
    "columns",
    "dtypes",
    "empty",
    "iat",
    "iloc",
    "index",
    "loc",
    "name",
    "ndim",
    "shape",
    "size",
    "str",
    "values",
}
_BOUND_PRESERVING_DATA_METHODS = {
    "abs",
    "astype",
    "between",
    "clip",
    "copy",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "diff",
    "drop",
    "drop_duplicates",
    "dropna",
    "duplicated",
    "eq",
    "ffill",
    "fillna",
    "ge",
    "gt",
    "head",
    "isin",
    "isna",
    "le",
    "lt",
    "notna",
    "pct_change",
    "rank",
    "rename",
    "reset_index",
    "round",
    "select_dtypes",
    "shift",
    "sort_index",
    "sort_values",
    "squeeze",
    "tail",
    "to_frame",
    "to_list",
    "to_numpy",
    "tolist",
    "unique",
    "where",
}
_PYPLOT = {
    "Rectangle",
    "bar",
    "barh",
    "boxplot",
    "contour",
    "contourf",
    "errorbar",
    "eventplot",
    "figure",
    "fill",
    "fill_between",
    "hexbin",
    "hist",
    "hlines",
    "imshow",
    "pie",
    "plot",
    "scatter",
    "stackplot",
    "stem",
    "step",
    "subplots",
    "subplot_mosaic",
    "violinplot",
    "vlines",
}
_SEABORN = {
    "barplot",
    "boxenplot",
    "boxplot",
    "catplot",
    "countplot",
    "displot",
    "ecdfplot",
    "factorplot",
    "heatmap",
    "histplot",
    "jointplot",
    "kdeplot",
    "lineplot",
    "lmplot",
    "pairplot",
    "pointplot",
    "regplot",
    "relplot",
    "residplot",
    "rugplot",
    "scatterplot",
    "stripplot",
    "swarmplot",
    "violinplot",
}
_TICKER = {
    "AutoLocator",
    "AutoMinorLocator",
    "FixedFormatter",
    "FixedLocator",
    "FormatStrFormatter",
    "FuncFormatter",
    "LinearLocator",
    "LogFormatter",
    "LogFormatterMathtext",
    "LogLocator",
    "MaxNLocator",
    "MultipleLocator",
    "NullFormatter",
    "NullLocator",
    "PercentFormatter",
    "ScalarFormatter",
    "StrMethodFormatter",
}
_DATES = {
    "AutoDateFormatter",
    "AutoDateLocator",
    "ConciseDateFormatter",
    "DateFormatter",
    "DayLocator",
    "HourLocator",
    "MicrosecondLocator",
    "MinuteLocator",
    "MonthLocator",
    "RRuleLocator",
    "SecondLocator",
    "WeekdayLocator",
    "YearLocator",
    "date2num",
    "num2date",
}
_FIGURE = {
    "add_axes",
    "add_gridspec",
    "add_subplot",
    "align_labels",
    "autofmt_xdate",
    "colorbar",
    "delaxes",
    "legend",
    "subplots",
    "suptitle",
    "supxlabel",
    "supylabel",
    "text",
    "tight_layout",
}
_AXES = {
    "acorr",
    "add_patch",
    "annotate",
    "arrow",
    "axhline",
    "axhspan",
    "axline",
    "axvline",
    "axvspan",
    "bar",
    "bar_label",
    "barbs",
    "barh",
    "boxplot",
    "broken_barh",
    "clabel",
    "contour",
    "contourf",
    "errorbar",
    "eventplot",
    "fill",
    "fill_between",
    "fill_betweenx",
    "get_figure",
    "get_legend",
    "get_legend_handles_labels",
    "grid",
    "hexbin",
    "hist",
    "hist2d",
    "hlines",
    "imshow",
    "inset_axes",
    "legend",
    "margins",
    "pcolor",
    "pcolormesh",
    "pie",
    "plot",
    "plot_date",
    "quiver",
    "scatter",
    "secondary_xaxis",
    "secondary_yaxis",
    "set",
    "set_aspect",
    "set_axisbelow",
    "set_axis_off",
    "set_axis_on",
    "set_box_aspect",
    "set_facecolor",
    "set_frame_on",
    "set_prop_cycle",
    "set_title",
    "set_xlabel",
    "set_xlim",
    "set_xscale",
    "set_xticks",
    "set_xticklabels",
    "set_ylabel",
    "set_ylim",
    "set_yscale",
    "set_yticks",
    "set_yticklabels",
    "sharex",
    "sharey",
    "specgram",
    "spy",
    "stackplot",
    "stem",
    "step",
    "streamplot",
    "table",
    "text",
    "tick_params",
    "tricontour",
    "tricontourf",
    "tripcolor",
    "triplot",
    "twinx",
    "twiny",
    "violinplot",
    "vlines",
    "xaxis_date",
    "yaxis_date",
}
_SAFE_AXES_SET_KEYWORDS = {
    "aspect",
    "box_aspect",
    "facecolor",
    "frame_on",
    "title",
    "xlabel",
    "xlim",
    "xscale",
    "ylabel",
    "ylim",
    "yscale",
}
_ARTIST = {
    "get_figure",
    "remove",
    "set_alpha",
    "set_color",
    "set_edgecolor",
    "set_facecolor",
    "set_label",
    "set_linestyle",
    "set_linewidth",
    "set_marker",
    "set_markersize",
    "set_visible",
}
_SEQUENCE_METHODS = {"count", "index"}
_AXIS = {
    "grid",
    "set_label",
    "set_label_position",
    "set_major_formatter",
    "set_major_locator",
    "set_minor_formatter",
    "set_minor_locator",
    "set_tick_params",
    "set_ticks",
    "set_ticks_position",
    "set_ticklabels",
}
_DANGEROUS_KEYWORDS = {
    "backend",
    "file",
    "filename",
    "filepath",
    "fname",
    "font",
    "fontproperties",
    "path",
    "picker",
    "url",
    "urls",
    "usetex",
}


def parse_response(response: str) -> tuple[str, str]:
    text = response.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    try:
        result = json.loads(text)
    except (ValueError, TypeError):
        raise GenerationError("Expected a JSON object with code and explanation strings.") from None
    if not isinstance(result, dict) or set(result) != {"status", "code", "explanation"}:
        raise GenerationError("Response does not match the Augplot response schema.")
    status, code, explanation = result["status"], result["code"], result["explanation"]
    if (
        status not in {"ok", "out_of_scope"}
        or not isinstance(code, str)
        or not isinstance(explanation, str)
        or not explanation.strip()
    ):
        raise GenerationError("Response has invalid code or explanation fields.")
    if status == "out_of_scope":
        if code.strip():
            raise GenerationError("Out-of-scope responses must have an empty code field.")
        raise ScopeError(explanation.strip()[:1000])
    if not code.strip():
        raise GenerationError("Successful responses must contain code.")
    if len(code) > MAX_SOURCE_CHARS:
        raise GenerationError("Generated code exceeds the source-size limit.")
    return code.strip() + "\n", explanation.strip()


def allowed_imports(backend: str) -> set[str]:
    result = {"numpy", "pandas", "matplotlib.pyplot", "matplotlib.ticker", "matplotlib.dates"}
    if backend in ("auto", "seaborn"):
        result.add("seaborn")
    return result


class _Validator:
    def __init__(self, code, backend):
        self.code = code
        self.backend = backend
        self.env = {"data": "data", "title": "value", "figsize": "value"}
        self.imports = set()
        self.figures = set()
        self.invalid_provenance = set()
        self.loops = 0
        self.loop_depth = 0

    def fail(self, rule, message):
        raise GenerationError(
            message,
            code=self.code,
            violations=[{"rule": rule, "message": message}],
        )

    def validate_keywords(self, node):
        for keyword in node.keywords:
            if keyword.arg is None:
                self.fail("star_args", "*args and **kwargs are not allowed.")
            if keyword.arg == "regex" and not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is False
            ):
                self.fail(
                    "resource_limit",
                    "Regex evaluation is outside the bounded plotting subset.",
                )
            if keyword.arg.casefold() in _DANGEROUS_KEYWORDS:
                self.fail(
                    "dangerous_keyword",
                    f"Keyword {keyword.arg!r} grants an external or active-content capability.",
                )

    def merge_branches(self, before, body, otherwise):
        merged = {}
        passive_kinds = {"value", "data", "bounded_sequence"}
        for name in before.keys() | body.keys() | otherwise.keys():
            body_kind = body.get(name)
            otherwise_kind = otherwise.get(name)
            if body_kind == otherwise_kind and body_kind is not None:
                merged[name] = body_kind
                self.invalid_provenance.discard(name)
            elif {body_kind, otherwise_kind} <= passive_kinds:
                # Keep ordinary values usable after a branch, without carrying a
                # bounded-iteration proof or a plotting-object capability across it.
                merged[name] = "data"
                self.invalid_provenance.discard(name)
            else:
                self.invalid_provenance.add(name)
        return merged

    def assigned_names(self, target):
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            return set().union(*(self.assigned_names(item) for item in target.elts))
        return set()

    def static_integer(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self.static_integer(node.operand)
            if value is None:
                return None
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Pow)
        ):
            left = self.static_integer(node.left)
            right = self.static_integer(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Pow) and (right < 0 or right > 16):
                self.fail("resource_limit", "Static exponent is outside the safe limit.")
            try:
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                return left**right
            except (ArithmeticError, OverflowError):
                self.fail("resource_limit", "Static arithmetic exceeds the safe limit.")
        return None

    def literal_number(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        return None

    def bounded_count(self, node, *, default=None):
        if node is None:
            return default
        value = self.static_integer(node)
        if value is not None and 0 <= value <= MAX_BOUNDED_ITERATION:
            return value
        return None

    def bounded_head_or_tail(self, node):
        count_keywords = [keyword for keyword in node.keywords if keyword.arg == "n"]
        if (
            len(node.args) > 1
            or len(count_keywords) > 1
            or any(keyword.arg != "n" for keyword in node.keywords)
            or (node.args and count_keywords)
        ):
            return False
        count_node = count_keywords[0].value if count_keywords else None
        if count_node is None and node.args:
            count_node = node.args[0]
        return self.bounded_count(count_node, default=5) is not None

    def bounded_range(self, node):
        if not 1 <= len(node.args) <= 3 or node.keywords:
            return False
        values = [self.static_integer(argument) for argument in node.args]
        if any(value is None or abs(value) > MAX_STATIC_RANGE for value in values):
            return False
        try:
            return len(range(*values)) <= MAX_BOUNDED_ITERATION
        except (TypeError, ValueError, OverflowError):
            return False

    def bounded_slice(self, node):
        if not isinstance(node, ast.Slice) or node.upper is None:
            return False
        lower = 0 if node.lower is None else self.static_integer(node.lower)
        upper = self.static_integer(node.upper)
        step = 1 if node.step is None else self.static_integer(node.step)
        return (
            lower is not None
            and upper is not None
            and step is not None
            and 0 <= lower <= upper
            and 0 < step
            and upper - lower <= MAX_BOUNDED_ITERATION
        )

    def bounded_arange(self, node):
        if len(node.args) != 1 or node.keywords:
            return False
        stop = node.args[0]
        return (
            isinstance(stop, ast.Call)
            and isinstance(stop.func, ast.Name)
            and stop.func.id == "len"
            and len(stop.args) == 1
            and not stop.keywords
            and isinstance(stop.args[0], ast.Name)
            and self.env.get(stop.args[0].id)
            in {"bounded_column", "bounded_data", "bounded_sequence"}
        )

    def data_extent(self, node):
        data_kinds = {"bounded_column", "bounded_data", "bounded_sequence", "data", "sequence"}
        if isinstance(node, ast.Name):
            return self.env.get(node.id) == "data_extent"
        if isinstance(node, ast.Call):
            return (
                isinstance(node.func, ast.Name)
                and node.func.id == "len"
                and len(node.args) == 1
                and not node.keywords
                and isinstance(node.args[0], ast.Name)
                and self.env.get(node.args[0].id) in data_kinds
            )
        if isinstance(node, ast.Attribute):
            return (
                node.attr == "size"
                and isinstance(node.value, ast.Name)
                and self.env.get(node.value.id) in data_kinds
            )
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            return (
                node.value.attr == "shape"
                and isinstance(node.value.value, ast.Name)
                and self.env.get(node.value.value.id) in data_kinds
                and isinstance(node.slice, ast.Constant)
                and node.slice.value in {0, 1}
            )
        return False

    def data_sized_stop(self, node):
        if self.data_extent(node):
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return (
                self.data_extent(node.left)
                and isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, int)
                and abs(node.right.value) <= 1
            )
        return False

    def data_sized_range(self, node):
        if node.keywords or len(node.args) not in {1, 2}:
            return False
        if len(node.args) == 1:
            return self.data_sized_stop(node.args[0])
        start = self.static_integer(node.args[0])
        return start in {0, 1} and self.data_sized_stop(node.args[1])

    def subplot_result(self, node):
        rows = self.keyword_value(node, "nrows")
        columns = self.keyword_value(node, "ncols")
        if rows is None and node.args:
            rows = node.args[0]
        if columns is None and len(node.args) > 1:
            columns = node.args[1]
        rows = 1 if rows is None else self.static_integer(rows)
        columns = 1 if columns is None else self.static_integer(columns)
        squeeze = self.keyword_value(node, "squeeze")
        is_flat_axes = (
            isinstance(rows, int)
            and isinstance(columns, int)
            and 0 < rows * columns <= 100
            and rows * columns > 1
            and (rows == 1 or columns == 1)
            and (
                squeeze is None
                or (isinstance(squeeze, ast.Constant) and squeeze.value is True)
            )
        )
        return "axes_sequence" if is_flat_axes else "axes"

    def keyword_value(self, node, name):
        return next((item.value for item in node.keywords if item.arg == name), None)

    def validate_resources(self, base, attr, node):
        if base in {"module:matplotlib.pyplot", "figure"} and attr == "subplots":
            rows_node = self.keyword_value(node, "nrows")
            columns_node = self.keyword_value(node, "ncols")
            if rows_node is None and node.args:
                rows_node = node.args[0]
            if columns_node is None and len(node.args) > 1:
                columns_node = node.args[1]
            rows = self.literal_number(rows_node) if rows_node is not None else 1
            columns = self.literal_number(columns_node) if columns_node is not None else 1
            if rows is not None and columns is not None and rows * columns > 100:
                self.fail("resource_limit", "Plot creates too many Axes.")

        if base == "module:matplotlib.pyplot" and attr in {"figure", "subplots"}:
            figsize = self.keyword_value(node, "figsize")
            if isinstance(figsize, (ast.Tuple, ast.List)) and len(figsize.elts) == 2:
                dimensions = [self.literal_number(item) for item in figsize.elts]
                if all(value is not None for value in dimensions) and (
                    max(dimensions) > 100 or dimensions[0] * dimensions[1] > 10_000
                ):
                    self.fail("resource_limit", "Figure dimensions exceed the safe limit.")

        if base in {"axes", "module:matplotlib.pyplot"} and attr in {"hist", "hist2d"}:
            bins_node = self.keyword_value(node, "bins")
            if bins_node is None and len(node.args) > 1:
                bins_node = node.args[1]
            bins = self.literal_number(bins_node) if bins_node is not None else None
            if bins is not None and bins > 10_000:
                self.fail("resource_limit", "Histogram bin count exceeds the safe limit.")

    def validate(self, tree):
        if len(list(ast.walk(tree))) > MAX_AST_NODES:
            self.fail("ast_size", "Generated source is too complex.")
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            self.fail(
                "module_shape", "Source must contain exactly one function, with imports inside it."
            )
        fn = tree.body[0]
        a = fn.args
        if fn.name != "plot_data" or fn.decorator_list or fn.returns:
            self.fail("function_contract", "Expected undecorated plot_data with no annotations.")
        if (
            [x.arg for x in a.args] != ["data"]
            or a.posonlyargs
            or a.defaults
            or a.vararg
            or a.kwarg
            or [x.arg for x in a.kwonlyargs] != ["title", "figsize"]
            or any(not isinstance(x, ast.Constant) or x.value is not None for x in a.kw_defaults)
            or any(x.annotation for x in a.args + a.kwonlyargs)
        ):
            self.fail(
                "function_signature",
                "Use the signature plot_data(data, *, title=None, figsize=None).",
            )
        if not fn.body or not isinstance(fn.body[-1], ast.Return):
            self.fail("return", "plot_data must end by returning its approved Figure.")
        for stmt in fn.body:
            self.stmt(stmt)
        if not isinstance(fn.body[-1].value, ast.Name) or fn.body[-1].value.id not in self.figures:
            self.fail(
                "return", "plot_data must return a Figure created by an approved plotting call."
            )

    def stmt(self, n):
        if isinstance(n, ast.Import):
            for x in n.names:
                if x.name not in allowed_imports(self.backend) or x.asname != _IMPORT_ALIASES.get(
                    x.name
                ):
                    self.fail("import", "Import is not in the approved manifest.")
                if x.asname in self.env:
                    self.fail(
                        "alias_shadowing", "Imported aliases cannot be reassigned or shadowed."
                    )
                self.env[x.asname] = "module:" + x.name
                self.imports.add(x.asname)
        elif isinstance(n, ast.Assign):
            if len(n.targets) != 1:
                self.fail("assignment", "Only one assignment target is allowed.")
            self.assign(n.targets[0], self.expr(n.value))
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
            self.fail("assignment", "Annotated and augmented assignments are not allowed.")
        elif isinstance(n, ast.Expr):
            self.expr(n.value)
        elif isinstance(n, ast.Return):
            self.expr(n.value)
        elif isinstance(n, ast.Raise):
            if (
                not isinstance(n.exc, ast.Call)
                or not isinstance(n.exc.func, ast.Name)
                or n.exc.func.id not in {"ValueError", "TypeError", "KeyError", "IndexError"}
                or n.cause is not None
            ):
                self.fail("raise", "Only approved local validation errors may be raised.")
            self.expr(n.exc)
        elif isinstance(n, ast.If):
            self.expr(n.test)
            before = self.env.copy()
            before_figures = self.figures.copy()
            for s in n.body:
                self.stmt(s)
            body = self.env.copy()
            body_figures = self.figures.copy()
            self.env = before.copy()
            self.figures = before_figures.copy()
            for s in n.orelse:
                self.stmt(s)
            otherwise = self.env.copy()
            otherwise_figures = self.figures.copy()
            self.env = self.merge_branches(before, body, otherwise)
            self.figures = body_figures & otherwise_figures
        elif isinstance(n, ast.For):
            self.loops += 1
            if self.loops > MAX_LOOPS:
                self.fail("loop_limit", "Generated source contains too many loops.")
            kind = self.expr(n.iter)
            if kind not in {"axes_sequence", "bounded_column", "bounded_sequence"}:
                self.fail(
                    "resource_limit",
                    "Loops may iterate only over statically or explicitly bounded sequences.",
                )
            if self.loop_depth:
                self.fail("resource_limit", "Nested loops are not allowed.")
            target_kind = kind if kind == "axes_sequence" else "value"
            if kind == "axes_sequence" and isinstance(n.target, ast.Name):
                target_kind = "axes"
            self.assign(n.target, target_kind)
            self.loop_depth += 1
            try:
                for s in n.body + n.orelse:
                    self.stmt(s)
            finally:
                self.loop_depth -= 1
        elif isinstance(n, (ast.Pass, ast.Break, ast.Continue)):
            pass
        else:
            self.fail("statement", "Unsupported Python construct in generated source.")

    def assign(self, target, kind):
        if isinstance(target, ast.Name):
            if target.id.startswith("_") or target.id in self.imports or target.id == "data":
                self.fail(
                    "alias_shadowing", "Reserved names and imported aliases cannot be reassigned."
                )
            if target.id in self.figures and kind != "figure":
                self.fail("figure_identity", "An approved Figure name cannot be replaced.")
            self.env[target.id] = kind
            self.invalid_provenance.discard(target.id)
            if kind == "figure":
                self.figures.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            if kind not in {
                "figure_axes",
                "figure_axes_sequence",
                "bounded_sequence",
                "sequence",
                "axes",
                "axes_sequence",
                "value",
            }:
                self.fail("assignment", "This value cannot be unpacked.")
            for i, elt in enumerate(target.elts):
                if i == 0 and kind in {"figure_axes", "figure_axes_sequence"}:
                    item_kind = "figure"
                elif kind == "figure_axes_sequence":
                    item_kind = "axes_sequence"
                elif kind in {"figure_axes", "axes"} or (kind == "axes_sequence" and i == 0):
                    item_kind = "axes"
                else:
                    item_kind = "value"
                self.assign(elt, item_kind)
        elif isinstance(target, ast.Subscript):
            # A local, data-derived frame may be reshaped in memory; the caller's
            # original `data` argument and all attribute/module mutation remain blocked.
            if (
                isinstance(target.value, ast.Name)
                and target.value.id != "data"
                and self.expr(target.value) == "data"
            ):
                self.expr(target.slice)
                return
            self.fail(
                "mutation",
                f"Subscript assignment at generated line {target.lineno} is not allowed.",
            )
        elif isinstance(target, ast.Attribute):
            self.fail(
                "mutation",
                f"Assignment to .{target.attr} at generated line {target.lineno} is not "
                "allowed. Derive a separate local value instead.",
            )
        else:
            self.fail("mutation", "Attribute and subscript assignment are not allowed.")

    def expr(self, n):
        if isinstance(n, (ast.operator, ast.boolop, ast.unaryop, ast.cmpop, ast.expr_context)):
            return "value"
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (str, bytes)) and len(n.value) > MAX_LITERAL_ITEMS:
                self.fail("literal_size", "Generated source contains an oversized literal.")
            if isinstance(n.value, int) and abs(n.value) > MAX_STATIC_INTEGER:
                self.fail("resource_limit", "Integer literal exceeds the safe limit.")
            return "value"
        if isinstance(n, ast.Name):
            if n.id in _BUILTINS:
                return "builtin:" + n.id
            if n.id in self.invalid_provenance:
                self.fail(
                    "provenance",
                    f"Name {n.id!r} at generated line {n.lineno} does not have one "
                    "capability on every path.",
                )
            if n.id.startswith("_") or n.id not in self.env:
                self.fail("name", "Source references an unknown capability or name.")
            return self.env[n.id]
        if isinstance(n, (ast.List, ast.Tuple, ast.Set)):
            if len(n.elts) > MAX_LITERAL_ITEMS:
                self.fail("literal_size", "Generated source contains an oversized literal.")
            kinds = [self.expr(x) for x in n.elts]
            if kinds and all(kind == "axes" for kind in kinds):
                return "axes_sequence"
            if "axes" in kinds or "axes_sequence" in kinds:
                self.fail("provenance", "Axes collections may contain only approved Axes.")
            return "bounded_sequence" if len(n.elts) <= MAX_BOUNDED_ITERATION else "sequence"
        if isinstance(n, ast.Dict):
            if len(n.keys) > MAX_LITERAL_ITEMS or any(k is None for k in n.keys):
                self.fail(
                    "literal_size", "Generated source contains an unsupported dictionary literal."
                )
            for k, v in zip(n.keys, n.values, strict=True):
                self.expr(k)
                self.expr(v)
            return "value"
        if isinstance(n, ast.Subscript):
            base = self.expr(n.value)
            self.expr(n.slice)
            if self.data_extent(n):
                return "data_extent"
            if base == "spines":
                return "artist"
            if base in {"axes", "axes_sequence"}:
                return "axes"
            if base == "bounded_data" and isinstance(n.slice, ast.Constant):
                return "bounded_column"
            if base == "bounded_column" and isinstance(n.slice, ast.Slice):
                return "bounded_column"
            if base == "bounded_sequence" and isinstance(n.slice, ast.Slice):
                return "bounded_sequence"
            if base == "sequence" and self.bounded_slice(n.slice):
                return "bounded_sequence"
            if base in {"bounded_column", "bounded_data"}:
                return "bounded_data"
            if base == "bounded_sequence":
                return "value"
            return "data"
        if isinstance(n, ast.Slice):
            for x in (n.lower, n.upper, n.step):
                if x:
                    self.expr(x)
            return "value"
        if isinstance(n, ast.IfExp):
            self.expr(n.test)
            kinds = {self.expr(n.body), self.expr(n.orelse)}
            if kinds <= {"axes", "axes_sequence"}:
                return "axes_sequence"
            if len(kinds) == 1:
                return kinds.pop()
            if kinds <= {
                "bounded_column",
                "bounded_data",
                "bounded_sequence",
                "data",
                "value",
                "sequence",
            }:
                return "data"
            self.fail("provenance", "Conditional expression has incompatible capabilities.")
        if isinstance(n, (ast.BinOp, ast.BoolOp, ast.Compare, ast.UnaryOp)):
            for c in ast.iter_child_nodes(n):
                self.expr(c)
            static_value = self.static_integer(n)
            if static_value is not None and abs(static_value) > MAX_STATIC_INTEGER:
                self.fail("resource_limit", "Static arithmetic exceeds the safe limit.")
            return "data"
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            if len(n.generators) != 1:
                self.fail("resource_limit", "Nested comprehensions are not allowed.")
            before = self.env.copy()
            generator = n.generators[0]
            iterable_kind = self.expr(generator.iter)
            if generator.is_async or iterable_kind not in {
                "bounded_data",
                "bounded_column",
                "bounded_sequence",
                "data",
                "sequence",
                "axes_sequence",
            }:
                self.fail("resource_limit", "Comprehensions require an approved local sequence.")
            self.assign(generator.target, "value")
            for condition in generator.ifs:
                self.expr(condition)
            if isinstance(n, ast.DictComp):
                self.expr(n.key)
                self.expr(n.value)
            else:
                self.expr(n.elt)
            self.env = before
            self.invalid_provenance.update(self.assigned_names(generator.target))
            return (
                "bounded_sequence"
                if iterable_kind in {"bounded_column", "bounded_sequence", "axes_sequence"}
                else "sequence"
            )
        if isinstance(n, ast.JoinedStr):
            if (
                sum(
                    len(value.value)
                    for value in n.values
                    if isinstance(value, ast.Constant) and isinstance(value.value, str)
                )
                > MAX_LITERAL_ITEMS
            ):
                self.fail("literal_size", "Generated source contains an oversized literal.")
            for value in n.values:
                self.expr(value)
            return "value"
        if isinstance(n, ast.FormattedValue):
            if n.conversion != -1 or n.format_spec is not None:
                self.fail(
                    "format_string",
                    "Formatted labels cannot use conversions or format specifications.",
                )
            self.expr(n.value)
            return "value"
        if isinstance(n, ast.Call):
            return self.call(n)
        if isinstance(n, ast.Attribute):
            return self.attribute(n)
        if isinstance(n, ast.Starred):
            self.fail("star_args", "Starred arguments are not allowed.")
        self.fail("expression", "Unsupported Python expression in generated source.")

    def attribute(self, n):
        if n.attr.startswith("_"):
            self.fail("private_access", "Private and dunder attributes are not allowed.")
        base = self.expr(n.value)
        if base == "module:numpy" and n.attr == "nan":
            return "value"
        if (
            base
            in {
                "bounded_column",
                "bounded_data",
                "bounded_sequence",
                "data",
                "value",
                "sequence",
            }
            and n.attr in _DATA_PROPERTIES
        ):
            return (
                "bounded_column"
                if base == "bounded_column"
                else "bounded_data"
                if base in {"bounded_data", "bounded_sequence"}
                else "data"
            )
        if base == "axes" and n.attr in {"xaxis", "yaxis"}:
            return "axis"
        if base == "axes" and n.attr == "spines":
            return "spines"
        if base == "artist" and n.attr in {"figure", "fig"}:
            return "figure"
        self.fail(
            "attribute",
            f"Attribute .{n.attr} at generated line {n.lineno} is not an approved capability.",
        )

    def call(self, n):
        if any(isinstance(a, ast.Starred) for a in n.args):
            self.fail("star_args", "*args and **kwargs are not allowed.")
        if isinstance(n.func, ast.Name):
            self.validate_keywords(n)
            arg_kinds = [self.expr(x) for x in n.args]
            for keyword in n.keywords:
                self.expr(keyword.value)
            if n.func.id not in _BUILTINS:
                self.fail("call", "Calls must resolve to an approved capability.")
            if n.func.id == "range" and not self.data_sized_range(n) and (
                len(n.args) > 3
                or any(
                    not isinstance(x, ast.Constant)
                    or not isinstance(x.value, int)
                    or abs(x.value) > MAX_STATIC_RANGE
                    for x in n.args
                )
            ):
                self.fail("loop_bound", "range() must use small static integer bounds.")
            if n.func.id == "zip" and arg_kinds and arg_kinds[0] in {"axes", "axes_sequence"}:
                return "axes_sequence"
            if n.func.id == "range" and self.bounded_range(n):
                return "bounded_sequence"
            if n.func.id == "range" and self.data_sized_range(n):
                return "sequence"
            bounded_kinds = {"bounded_column", "bounded_sequence"}
            if n.func.id == "zip" and any(kind in bounded_kinds for kind in arg_kinds):
                return "bounded_sequence"
            if n.func.id in {
                "dict",
                "enumerate",
                "filter",
                "list",
                "map",
                "reversed",
                "set",
                "sorted",
                "tuple",
            } and any(kind in bounded_kinds for kind in arg_kinds):
                return "bounded_sequence"
            return (
                "sequence"
                if n.func.id
                in {
                    "list",
                    "tuple",
                    "set",
                    "dict",
                    "range",
                    "zip",
                    "map",
                    "filter",
                    "sorted",
                    "reversed",
                    "enumerate",
                }
                else "value"
            )
        if not isinstance(n.func, ast.Attribute):
            self.fail("call", "Indirect call targets are not allowed.")
        attr, base = n.func.attr, self.expr(n.func.value)
        if attr.startswith("_"):
            self.fail("private_access", "Private and dunder attributes are not allowed.")
        active_keywords = {
            keyword.arg.casefold()
            for keyword in n.keywords
            if keyword.arg is not None and keyword.arg.casefold() != "backend"
        } & _DANGEROUS_KEYWORDS
        if (
            base in {"bounded_column", "bounded_data", "data"}
            and attr in _DYNAMIC_BACKEND_METHODS
            and active_keywords
        ):
            self.validate_keywords(n)
        if base in {"bounded_column", "bounded_data", "data"} and attr in _DYNAMIC_BACKEND_METHODS:
            self.fail(
                "dynamic_backend",
                "Pandas plotting dispatch is not allowed; use Matplotlib or Seaborn directly.",
            )
        if base in {"bounded_column", "bounded_data", "data"} and attr in _DYNAMIC_DISPATCH_METHODS:
            self.fail(
                "dynamic_dispatch",
                "Pandas callable and string dispatch methods are not allowed.",
            )
        self.validate_keywords(n)
        for argument in n.args:
            self.expr(argument)
        for keyword in n.keywords:
            self.expr(keyword.value)
        self.validate_resources(base, attr, n)
        if base == "module:numpy" and attr in _RESOURCE_NUMPY:
            if attr == "arange" and self.bounded_arange(n):
                return "bounded_sequence"
            if attr == "arange" and self.data_sized_range(n):
                return "data"
            self.fail(
                "resource_limit",
                "This NumPy allocation or expansion API is outside the safe plotting subset.",
            )
        if base == "module:pandas" and attr in _RESOURCE_PANDAS:
            self.fail(
                "resource_limit",
                "This Pandas expansion API is outside the safe plotting subset.",
            )
        if base == "module:numpy" and attr in _NUMPY:
            return "data"
        if base == "module:pandas" and attr in _PANDAS:
            return "data"
        if base == "builtin:dict" and attr == "fromkeys":
            return "data"
        if base == "module:matplotlib.ticker" and attr in _TICKER:
            return "artist"
        if base == "module:matplotlib.dates" and attr in _DATES:
            return "artist"
        if base == "module:matplotlib.pyplot" and attr in _PYPLOT:
            if attr == "subplots":
                return (
                    "figure_axes_sequence"
                    if self.subplot_result(n) == "axes_sequence"
                    else "figure_axes"
                )
            if attr == "subplot_mosaic":
                return "figure_axes"
            return "figure" if attr == "figure" else "artist"
        if base == "module:seaborn" and attr in _SEABORN:
            return "artist"
        if base in {"bounded_column", "bounded_data", "data"} and attr in _DATA_METHODS:
            if attr == "iterrows":
                if base == "bounded_data" and not n.args and not n.keywords:
                    return "bounded_sequence"
                self.fail(
                    "resource_limit",
                    "iterrows() requires a DataFrame explicitly capped with head(N) or tail(N).",
                )
            if attr in {"head", "tail"} and self.bounded_head_or_tail(n):
                return "bounded_column" if base == "bounded_column" else "bounded_data"
            if base in {"bounded_column", "bounded_data"} and attr in {
                "to_list",
                "to_numpy",
                "tolist",
                "unique",
            }:
                return "bounded_sequence"
            if base in {"bounded_column", "bounded_data"} and attr in (
                _BOUND_PRESERVING_DATA_METHODS
            ):
                return base
            return "data"
        if base == "figure" and attr in _FIGURE:
            if attr == "subplots":
                return self.subplot_result(n)
            return "axes" if attr in {"add_axes", "add_subplot"} else "artist"
        if base == "value" and attr == "startswith" and len(n.args) == 1 and not n.keywords:
            if isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                return "value"
        if base == "axes" and attr == "set":
            names = {keyword.arg for keyword in n.keywords}
            if n.args or None in names or not names <= _SAFE_AXES_SET_KEYWORDS:
                self.fail(
                    "dangerous_keyword",
                    "Axes.set accepts only explicitly approved visual properties.",
                )
            return "artist"
        if base == "axes" and attr in _AXES:
            return (
                "axes"
                if attr
                in {
                    "inset_axes",
                    "secondary_xaxis",
                    "secondary_yaxis",
                    "sharex",
                    "sharey",
                    "twinx",
                    "twiny",
                }
                else "figure"
                if attr == "get_figure"
                else "sequence"
                if attr == "get_legend_handles_labels"
                else "artist"
            )
        if base == "artist" and attr in _ARTIST:
            return "figure" if attr == "get_figure" else "artist"
        if base == "axis" and attr in _AXIS:
            return "artist"
        if base in {"bounded_sequence", "sequence"} and attr in _SEQUENCE_METHODS:
            return "value"
        self.fail(
            "call",
            f"Call to .{attr}() at generated line {n.lineno} is not in the approved "
            "capability manifest.",
        )


def validate_code(code: str, backend: str) -> ast.Module:
    if not isinstance(code, str) or len(code) > MAX_SOURCE_CHARS:
        raise GenerationError(
            "Generated source exceeds the source-size limit.",
            code=code,
            violations=[{"rule": "source_size", "message": "Source exceeds size limit."}],
        )
    try:
        tree = ast.parse(code)
    except SyntaxError:
        raise GenerationError(
            "Generated source has invalid Python syntax.",
            code=code,
            violations=[{"rule": "syntax", "message": "Invalid Python syntax."}],
        ) from None
    _Validator(code, backend).validate(tree)
    return tree


def execute(code, data, *, backend, title=None, figsize=None):
    tree = validate_code(code, backend)
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    permitted = allowed_imports(backend)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        numpy_internal = name in {"numpy._core._methods", "numpy.core._methods"}
        # NumPy's ndarray reductions import this module lazily through the calling
        # function's builtins. The AST still forbids generated source from importing it.
        if level or name not in permitted and not numpy_internal:
            raise ImportError("Import is not permitted.")
        return builtins.__import__(name, globals, locals, fromlist, level)

    namespace = {"__builtins__": {name: getattr(builtins, name) for name in _BUILTINS}}
    namespace["__builtins__"]["__import__"] = guarded_import
    previous_figures = set(plt.get_fignums())
    try:
        with ExitStack() as stack:
            stack.enter_context(mpl.rc_context())
            if backend in ("auto", "seaborn"):
                import seaborn as sns

                stack.enter_context(sns.axes_style("whitegrid"))
                stack.enter_context(sns.plotting_context("notebook"))
            exec(compile(tree, "<augplot-generated>", "exec"), namespace)
            figure = namespace["plot_data"](copy_data(data), title=title, figsize=figsize)
            if not isinstance(figure, Figure) or not figure.axes:
                raise GenerationError(
                    "Function must return a nonempty Figure for the backend.", code=code
                )
            return figure
    except GenerationError:
        raise
    except Exception as exc:
        trace, line = exc.__traceback__, None
        while trace is not None:
            if trace.tb_frame.f_code.co_filename == "<augplot-generated>":
                line = trace.tb_lineno
            trace = trace.tb_next
        detail = "Scatter x and y must have the same number of values." if (
            isinstance(exc, ValueError) and str(exc) == "x and y must be the same size"
        ) else None
        raise GenerationError(
            f"Plot execution failed ({type(exc).__name__})"
            + (f" at generated line {line}." if line else ".")
            + (f" {detail}" if detail else ""),
            code=code,
        ) from None
    finally:
        for number in set(plt.get_fignums()) - previous_figures:
            plt.close(number)
