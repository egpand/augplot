"""Restricted generated-source validation and local execution (not an OS sandbox)."""
import ast
import builtins
import json
from contextlib import ExitStack

from .errors import GenerationError, ScopeError
from .profiling import copy_data

API_MANIFEST_VERSION = 1
MAX_SOURCE_CHARS, MAX_AST_NODES, MAX_LITERAL_ITEMS, MAX_STATIC_RANGE, MAX_LOOPS = 50_000, 4_000, 2_000, 10_000, 100
_IMPORT_ALIASES = {"numpy": "np", "pandas": "pd", "matplotlib.pyplot": "plt", "matplotlib.ticker": "ticker", "matplotlib.dates": "dates", "seaborn": "sns"}
_BUILTINS = {"abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "issubclass", "len", "list", "map", "max", "min", "next", "range", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip", "ValueError", "TypeError", "KeyError", "IndexError"}

# Default-deny, versioned capability manifest. Deliberately absent: IO, serialization,
# environment, subprocess, configuration/backend APIs, dynamic execution and reflection.
_NUMPY = {"abs", "all", "any", "arange", "argmax", "argmin", "argsort", "array", "asarray", "average", "ceil", "clip", "concatenate", "corrcoef", "count_nonzero", "cumsum", "diff", "digitize", "divide", "exp", "floor", "histogram", "isfinite", "isinf", "isnan", "linspace", "log", "log10", "max", "mean", "median", "min", "nanmax", "nanmean", "nanmedian", "nanmin", "nanpercentile", "nanstd", "nansum", "percentile", "quantile", "ravel", "repeat", "reshape", "round", "sort", "sqrt", "std", "sum", "unique", "var", "where", "zeros", "ones", "full", "vstack", "hstack", "column_stack"}
_PANDAS = {"DataFrame", "Series", "concat", "crosstab", "cut", "qcut", "get_dummies", "infer_freq", "isna", "notna", "to_datetime", "to_numeric", "pivot_table"}
_DATA_METHODS = {"abs", "agg", "aggregate", "all", "any", "append", "apply", "astype", "between", "clip", "combine_first", "copy", "corr", "count", "cov", "cummax", "cummin", "cumprod", "cumsum", "describe", "diff", "drop", "drop_duplicates", "dropna", "duplicated", "eq", "explode", "extend", "ffill", "fillna", "filter", "first_valid_index", "ge", "groupby", "gt", "head", "hist", "idxmax", "idxmin", "infer_objects", "interpolate", "isin", "isna", "items", "keys", "kurt", "last", "le", "lt", "map", "max", "mean", "median", "melt", "min", "mode", "nsmallest", "nlargest", "notna", "nunique", "pct_change", "pivot", "plot", "quantile", "rank", "reindex", "rename", "replace", "reset_index", "resample", "reverse", "rolling", "round", "sample", "select_dtypes", "sem", "set_index", "shift", "skew", "sort", "sort_index", "sort_values", "squeeze", "std", "sum", "tail", "transform", "transpose", "truediv", "unique", "unstack", "value_counts", "var", "where", "xs", "to_numpy", "to_list", "to_dict", "to_frame", "tolist"}
_DATA_PROPERTIES = {"T", "at", "columns", "dtypes", "empty", "iat", "iloc", "index", "loc", "name", "ndim", "shape", "size", "str", "values"}
_PYPLOT = {"Rectangle", "bar", "barh", "boxplot", "contour", "contourf", "errorbar", "eventplot", "figure", "fill", "fill_between", "hexbin", "hist", "hlines", "imshow", "pie", "plot", "scatter", "stackplot", "stem", "step", "subplots", "subplot_mosaic", "violinplot", "vlines"}
_SEABORN = {"barplot", "boxenplot", "boxplot", "catplot", "countplot", "displot", "ecdfplot", "factorplot", "heatmap", "histplot", "jointplot", "kdeplot", "lineplot", "lmplot", "pairplot", "pointplot", "regplot", "relplot", "residplot", "rugplot", "scatterplot", "stripplot", "swarmplot", "violinplot"}
_TICKER = {"AutoLocator", "AutoMinorLocator", "FixedFormatter", "FixedLocator", "FormatStrFormatter", "FuncFormatter", "LinearLocator", "LogFormatter", "LogFormatterMathtext", "LogLocator", "MaxNLocator", "MultipleLocator", "NullFormatter", "NullLocator", "PercentFormatter", "ScalarFormatter", "StrMethodFormatter"}
_DATES = {"AutoDateFormatter", "AutoDateLocator", "ConciseDateFormatter", "DateFormatter", "DayLocator", "HourLocator", "MicrosecondLocator", "MinuteLocator", "MonthLocator", "RRuleLocator", "SecondLocator", "WeekdayLocator", "YearLocator", "date2num", "num2date"}
_FIGURE = {"add_axes", "add_gridspec", "add_subplot", "align_labels", "autofmt_xdate", "colorbar", "delaxes", "legend", "subplots", "suptitle", "supxlabel", "supylabel", "tight_layout"}
_AXES = {"acorr", "add_patch", "annotate", "arrow", "axhline", "axhspan", "axline", "axvline", "axvspan", "bar", "bar_label", "barbs", "barh", "boxplot", "broken_barh", "clabel", "contour", "contourf", "errorbar", "eventplot", "fill", "fill_between", "fill_betweenx", "get_figure", "get_legend", "get_legend_handles_labels", "grid", "hexbin", "hist", "hist2d", "hlines", "imshow", "inset_axes", "legend", "margins", "pcolor", "pcolormesh", "pie", "plot", "plot_date", "quiver", "scatter", "secondary_xaxis", "secondary_yaxis", "set", "set_aspect", "set_axis_off", "set_axis_on", "set_box_aspect", "set_facecolor", "set_frame_on", "set_prop_cycle", "set_title", "set_xlabel", "set_xlim", "set_xscale", "set_xticks", "set_ylabel", "set_ylim", "set_yscale", "set_yticks", "sharex", "sharey", "specgram", "spy", "stackplot", "stem", "step", "streamplot", "table", "text", "tick_params", "tricontour", "tricontourf", "tripcolor", "triplot", "twinx", "twiny", "violinplot", "vlines", "xaxis_date", "yaxis_date"}
_ARTIST = {"get_figure", "remove", "set", "set_alpha", "set_color", "set_edgecolor", "set_facecolor", "set_label", "set_linestyle", "set_linewidth", "set_marker", "set_markersize", "set_visible"}
_SEQUENCE_METHODS = {"count", "index"}
_AXIS = {"grid", "set_label", "set_label_position", "set_major_formatter", "set_major_locator", "set_minor_formatter", "set_minor_locator", "set_tick_params", "set_ticks", "set_ticks_position", "set_ticklabels"}


def parse_response(response):
    text = response.strip()
    if text.startswith("```") and text.endswith("```"): text = "\n".join(text.splitlines()[1:-1])
    try: result = json.loads(text)
    except (ValueError, TypeError): raise GenerationError("Expected a JSON object with code and explanation strings.") from None
    if not isinstance(result, dict) or set(result) != {"status", "code", "explanation"}: raise GenerationError("Response does not match the Augplot response schema.")
    status, code, explanation = result["status"], result["code"], result["explanation"]
    if status not in {"ok", "out_of_scope"} or not isinstance(code, str) or not isinstance(explanation, str) or not explanation.strip(): raise GenerationError("Response has invalid code or explanation fields.")
    if status == "out_of_scope":
        if code.strip(): raise GenerationError("Out-of-scope responses must have an empty code field.")
        raise ScopeError(explanation.strip()[:1000])
    if not code.strip(): raise GenerationError("Successful responses must contain code.")
    if len(code) > MAX_SOURCE_CHARS: raise GenerationError("Generated code exceeds the source-size limit.")
    return code.strip() + "\n", explanation.strip()


def allowed_imports(backend):
    result = {"numpy", "pandas", "matplotlib.pyplot", "matplotlib.ticker", "matplotlib.dates"}
    if backend in ("auto", "seaborn"): result.add("seaborn")
    return result


class _Validator:
    def __init__(self, code, backend): self.code, self.backend, self.env, self.imports, self.figures, self.loops = code, backend, {"data": "data", "title": "value", "figsize": "value"}, set(), set(), 0
    def fail(self, rule, message): raise GenerationError(message, code=self.code, violations=[{"rule": rule, "message": message}])
    def validate(self, tree):
        if len(list(ast.walk(tree))) > MAX_AST_NODES: self.fail("ast_size", "Generated source is too complex.")
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef): self.fail("module_shape", "Source must contain exactly one function, with imports inside it.")
        fn = tree.body[0]; a = fn.args
        if fn.name != "plot_data" or fn.decorator_list or fn.returns: self.fail("function_contract", "Expected undecorated plot_data with no annotations.")
        if [x.arg for x in a.args] != ["data"] or a.posonlyargs or a.defaults or a.vararg or a.kwarg or [x.arg for x in a.kwonlyargs] != ["title", "figsize"] or any(not isinstance(x, ast.Constant) or x.value is not None for x in a.kw_defaults) or any(x.annotation for x in a.args + a.kwonlyargs): self.fail("function_signature", "Use the signature plot_data(data, *, title=None, figsize=None).")
        if not fn.body or not isinstance(fn.body[-1], ast.Return): self.fail("return", "plot_data must end by returning its approved Figure.")
        for stmt in fn.body: self.stmt(stmt)
        if not isinstance(fn.body[-1].value, ast.Name) or fn.body[-1].value.id not in self.figures: self.fail("return", "plot_data must return a Figure created by an approved plotting call.")
    def stmt(self, n):
        if isinstance(n, ast.Import):
            for x in n.names:
                if x.name not in allowed_imports(self.backend) or x.asname != _IMPORT_ALIASES.get(x.name): self.fail("import", "Import is not in the approved manifest.")
                if x.asname in self.env: self.fail("alias_shadowing", "Imported aliases cannot be reassigned or shadowed.")
                self.env[x.asname] = "module:" + x.name; self.imports.add(x.asname)
        elif isinstance(n, ast.Assign):
            if len(n.targets) != 1: self.fail("assignment", "Only one assignment target is allowed.")
            self.assign(n.targets[0], self.expr(n.value))
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign)): self.fail("assignment", "Annotated and augmented assignments are not allowed.")
        elif isinstance(n, ast.Expr): self.expr(n.value)
        elif isinstance(n, ast.Return): self.expr(n.value)
        elif isinstance(n, ast.Raise):
            if not isinstance(n.exc, ast.Call) or not isinstance(n.exc.func, ast.Name) or n.exc.func.id not in {"ValueError", "TypeError", "KeyError", "IndexError"} or n.cause is not None: self.fail("raise", "Only approved local validation errors may be raised.")
            self.expr(n.exc)
        elif isinstance(n, ast.If):
            self.expr(n.test); before = self.env.copy()
            for s in n.body: self.stmt(s)
            body = self.env.copy(); self.env = before.copy()
            for s in n.orelse: self.stmt(s)
            self.env = {k: body.get(k, self.env.get(k)) for k in set(body) | set(self.env)}
        elif isinstance(n, ast.For):
            self.loops += 1
            if self.loops > MAX_LOOPS: self.fail("loop_limit", "Generated source contains too many loops.")
            kind = self.expr(n.iter)
            if kind not in {"data", "value", "sequence", "axes", "axes_sequence"}: self.fail("loop", "Loops may iterate only over approved in-memory values.")
            self.assign(n.target, "axes_sequence" if kind == "axes_sequence" else "axes" if kind == "axes" else "value")
            for s in n.body + n.orelse: self.stmt(s)
        elif isinstance(n, (ast.Pass, ast.Break, ast.Continue)): pass
        else: self.fail("statement", "Unsupported Python construct in generated source.")
    def assign(self, target, kind):
        if isinstance(target, ast.Name):
            if target.id.startswith("_") or target.id in self.imports or target.id == "data": self.fail("alias_shadowing", "Reserved names and imported aliases cannot be reassigned.")
            if target.id in self.figures and kind != "figure": self.fail("figure_identity", "An approved Figure name cannot be replaced.")
            self.env[target.id] = kind
            if kind == "figure": self.figures.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            if kind not in {"figure_axes", "sequence", "axes", "axes_sequence", "value"}: self.fail("assignment", "This value cannot be unpacked.")
            for i, elt in enumerate(target.elts): self.assign(elt, "figure" if i == 0 and kind == "figure_axes" else "axes" if kind in {"figure_axes", "axes"} or (kind == "axes_sequence" and i == 0) else "value")
        elif isinstance(target, ast.Subscript):
            # A local, data-derived frame may be reshaped in memory; the caller's
            # original `data` argument and all attribute/module mutation remain blocked.
            if isinstance(target.value, ast.Name) and target.value.id != "data" and self.expr(target.value) == "data":
                self.expr(target.slice)
                return
            self.fail("mutation", "Attribute and subscript assignment are not allowed.")
        else: self.fail("mutation", "Attribute and subscript assignment are not allowed.")
    def expr(self, n):
        if isinstance(n, (ast.operator, ast.boolop, ast.unaryop, ast.cmpop, ast.expr_context)): return "value"
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (str, bytes)) and len(n.value) > MAX_LITERAL_ITEMS: self.fail("literal_size", "Generated source contains an oversized literal.")
            return "value"
        if isinstance(n, ast.Name):
            if n.id in _BUILTINS: return "builtin:" + n.id
            if n.id.startswith("_") or n.id not in self.env: self.fail("name", "Source references an unknown capability or name.")
            return self.env[n.id]
        if isinstance(n, (ast.List, ast.Tuple, ast.Set)):
            if len(n.elts) > MAX_LITERAL_ITEMS: self.fail("literal_size", "Generated source contains an oversized literal.")
            kinds = [self.expr(x) for x in n.elts]
            return "axes_sequence" if "axes" in kinds or "axes_sequence" in kinds else "sequence"
        if isinstance(n, ast.Dict):
            if len(n.keys) > MAX_LITERAL_ITEMS or any(k is None for k in n.keys): self.fail("literal_size", "Generated source contains an unsupported dictionary literal.")
            for k, v in zip(n.keys, n.values): self.expr(k); self.expr(v)
            return "value"
        if isinstance(n, ast.Subscript):
            base = self.expr(n.value); self.expr(n.slice)
            return "axes" if base in {"axes", "axes_sequence"} else "data"
        if isinstance(n, ast.Slice):
            for x in (n.lower, n.upper, n.step):
                if x: self.expr(x)
            return "value"
        if isinstance(n, ast.IfExp):
            self.expr(n.test)
            kinds = {self.expr(n.body), self.expr(n.orelse)}
            return "axes_sequence" if kinds & {"axes", "axes_sequence"} else "data"
        if isinstance(n, (ast.BinOp, ast.BoolOp, ast.Compare, ast.UnaryOp)):
            for c in ast.iter_child_nodes(n): self.expr(c)
            return "data"
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            for g in n.generators:
                if g.is_async or self.expr(g.iter) not in {"data", "value", "sequence"}: self.fail("comprehension", "Comprehensions may use only approved values.")
                self.assign(g.target, "value")
                for x in g.ifs: self.expr(x)
            for x in ast.iter_child_nodes(n):
                if not isinstance(x, ast.comprehension): self.expr(x)
            return "sequence"
        if isinstance(n, ast.Call): return self.call(n)
        if isinstance(n, ast.Attribute): return self.attribute(n)
        if isinstance(n, ast.Starred): self.fail("star_args", "Starred arguments are not allowed.")
        self.fail("expression", "Unsupported Python expression in generated source.")
    def attribute(self, n):
        if n.attr.startswith("_"): self.fail("private_access", "Private and dunder attributes are not allowed.")
        base = self.expr(n.value)
        if base in {"data", "value", "sequence"} and n.attr in _DATA_PROPERTIES: return "data"
        if base == "axes" and n.attr in {"xaxis", "yaxis"}: return "axis"
        if base == "artist" and n.attr in {"figure", "fig"}: return "figure"
        self.fail("attribute", "Attribute access is not an approved capability.")
    def call(self, n):
        if any(k.arg is None for k in n.keywords) or any(isinstance(a, ast.Starred) for a in n.args): self.fail("star_args", "*args and **kwargs are not allowed.")
        arg_kinds = [self.expr(x) for x in n.args]
        for x in n.keywords: self.expr(x.value)
        if isinstance(n.func, ast.Name):
            if n.func.id not in _BUILTINS: self.fail("call", "Calls must resolve to an approved capability.")
            if n.func.id == "range" and (len(n.args) > 3 or any(not isinstance(x, ast.Constant) or not isinstance(x.value, int) or abs(x.value) > MAX_STATIC_RANGE for x in n.args)): self.fail("loop_bound", "range() must use small static integer bounds.")
            if n.func.id == "zip" and arg_kinds and arg_kinds[0] in {"axes", "axes_sequence"}: return "axes_sequence"
            return "sequence" if n.func.id in {"list", "tuple", "set", "dict", "range", "zip", "map", "filter", "sorted", "reversed", "enumerate"} else "value"
        if not isinstance(n.func, ast.Attribute): self.fail("call", "Indirect call targets are not allowed.")
        attr, base = n.func.attr, self.expr(n.func.value)
        if attr.startswith("_"): self.fail("private_access", "Private and dunder attributes are not allowed.")
        if base == "module:numpy" and attr in _NUMPY: return "data"
        if base == "module:pandas" and attr in _PANDAS: return "data"
        if base == "builtin:dict" and attr == "fromkeys": return "data"
        if base == "module:matplotlib.ticker" and attr in _TICKER: return "artist"
        if base == "module:matplotlib.dates" and attr in _DATES: return "artist"
        if base == "module:matplotlib.pyplot" and attr in _PYPLOT: return "figure_axes" if attr in {"subplots", "subplot_mosaic"} else "figure" if attr == "figure" else "artist"
        if base == "module:seaborn" and attr in _SEABORN: return "artist"
        if base == "data" and attr in _DATA_METHODS: return "axes" if attr in {"plot", "hist"} else "data"
        if base == "figure" and attr in _FIGURE: return "axes" if attr in {"add_axes", "add_subplot", "subplots"} else "artist"
        if base == "axes" and attr in _AXES: return "axes" if attr in {"inset_axes", "secondary_xaxis", "secondary_yaxis", "sharex", "sharey", "twinx", "twiny"} else "figure" if attr == "get_figure" else "sequence" if attr == "get_legend_handles_labels" else "artist"
        if base == "artist" and attr in _ARTIST: return "figure" if attr == "get_figure" else "artist"
        if base == "axis" and attr in _AXIS: return "artist"
        if base == "sequence" and attr in _SEQUENCE_METHODS: return "value"
        self.fail("call", "Call target is not in the approved capability manifest.")


def validate_code(code, backend):
    if not isinstance(code, str) or len(code) > MAX_SOURCE_CHARS: raise GenerationError("Generated source exceeds the source-size limit.", code=code, violations=[{"rule": "source_size", "message": "Source exceeds size limit."}])
    try: tree = ast.parse(code)
    except SyntaxError: raise GenerationError("Generated source has invalid Python syntax.", code=code, violations=[{"rule": "syntax", "message": "Invalid Python syntax."}]) from None
    _Validator(code, backend).validate(tree)
    return tree


def execute(code, data, *, backend, title=None, figsize=None):
    tree = validate_code(code, backend)
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    permitted = allowed_imports(backend)
    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in permitted: raise ImportError("Import is not permitted.")
        return builtins.__import__(name, globals, locals, fromlist, level)
    namespace = {"__builtins__": {name: getattr(builtins, name) for name in _BUILTINS}}
    namespace["__builtins__"]["__import__"] = guarded_import
    previous_figures = set(plt.get_fignums())
    try:
        with ExitStack() as stack:
            stack.enter_context(mpl.rc_context())
            if backend in ("auto", "seaborn"):
                import seaborn as sns
                stack.enter_context(sns.axes_style("whitegrid")); stack.enter_context(sns.plotting_context("notebook"))
            exec(compile(tree, "<augplot-generated>", "exec"), namespace)
            figure = namespace["plot_data"](copy_data(data), title=title, figsize=figsize)
            if not isinstance(figure, Figure) or not figure.axes: raise GenerationError("Function must return a nonempty Figure for the backend.", code=code)
            return figure
    except GenerationError: raise
    except Exception as exc:
        trace, line = exc.__traceback__, None
        while trace is not None:
            if trace.tb_frame.f_code.co_filename == "<augplot-generated>": line = trace.tb_lineno
            trace = trace.tb_next
        raise GenerationError(f"Plot execution failed ({type(exc).__name__})" + (f" at generated line {line}." if line else "."), code=code) from None
    finally:
        for number in set(plt.get_fignums()) - previous_figures: plt.close(number)
