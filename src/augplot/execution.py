"""Security checks for generated plotting source and local execution (not an OS sandbox)."""

import ast
import builtins
import json
from contextlib import ExitStack

from .errors import GenerationError, ScopeError
from .profiling import copy_data

VALIDATOR_VERSION = 9
MAX_SOURCE_CHARS = 50_000
MAX_AST_NODES = 4_000
MAX_LITERAL_ITEMS = 2_000
MAX_STATIC_INTEGER = 10_000_000
MAX_STATIC_RANGE = 10_000
MAX_LOOPS = 10
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

# Block effects and escape routes, not chart types or data transformations. The
# list applies to *every* receiver, including values returned by library calls.
_BLOCKED_ATTRIBUTES = {
    "DataSource",
    "ExcelFile",
    "ExcelWriter",
    "HDFStore",
    "applymap",
    "backend",
    "callbacks",
    "canvas",
    "close",
    "ctypeslib",
    "draw_all",
    "dump",
    "dumps",
    "eval",
    "exec",
    "fromfile",
    "fromregex",
    "genfromtxt",
    "get_backend",
    "get_config",
    "get_current_fig_manager",
    "get_data_home",
    "get_dataset_names",
    "getenv",
    "ginput",
    "imread",
    "imsave",
    "io",
    "ion",
    "ioff",
    "manager",
    "memmap",
    "new_figure_manager",
    "option_context",
    "options",
    "os",
    "pause",
    "popen",
    "query",
    "rc",
    "rcdefaults",
    "rcParams",
    "recfromcsv",
    "recfromtxt",
    "request",
    "requests",
    "reset_option",
    "set_backend",
    "set_config",
    "set_option",
    "set_picker",
    "set_url",
    "set_urls",
    "setp",
    "show",
    "socket",
    "style",
    "subplot_tool",
    "subprocess",
    "switch_backend",
    "system",
    "tofile",
    "urlopen",
    "waitforbuttonpress",
}
_BLOCKED_PREFIXES = ("read", "save", "load", "open", "write", "print_")


_BLOCKED_KEYWORDS = {
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
_MODULE_VALUES = {
    "DataFrame",
    "Index",
    "Series",
    "Timestamp",
    "ndarray",
    "nan",
    "inf",
    "pi",
    "e",
    "NA",
    "NaT",
    "mean",
    "sum",
    "median",
    "std",
    "var",
    "min",
    "max",
    "quantile",
    "percentile",
    "nanmean",
    "nanmedian",
    "nansum",
}
_DYNAMIC_METHODS = {"apply", "agg", "aggregate", "map", "transform"}
_RESOURCE_ALLOCATORS = {
    "arange",
    "linspace",
    "logspace",
    "geomspace",
    "zeros",
    "ones",
    "empty",
    "full",
    "repeat",
    "histogram",
    "histogram2d",
    "histogramdd",
    "tile",
    "broadcast_to",
    "meshgrid",
    "eye",
    "identity",
}
_SAFE_TO_METHODS = {
    "to_array",
    "to_datetime",
    "to_dict",
    "to_frame",
    "to_list",
    "to_numpy",
    "to_numeric",
    "to_period",
    "to_pydatetime",
    "to_records",
    "to_series",
    "to_timedelta",
    "to_timestamp",
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


class _Validator(ast.NodeVisitor):
    """Small AST gate for effects, escape hatches, and obvious resource bombs.

    Library objects are deliberately not classified by plotting or data capability.
    There is no OS isolation, so this is a risk reduction layer rather than a proof
    that arbitrary generated Python is safe.
    """

    def __init__(self, code, backend):
        self.code = code
        self.backend = backend
        self.imports = set()
        self.loops = 0
        self.parents = {}

    def fail(self, rule, message):
        raise GenerationError(
            message, code=self.code, violations=[{"rule": rule, "message": message}]
        )

    def validate(self, tree):
        nodes = list(ast.walk(tree))
        if len(nodes) > MAX_AST_NODES:
            self.fail("ast_size", "Generated source is too complex.")
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            self.fail("module_shape", "Source must contain exactly one function.")
        fn = tree.body[0]
        args = fn.args
        if (
            fn.name != "plot_data"
            or fn.decorator_list
            or fn.returns
            or (
                [arg.arg for arg in args.args] != ["data"]
                or args.posonlyargs
                or args.defaults
                or args.vararg
                or args.kwarg
                or [arg.arg for arg in args.kwonlyargs] != ["title", "figsize"]
                or any(
                    not isinstance(value, ast.Constant) or value.value is not None
                    for value in args.kw_defaults
                )
                or any(arg.annotation for arg in args.args + args.kwonlyargs)
            )
        ):
            self.fail(
                "function_contract",
                "Use the required plot_data signature without decorators or annotations.",
            )
        if not fn.body or not isinstance(fn.body[-1], ast.Return):
            self.fail("return", "plot_data must end with a return statement.")
        self.imports = {
            alias.asname
            for node in nodes
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.asname
        }
        self.parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
        for node in nodes:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node is not fn
            ):
                self.fail("definition", "Nested or additional definitions are not allowed.")
            if isinstance(
                node,
                (
                    ast.ImportFrom,
                    ast.Lambda,
                    ast.With,
                    ast.AsyncWith,
                    ast.Try,
                    ast.TryStar,
                    ast.While,
                    ast.Global,
                    ast.Nonlocal,
                    ast.Delete,
                    ast.Yield,
                    ast.YieldFrom,
                    ast.Await,
                    ast.NamedExpr,
                    ast.Match,
                ),
            ):
                self.fail("syntax", f"{type(node).__name__} is not allowed in generated code.")
        self.visit(fn)

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name not in allowed_imports(self.backend) or (
                alias.asname != _IMPORT_ALIASES.get(alias.name)
            ):
                self.fail("import", "Import is not permitted.")

    def visit_Name(self, node):
        if node.id.startswith("_"):
            self.fail("private_access", "Private and dunder names are not allowed.")
        if isinstance(node.ctx, ast.Load) and node.id == "plot_data":
            self.fail("call", "Recursive calls are not allowed.")
        if isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in self.imports:
            self.fail("alias_shadowing", "Imported aliases cannot be reassigned.")

    def visit_Attribute(self, node):
        name = node.attr
        root = node.value
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(node.ctx, (ast.Store, ast.Del)) and (
            isinstance(root, ast.Name) and root.id in self.imports
        ):
            self.fail("mutation", "Imported modules cannot be modified.")
        if name.startswith("_"):
            self.fail("private_access", "Private and dunder attributes are not allowed.")
        if self._blocked_member(name):
            self.fail(
                "external_access", f"Attribute .{name} at generated line {node.lineno} is blocked."
            )
        # A module alias may call a public top-level API, but cannot be traversed
        # into internal modules or kept as a general-purpose capability.
        if isinstance(node.value, ast.Name) and node.value.id in self.imports:
            parent = self.parents.get(node)
            if not (isinstance(parent, ast.Call) and parent.func is node) and (
                name not in _MODULE_VALUES
            ):
                self.fail("module_access", "Module members must be called directly.")
        if name in _DYNAMIC_METHODS and not (
            isinstance(self.parents.get(node), ast.Call) and self.parents[node].func is node
        ):
            self.fail("dynamic_dispatch", "Dynamic methods must be called directly.")
        self.generic_visit(node)

    def visit_Call(self, node):
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            keyword.arg is None for keyword in node.keywords
        ):
            self.fail("star_args", "Starred call arguments are not allowed.")
        if isinstance(node.func, ast.Name):
            if node.func.id not in _BUILTINS:
                self.fail(
                    "call",
                    f"Indirect call {node.func.id}() at generated line {node.lineno} is blocked.",
                )
        elif not isinstance(node.func, ast.Attribute):
            self.fail("call", "Indirect calls are not allowed.")
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
            if name in {"plot", "hist"} and any(item.arg == "backend" for item in node.keywords):
                self.fail("dynamic_backend", "Dynamic plotting backends are not allowed.")
            if name in _DYNAMIC_METHODS:
                target = (
                    node.args[0]
                    if node.args
                    else next((item.value for item in node.keywords if item.arg == "func"), None)
                )
                if not self._safe_dispatch(target, mapping=name == "map"):
                    self.fail(
                        "dynamic_dispatch", "Dynamic dispatch requires a simple safe function."
                    )
            if (
                name in _RESOURCE_ALLOCATORS
                and isinstance(node.func.value, ast.Name)
                and (node.func.value.id in {"np", "pd"})
            ):
                for arg in node.args:
                    if self._large_static(arg) or self._large_data_multiplier(arg):
                        self.fail(
                            "resource_limit", "Array allocation exceeds the static size limit."
                        )
            if name in {"subplots", "figure", "hist", "hist2d"}:
                self._check_plot_size(node)
        for keyword in node.keywords:
            if keyword.arg.casefold() in _BLOCKED_KEYWORDS:
                self.fail(
                    "dangerous_keyword", f"Keyword {keyword.arg!r} can trigger an external effect."
                )
            if keyword.arg == "regex" and not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is False
            ):
                self.fail("resource_limit", "Regex evaluation is not allowed.")
        if isinstance(node.func, ast.Name) and node.func.id == "range":
            if any(
                (value := self._static_integer(arg)) is not None and abs(value) > MAX_STATIC_RANGE
                for arg in node.args
            ):
                self.fail("resource_limit", "range() exceeds the static size limit.")
        self.generic_visit(node)

    @staticmethod
    def _blocked_member(name):
        return (
            name in _BLOCKED_ATTRIBUTES
            or name.startswith(_BLOCKED_PREFIXES)
            or name.startswith("to_")
            and name not in _SAFE_TO_METHODS
        )

    def _safe_dispatch(self, node, *, mapping=False):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            name = node.value
            return mapping or not (
                name in _BLOCKED_ATTRIBUTES | _DYNAMIC_METHODS | {"plot", "hist", "pipe"}
                or name.startswith(("read_", "save_", "load_", "open_", "write_"))
                or name.startswith("to_")
                and name not in _SAFE_TO_METHODS
            )
        if isinstance(node, (ast.List, ast.Tuple)):
            return all(self._safe_dispatch(item, mapping=mapping) for item in node.elts)
        if isinstance(node, ast.Dict):
            return all(self._safe_dispatch(item, mapping=mapping) for item in node.values)
        if isinstance(node, ast.Name):
            return node.id in _BUILTINS or mapping and node.id not in self.imports | {"plot_data"}
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and (node.value.id == "np" and node.attr in _MODULE_VALUES)
        )

    def _static_integer(self, node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._static_integer(node.operand)
            return value if value is None or isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Pow)
        ):
            left, right = self._static_integer(node.left), self._static_integer(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Pow) and (right < 0 or right > 16):
                return MAX_STATIC_INTEGER + 1
            try:
                return {
                    ast.Add: lambda: left + right,
                    ast.Sub: lambda: left - right,
                    ast.Mult: lambda: left * right,
                    ast.Pow: lambda: left**right,
                }[type(node.op)]()
            except (ArithmeticError, OverflowError):
                return MAX_STATIC_INTEGER + 1
        return None

    def _large_static(self, node):
        if isinstance(node, (ast.List, ast.Tuple)):
            product = 1
            for item in node.elts:
                value = self._static_integer(item)
                if value is None:
                    return False
                product *= max(value, 0)
            return product > MAX_STATIC_INTEGER
        value = self._static_integer(node)
        return value is not None and abs(value) > MAX_STATIC_INTEGER

    def _literal_number(self, node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._literal_number(node.operand)
            return value if value is None or isinstance(node.op, ast.UAdd) else -value
        return None

    def _large_data_multiplier(self, node):
        return (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mult)
            and any(
                self._static_integer(part) is not None
                and abs(self._static_integer(part)) > MAX_STATIC_RANGE
                for part in (node.left, node.right)
            )
        )

    def _check_plot_size(self, node):
        name = node.func.attr
        keywords = {item.arg: item.value for item in node.keywords}
        if name == "subplots":
            rows = self._static_integer(keywords.get("nrows", node.args[0] if node.args else None))
            cols = self._static_integer(
                keywords.get("ncols", node.args[1] if len(node.args) > 1 else None)
            )
            if (rows or 1) * (cols or 1) > 100:
                self.fail("resource_limit", "Plot creates too many Axes.")
        if name in {"subplots", "figure"}:
            size = keywords.get("figsize")
            if isinstance(size, (ast.Tuple, ast.List)) and len(size.elts) == 2:
                dims = [self._literal_number(item) for item in size.elts]
                if all(value is not None for value in dims) and (
                    max(dims) > 100 or dims[0] * dims[1] > 10_000
                ):
                    self.fail("resource_limit", "Figure dimensions exceed the safe limit.")
        if name in {"hist", "hist2d"}:
            bins = self._static_integer(keywords.get("bins"))
            if bins is not None and bins > 10_000:
                self.fail("resource_limit", "Histogram bin count exceeds the safe limit.")

    def visit_For(self, node):
        self.loops += 1
        if self.loops > MAX_LOOPS:
            self.fail("resource_limit", "Generated source contains too many loops.")
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, (str, bytes)) and len(node.value) > MAX_LITERAL_ITEMS:
            self.fail("literal_size", "Generated source contains an oversized literal.")
        if type(node.value) is int and abs(node.value) > MAX_STATIC_INTEGER:
            self.fail("resource_limit", "Integer literal exceeds the static size limit.")

    def visit_BinOp(self, node):
        if self._large_static(node) or self._large_data_multiplier(node):
            self.fail("resource_limit", "Static arithmetic exceeds the size limit.")
        self.generic_visit(node)

    def visit_JoinedStr(self, node):
        for value in node.values:
            if isinstance(value, ast.FormattedValue) and value.format_spec is not None:
                if not isinstance(value.format_spec, ast.JoinedStr) or any(
                    not isinstance(part, ast.Constant) or len(str(part.value)) > 40
                    for part in value.format_spec.values
                ):
                    self.fail("format_string", "Dynamic or oversized format specs are blocked.")
                spec = "".join(str(part.value) for part in value.format_spec.values)
                width = "".join(char for char in spec if char.isdigit())
                if width and int(width) > 1_000:
                    self.fail("format_string", "Format width exceeds the static size limit.")
        self.generic_visit(node)

    def visit_ListComp(self, node):
        self._check_comprehension(node)
        self.generic_visit(node)

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def _check_comprehension(self, node):
        count = 1
        for generator in node.generators:
            source = generator.iter
            if (
                isinstance(source, ast.Call)
                and isinstance(source.func, ast.Name)
                and (source.func.id == "range" and len(source.args) == 1)
            ):
                bound = self._static_integer(source.args[0])
                if bound is not None:
                    count *= max(bound, 0)
        if count > MAX_STATIC_INTEGER:
            self.fail("resource_limit", "Comprehension has too many static iterations.")

    def visit_List(self, node):
        if len(node.elts) > MAX_LITERAL_ITEMS:
            self.fail("literal_size", "Generated source contains an oversized literal.")
        self.generic_visit(node)

    visit_Tuple = visit_List
    visit_Set = visit_List

    def visit_Dict(self, node):
        if len(node.keys) > MAX_LITERAL_ITEMS or any(key is None for key in node.keys):
            self.fail("literal_size", "Generated source contains an oversized dictionary.")
        self.generic_visit(node)


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
        detail = (
            "Scatter x and y must have the same number of values."
            if (isinstance(exc, ValueError) and str(exc) == "x and y must be the same size")
            else None
        )
        raise GenerationError(
            f"Plot execution failed ({type(exc).__name__})"
            + (f" at generated line {line}." if line else ".")
            + (f" {detail}" if detail else ""),
            code=code,
        ) from None
    finally:
        for number in set(plt.get_fignums()) - previous_figures:
            plt.close(number)
