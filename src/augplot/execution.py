"""Conservative source checks and local execution. This is NOT a security sandbox."""

import ast
import builtins
import json
from contextlib import ExitStack

from .errors import GenerationError, ScopeError
from .profiling import copy_data

_BASE_IMPORTS = {"numpy", "pandas"}
_MPL_IMPORTS = {"matplotlib.pyplot", "matplotlib.ticker", "matplotlib.dates"}
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
    "Exception",
}
_FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "help",
    "breakpoint",
    "exit",
    "quit",
    "load",
    "loads",
    "save",
    "savez",
    "savez_compressed",
    "loadtxt",
    "genfromtxt",
    "fromfile",
    "tofile",
    "memmap",
    "read",
    "write",
    "system",
    "popen",
    "show",
    "display",
    "savefig",
    "close",
    "close_all",
    "use",
    "switch_backend",
    "set_backend",
    "rc",
    "rcdefaults",
    "rc_file",
    "set_theme",
    "set_style",
    "set_context",
    "set_palette",
    "set_config",
    "set_option",
    "reset_option",
    "print",
    "query",
    "test",
    "main",
    "get_data",
    "get_sample_data",
    "read_clipboard",
    "seterr",
    "seterrcall",
    "set_printoptions",
    "seed",
    "interactive",
    "ion",
    "ioff",
}
_FORBIDDEN_ATTRIBUTES = {"rcParams", "rcParamsDefault", "rcParamsOrig", "builtins"}


def parse_response(response: str) -> tuple[str, str]:
    text = response.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    try:
        result = json.loads(text)
    except (ValueError, TypeError):
        raise GenerationError("Expected a JSON object with code and explanation strings.") from None
    if not isinstance(result, dict) or set(result) != {"status", "code", "explanation"}:
        raise GenerationError("Response does not match the Augplot response schema.")
    status, code, explanation = result["status"], result["code"], result["explanation"]
    if status not in {"ok", "out_of_scope"}:
        raise GenerationError("Response has an invalid status.")
    if not isinstance(code, str) or not isinstance(explanation, str) or not explanation.strip():
        raise GenerationError("Response has invalid code or explanation fields.")
    if status == "out_of_scope":
        if code.strip():
            raise GenerationError("Out-of-scope responses must have an empty code field.")
        raise ScopeError(explanation.strip()[:1000])
    if not code.strip():
        raise GenerationError("Successful responses must contain code.")
    if len(code) > 50_000:
        raise GenerationError("Generated code exceeds the 50,000-character limit.")
    return code.strip() + "\n", explanation.strip()


def allowed_imports(backend):
    modules = _BASE_IMPORTS.copy()
    modules |= _MPL_IMPORTS
    if backend in ("auto", "seaborn"):
        modules.add("seaborn")
    return modules


def validate_code(code: str, backend: str) -> ast.Module:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        raise GenerationError("Generated source has invalid Python syntax.", code=code) from None

    def fail(message):
        raise GenerationError(message, code=code)

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        fail("Source must contain exactly one function, with imports inside it.")
    function = tree.body[0]
    if function.name != "plot_data" or function.decorator_list or function.returns:
        fail("Expected undecorated plot_data with no annotations.")
    args = function.args
    if (
        [a.arg for a in args.args] != ["data"]
        or args.posonlyargs
        or args.defaults
        or args.vararg
        or args.kwarg
        or [a.arg for a in args.kwonlyargs] != ["title", "figsize"]
        or any(not isinstance(d, ast.Constant) or d.value is not None for d in args.kw_defaults)
        or any(a.annotation for a in args.args + args.kwonlyargs)
    ):
        fail("Use the signature plot_data(data, *, title=None, figsize=None).")
    modules = allowed_imports(backend)
    forbidden_nodes = (
        ast.ClassDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.Global,
        ast.Nonlocal,
        ast.While,
        ast.With,
        ast.AsyncWith,
        ast.AsyncFor,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.Delete,
    )
    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            fail("Unsupported Python construct in generated source.")
        if isinstance(node, ast.FunctionDef) and node is not function:
            fail("Nested functions are not supported.")
        if isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in _FORBIDDEN_CALLS:
                fail("Source references a prohibited name.")
            if node.id == "plot_data":
                fail("Recursive/self-referencing plot functions are not supported.")
        if isinstance(node, ast.Attribute):
            name = node.attr
            if (
                name.startswith("_")
                or name in _FORBIDDEN_CALLS
                or name in _FORBIDDEN_ATTRIBUTES
                or name.startswith(("read_", "to_", "write_", "load_", "save_"))
                and name not in {"to_numpy", "to_list", "to_dict", "to_frame", "to_flat_index"}
            ):
                fail("Source references a prohibited attribute or I/O operation.")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in modules or not alias.asname or alias.asname.startswith("_"):
                    fail("Imports must use an allowed module and an explicit public alias.")
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in modules:
                fail("Import is not allowed for this backend.")
            for alias in node.names:
                if (
                    alias.name == "*"
                    or alias.name.startswith("_")
                    or alias.name in _FORBIDDEN_CALLS
                    or alias.name.startswith(("read_", "write_", "load_", "save_"))
                    or alias.asname
                    and alias.asname.startswith("_")
                ):
                    fail("Imported symbol is not allowed.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                fail("Source calls a prohibited operation.")
    return tree


def execute(code, data, *, backend, title=None, figsize=None):
    """Execute on a copy; isolate plot styles and close only newly created figures."""
    tree = validate_code(code, backend)
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    permitted = allowed_imports(backend)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in permitted:
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
            valid = isinstance(figure, Figure) and bool(figure.axes)
            if not valid:
                raise GenerationError(
                    "Function must return a nonempty Figure for the backend.", code=code
                )
            return figure
    except GenerationError:
        raise
    except Exception as exc:
        # Only exception *type* leaves the execution boundary, never the message/locals.
        trace = exc.__traceback__
        line = None
        while trace is not None:
            if trace.tb_frame.f_code.co_filename == "<augplot-generated>":
                line = trace.tb_lineno
            trace = trace.tb_next
        location = f" at generated line {line}" if line is not None else ""
        raise GenerationError(
            f"Plot execution failed ({type(exc).__name__}){location}.", code=code
        ) from None
    finally:
        # Detach figures from pyplot's automatic end-of-cell display, avoiding duplicates.
        for number in set(plt.get_fignums()) - previous_figures:
            plt.close(number)
