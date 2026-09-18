"""Public notebook workflow, independent of the inference transport."""

import importlib.util
import json
import os
from pathlib import Path

from . import provider
from .errors import ConfigurationError, GenerationError
from .execution import execute, parse_response
from .exporting import write_python_function
from .history import HISTORY_VERSION, History, fingerprint, request_key
from .profiling import copy_data, profile_data, validate_data
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT


class _Visualization:
    """Generate, refine, and reuse a plot with inspectable Python source.

    Provider credentials are read by LiteLLM when a request is made. Generated Python
    executes locally after conservative checks; it is not sandboxed. Data samples are
    sent to the configured model. See the README for the supported data and trust model.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        backend: str = "auto",
        display_format: str = "retina",
        api_base: str | None = None,
        sample_rows: int = 5,
        max_profile_chars: int = 20_000,
        timeout: float = 60,
        max_repairs: int = 1,
        cache_dir: str | Path | None = ".augplot/plots",
    ):
        if backend not in {"auto", "matplotlib", "seaborn", "plotly"}:
            raise ConfigurationError("backend must be auto, matplotlib, seaborn, or plotly.")
        if display_format not in {"retina", "png", "svg"}:
            raise ConfigurationError("display_format must be retina, png, or svg.")
        if not isinstance(sample_rows, int) or not 0 <= sample_rows <= 100:
            raise ConfigurationError("sample_rows must be an integer between 0 and 100.")
        if not isinstance(max_profile_chars, int) or not 500 <= max_profile_chars <= 100_000:
            raise ConfigurationError("max_profile_chars must be between 500 and 100,000.")
        if not isinstance(max_repairs, int) or max_repairs not in (0, 1):
            raise ConfigurationError("max_repairs must be 0 or 1.")
        if not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
            raise ConfigurationError("timeout must be a finite positive number of seconds.")
        self.model = model
        self.backend = backend
        self.display_format = display_format
        self.api_base = api_base
        self.sample_rows = sample_rows
        self.max_profile_chars = max_profile_chars
        self.timeout = timeout
        self.max_repairs = max_repairs
        self._history = History(cache_dir) if cache_dir is not None else None
        self.history_path: Path | None = None
        self.cache_hit = False
        self._record = None
        self._root = None
        self._data_fingerprint = None
        self.code: str | None = None
        self.figure = None
        self.explanation: str | None = None
        self.profile: dict | None = None
        self._data = None
        self._prompt = "auto"

    def _configuration(self):
        model = self.model if self.model is not None else os.getenv("AUGPLOT_MODEL")
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationError("Set AUGPLOT_MODEL or pass model='provider/model-name'.")
        if self.backend == "plotly" and (
            importlib.util.find_spec("plotly") is None
            or importlib.util.find_spec("nbformat") is None
        ):
            raise ConfigurationError("Install Plotly support with pip install 'augplot[plotly]'.")
        api_base = self.api_base if self.api_base is not None else os.getenv("AUGPLOT_API_BASE")
        return model, api_base

    @staticmethod
    def _validate_prompt(prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ConfigurationError("prompt must be 'auto' or a nonempty instruction string.")

    def _generate(self, data, profile, prompt, *, previous_code=None, original_prompt=None):
        model, api_base = self._configuration()
        context = {
            "prompt_version": PROMPT_VERSION,
            "backend": self.backend,
            "request": prompt,
            "data_profile": profile,
        }
        if previous_code is not None:
            context.update(
                previous_code=previous_code,
                original_request=original_prompt,
                task="Refine the existing function according to request.",
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context)},
        ]
        last_error = None
        code = None
        for attempt in range(self.max_repairs + 1):
            response = provider.complete(
                model=model, messages=messages, api_base=api_base, timeout=self.timeout
            )
            try:
                code, explanation = parse_response(response)
                figure = execute(code, data, backend=self.backend)
                return code, explanation, figure
            except GenerationError as exc:
                last_error = exc
                if attempt < self.max_repairs:
                    messages.extend(
                        [
                            {"role": "assistant", "content": response[:60_000]},
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "repair": "Return corrected JSON per the contract.",
                                        "diagnostic": str(exc)[:500],
                                    }
                                ),
                            },
                        ]
                    )
        raise GenerationError(
            f"Could not generate a working visualization after {self.max_repairs + 1} "
            f"attempt(s). {last_error}",
            code=code,
        ) from None

    def _resolve(self, data, profile, prompt, *, data_hash, regenerate, refining=False):
        """Resolve this exact step; descendants never replace their parent's lookup."""
        model, api_base = self._configuration()
        settings = {
            "history_version": HISTORY_VERSION,
            "prompt_version": PROMPT_VERSION,
            "data": data_hash,
            "model": model,
            "api_base": api_base,
            "backend": self.backend,
            "sample_rows": self.sample_rows,
            "max_profile_chars": self.max_profile_chars,
            "prompt": prompt,
        }
        if refining:
            settings.update(
                parent=self._record["revision"] if self._record else None,
                previous_code=self.code,
                original_prompt=self._prompt,
            )
        key = request_key(settings)
        root = self._root if refining else key
        if self._history is not None and not regenerate:
            saved = self._history.load(root, key)
            if saved is not None:
                record, code, path = saved
                # Apply the same validation and defensive execution as freshly generated code.
                code, explanation = parse_response(
                    json.dumps({"code": code, "explanation": record["explanation"]})
                )
                try:
                    figure = execute(code, data, backend=self.backend)
                except GenerationError as exc:
                    raise GenerationError(
                        "Saved visualization failed; no LLM request was made. Restore the "
                        "compatible environment or use regenerate=True explicitly. " + str(exc),
                        code=code,
                    ) from None
                return code, explanation, figure, record, path, root, True
        code, explanation, figure = self._generate(
            data,
            profile,
            prompt,
            previous_code=self.code if refining else None,
            original_prompt=self._prompt if refining else None,
        )
        record, path = None, None
        if self._history is not None:
            record, _, path = self._history.save(
                root,
                key,
                code=code,
                explanation=explanation,
                depth=self._record["depth"] + 1 if refining else 0,
                parent=self._record["revision"] if refining else None,
                data_fingerprint=data_hash,
            )
        return code, explanation, figure, record, path, root, False

    def _accept(self, resolved):
        (
            self.code, self.explanation, self.figure, self._record,
            self.history_path, self._root, self.cache_hit,
        ) = resolved

    def _display(self, figure):
        """Render static figures explicitly, without altering notebook formatters or DPI."""
        from IPython import get_ipython
        from IPython.core.pylabtools import print_figure, retina_figure
        from IPython.display import SVG, Image, display
        from matplotlib.figure import Figure

        # Do not open browser windows or print Figure reprs from scripts.
        shell = get_ipython()
        if shell is not None and getattr(shell, "kernel", None) is not None:
            if not isinstance(figure, Figure):
                display(figure)
            elif self.display_format == "retina":
                rendered = retina_figure(figure)
                if rendered is not None:
                    png, dimensions = rendered
                    display(Image(data=png, format="png", **dimensions))
            elif self.display_format == "svg":
                svg = print_figure(figure, fmt="svg")
                if svg is not None:
                    display(SVG(data=svg))
            else:
                png = print_figure(figure, fmt="png")
                if png is not None:
                    display(Image(data=png, format="png"))

    def fit(self, data, prompt: str = "auto", *, show: bool = True, regenerate: bool = False):
        """Reuse or generate the original plot. Replace existing state only after success."""
        self._validate_prompt(prompt)
        self._configuration()
        profile = profile_data(data, sample_rows=self.sample_rows, max_chars=self.max_profile_chars)
        snapshot = copy_data(data)
        data_hash = fingerprint(snapshot) if self._history is not None else None
        resolved = self._resolve(
            snapshot, profile, prompt, data_hash=data_hash, regenerate=regenerate
        )
        self._data, self._prompt = snapshot, prompt
        self.profile, self._data_fingerprint = profile, data_hash
        self._accept(resolved)
        if show:
            self._display(self.figure)
        return self

    def _require_fit(self):
        if self.code is None:
            raise ConfigurationError(
                "Call fit(data) before refining, rendering, or writing Python code."
            )

    @property
    def data_fingerprint(self):
        """Full fitted-data hash, or None before fit / when persistence is disabled."""
        return self._data_fingerprint

    def refine(self, prompt: str, *, show: bool = True, regenerate: bool = False):
        """Reuse or generate a revision identified by its parent and instruction."""
        self._require_fit()
        self._validate_prompt(prompt)
        resolved = self._resolve(
            self._data,
            self.profile,
            prompt,
            data_hash=self._data_fingerprint,
            regenerate=regenerate,
            refining=True,
        )
        self._accept(resolved)
        if show:
            self._display(self.figure)
        return self

    def render(self, data=None, *, title=None, figsize=None, show: bool = True):
        """Run the saved function locally on matching data; return self, with no LLM call.

        The fitted data remains the basis for future refinement. Pass new data to fit()
        if the schema or the data used for refinement should change.
        """
        self._require_fit()
        selected = self._data if data is None else data
        validate_data(selected)
        figure = execute(self.code, selected, backend=self.backend, title=title, figsize=figsize)
        self.figure = figure
        if show:
            self._display(figure)
        return self

    def to_python(
        self, path: str | Path | None = None, *, function_name: str = "plot_visualization"
    ) -> Path:
        """Write reusable Python to a local module, without an LLM call."""
        self._require_fit()
        target = Path("augplot_utils.py") if path is None else Path(path)
        return write_python_function(self.code, target, function_name, backend=self.backend)

    def __repr__(self):
        state = "fitted" if self.code is not None else "unfitted"
        return f"Augplot(backend={self.backend!r}, state={state!r})"


def plot(
    data, prompt: str = "auto", *, show: bool = True, regenerate: bool = False, **kwargs
) -> _Visualization:
    """Create a visualization that can be refined, rendered, and written as Python."""
    return _Visualization(**kwargs).fit(data, prompt=prompt, show=show, regenerate=regenerate)
