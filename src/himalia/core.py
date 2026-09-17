"""Public notebook workflow, independent of the inference transport."""

import importlib.util
import json
import os
from pathlib import Path

from . import provider
from .errors import ConfigurationError, GenerationError
from .execution import execute, parse_response
from .exporting import save_function
from .profiling import copy_data, profile_data, validate_data
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT


class Visualizer:
    """Generate, refine, and export a plot using a provider model and a plotting backend.

    Provider credentials are read by LiteLLM when a request is made. Generated Python
    executes locally after conservative checks; it is not sandboxed. Data samples are
    sent to the configured model. See the README for the supported data and trust model.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        backend: str = "auto",
        api_base: str | None = None,
        sample_rows: int = 5,
        max_profile_chars: int = 20_000,
        timeout: float = 60,
        max_repairs: int = 1,
    ):
        if backend not in {"auto", "matplotlib", "seaborn", "plotly"}:
            raise ConfigurationError("backend must be auto, matplotlib, seaborn, or plotly.")
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
        self.api_base = api_base
        self.sample_rows = sample_rows
        self.max_profile_chars = max_profile_chars
        self.timeout = timeout
        self.max_repairs = max_repairs
        self.code: str | None = None
        self.figure = None
        self.explanation: str | None = None
        self.profile: dict | None = None
        self._data = None
        self._prompt = "auto"

    def _configuration(self):
        model = self.model if self.model is not None else os.getenv("HIMALIA_MODEL")
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationError("Set HIMALIA_MODEL or pass model='provider/model-name'.")
        if self.backend == "plotly" and (
            importlib.util.find_spec("plotly") is None
            or importlib.util.find_spec("nbformat") is None
        ):
            raise ConfigurationError("Install Plotly support with pip install 'himalia[plotly]'.")
        api_base = self.api_base if self.api_base is not None else os.getenv("HIMALIA_API_BASE")
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

    @staticmethod
    def _display(figure):
        from IPython import get_ipython
        from IPython.display import display

        # Do not open browser windows or print Figure reprs from scripts.
        shell = get_ipython()
        if shell is not None and getattr(shell, "kernel", None) is not None:
            display(figure)

    def fit(self, data, prompt: str = "auto", *, show: bool = True):
        """Generate and display a plot. Replace existing state only after success."""
        self._validate_prompt(prompt)
        self._configuration()
        profile = profile_data(data, sample_rows=self.sample_rows, max_chars=self.max_profile_chars)
        snapshot = copy_data(data)
        code, explanation, figure = self._generate(snapshot, profile, prompt)
        self._data, self._prompt = snapshot, prompt
        self.profile, self.code, self.explanation, self.figure = profile, code, explanation, figure
        if show:
            self._display(figure)
        return self

    def _require_fit(self):
        if self.code is None:
            raise ConfigurationError("Call fit(data) before refining, rendering, or saving.")

    def refine(self, prompt: str, *, show: bool = True):
        """Revise the last successful function using the original data and a new instruction."""
        self._require_fit()
        self._validate_prompt(prompt)
        code, explanation, figure = self._generate(
            self._data,
            self.profile,
            prompt,
            previous_code=self.code,
            original_prompt=self._prompt,
        )
        self.code, self.explanation, self.figure = code, explanation, figure
        if show:
            self._display(figure)
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

    def save(
        self, path: str | Path = "vis_utils.py", *, function_name: str = "plot_visualization"
    ) -> Path:
        """Export the current function and print a notebook usage example, without an LLM call."""
        self._require_fit()
        return save_function(self.code, path, function_name, backend=self.backend)

    def __repr__(self):
        state = "fitted" if self.code is not None else "unfitted"
        return f"Visualizer(backend={self.backend!r}, state={state!r})"


def plot(data, prompt: str = "auto", *, show: bool = True, **kwargs) -> Visualizer:
    """Fit and return a Visualizer. Keyword arguments configure Visualizer()."""
    return Visualizer(**kwargs).fit(data, prompt=prompt, show=show)
