"""Narrow compatibility for colormap objects and bounded numeric formatting."""

import textwrap

import numpy as np
import pytest

from augplot import GenerationError
from augplot.execution import execute, validate_code


def _code(statements, *, pyplot=True):
    imports = "import numpy as np\n"
    if pyplot:
        imports += "import matplotlib.pyplot as plt\n"
    body = imports + textwrap.dedent(statements).strip() + "\nreturn data\n"
    return (
        "def plot_data(data, *, title=None, figsize=None):\n"
        + textwrap.indent(body, "    ")
    )


@pytest.mark.parametrize(
    "factory",
    [
        "plt.get_cmap('tab10')",
        "plt.cm.colors.LinearSegmentedColormap.from_list('custom', ['blue', 'red'])",
    ],
)
def test_single_binding_colormap_calls_execute_for_scalars_and_arrays(factory):
    code = _code(f"""
        palette = {factory}
        fig, ax = plt.subplots()
        ax.scatter([0, 1], [0, 1], color=palette(np.array([0.0, 1.0])))
        ax.set_facecolor(palette(0.5))
    """).replace("return data", "return fig")
    figure = execute(code, [1], backend="matplotlib")
    assert len(figure.axes[0].collections[0].get_facecolors()) == 2
    assert np.all(np.isfinite(figure.axes[0].get_facecolor()))


@pytest.mark.parametrize(
    "statements",
    [
        "callable_value = np.mean\ncallable_value(data)",
        "callable_value = data[0]\ncallable_value(data)",
        "palette = plt.get_cmap\npalette('tab10')",
        "palette = plt.get_cmap('tab10')\nalias = palette\nalias(0)",
        "palette = plt.get_cmap('tab10')\npalette = data[0]\npalette(0)",
        "palette = data[0]\npalette = plt.get_cmap('tab10')\npalette(0)",
        "palette = plt.get_cmap('tab10')\nfor palette in data:\n    palette(0)",
        "palette = plt.get_cmap('tab10')\nvalues = [palette(0) for palette in data]",
        "palette = plt.get_cmap('tab10')\npalette += data[0]\npalette(0)",
        "palette = plt.get_cmap('tab10')\npalette: object = data[0]\npalette(0)",
        "palette = alias = plt.get_cmap('tab10')\npalette(0)",
        "palette = plt.get_cmap('tab10').reversed()\npalette(0)",
        "palette = plt.cm\npalette(0)",
        "palette = plt.cm.colors.LinearSegmentedColormap.from_list\npalette(0)",
        "module = plt.cm.colors\nmodule.LinearSegmentedColormap.from_list('x', ['red'])",
    ],
)
def test_colormap_compatibility_does_not_allow_arbitrary_calls_or_rebinding(statements):
    with pytest.raises(GenerationError):
        validate_code(_code(statements), "matplotlib")


@pytest.mark.parametrize("name", ["data", "title", "figsize"])
def test_colormap_callable_cannot_shadow_parameters(name):
    with pytest.raises(GenerationError):
        validate_code(_code(f"{name} = plt.get_cmap('tab10')\n{name}(0)"), "matplotlib")


def test_colormap_factory_requires_the_validated_pyplot_import():
    with pytest.raises(GenerationError):
        validate_code(
            _code("plt = data\npalette = plt.get_cmap('tab10')\npalette(0)", pyplot=False),
            "matplotlib",
        )


@pytest.mark.parametrize("spec, expected", [("", "1234.0"), (",g", "1,234"), (".2f", "1234.00")])
def test_literal_builtin_format_executes(spec, expected):
    code = _code(f"""
        note = format(data[0], {spec!r})
        fig, ax = plt.subplots()
        ax.plot(data)
        ax.set_title(note)
    """).replace("return data", "return fig")
    assert execute(code, [1234.0], backend="matplotlib").axes[0].get_title() == expected


@pytest.mark.parametrize(
    "statements",
    [
        "note = format(data[0], data[1])",
        "note = format(data[0], '>' + '10')",
        "note = format(data[0], f'{data[1]}g')",
        "note = format(data[0], '1001g')",
        "note = format(data[0], '.1001f')",
        "note = format(data[0], 'x' * 41)",
        "note = format(data[0], 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')",
        "note = format(data[0])",
        "note = format(data[0], format_spec='g')",
        "formatter = format\nnote = formatter(data[0], 'g')",
        "notes = list(map(format, data, data))",
        "format = data[0]\nnote = format(data[0], 'g')",
        "note = format(data[0], 'g')\nformat = data[0]",
        "for format in data:\n    note = format(data[0], 'g')",
        "notes = [format(data[0], 'g') for format in data]",
    ],
)
def test_format_rejects_dynamic_oversized_and_aliased_calls(statements):
    with pytest.raises(GenerationError):
        validate_code(_code(statements), "matplotlib")


@pytest.mark.parametrize("expression", ["format(data[0], '1001g')", "f'{data[0]:1001g}'"])
def test_builtin_and_fstring_formatting_share_width_limit(expression):
    with pytest.raises(GenerationError, match="Format width"):
        validate_code(_code(f"note = {expression}"), "matplotlib")


@pytest.mark.parametrize("digits", ["١٠٠١", "１００１", "𝟙𝟘𝟘𝟙"])
@pytest.mark.parametrize("precision", [False, True])
@pytest.mark.parametrize("builtin", [False, True])
def test_unicode_decimal_format_widths_and_precision_are_bounded(digits, precision, builtin):
    spec = ("." if precision else "") + digits + "f"
    expression = f"format(data[0], {spec!r})" if builtin else f"f'{{data[0]:{spec}}}'"
    with pytest.raises(GenerationError, match="Format width"):
        validate_code(_code(f"note = {expression}"), "matplotlib")


@pytest.mark.parametrize("spec", ["٤g", "４g", "𝟜g", "²>4g"])
@pytest.mark.parametrize("builtin", [False, True])
def test_small_unicode_format_specs_execute(spec, builtin):
    expression = f"format(data[0], {spec!r})" if builtin else f"f'{{data[0]:{spec}}}'"
    code = _code(f"""
        note = {expression}
        fig, ax = plt.subplots()
        ax.plot(data)
        ax.set_title(note)
    """).replace("return data", "return fig")
    assert execute(code, [12], backend="matplotlib").axes[0].get_title() == format(12, spec)
