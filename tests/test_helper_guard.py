import pytest

from augplot import GenerationError
from augplot.execution import validate_code


@pytest.mark.parametrize(
    "statement",
    [
        "apx.np.asarray([1])",
        "apx.__dict__",
        "apx.PLOTTING_VERSION",
        "apx.open('anything')",
        "helper = apx.smooth_trend",
        "apx.smooth_trend = np.mean",
        "apx = data",
        "alias = apx",
        "modules = [apx]",
        "modules = dict(helper=apx)",
        "array = np.asarray(apx)",
    ],
)
def test_plotting_helper_import_exposes_only_direct_helpers(statement):
    code = f"""def plot_data(data, *, title=None, figsize=None):
    import augplot.plotting as apx
    import numpy as np
    {statement}
    return data
"""
    with pytest.raises(GenerationError):
        validate_code(code, "matplotlib")


@pytest.mark.parametrize("backend", ["matplotlib", "seaborn", "auto"])
def test_plotting_helpers_available_with_every_backend(backend):
    validate_code(
        """def plot_data(data, *, title=None, figsize=None):
    import augplot.plotting as apx
    segments = apx.smooth_trend(data[0], data[1])
    return segments
""",
        backend,
    )


def test_legacy_pyplot_colormap_constructor_executes_without_module_escape():
    from augplot.execution import execute

    code = '''def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    palette = plt.cm.colors.LinearSegmentedColormap.from_list("custom", ["blue", "red"])
    fig, ax = plt.subplots()
    ax.scatter(data, data, c=data, cmap=palette)
    return fig
'''
    figure = execute(code, [0, 1], backend="matplotlib")
    assert figure.axes[0].collections[0].get_cmap().name == "custom"


@pytest.mark.parametrize(
    "statement",
    [
        "module = plt.cm",
        "module = plt.cm.colors",
        "constructor = plt.cm.colors.LinearSegmentedColormap.from_list",
        "values = [plt.cm.colors.LinearSegmentedColormap.from_list]",
        "plt.cm.colors.LinearSegmentedColormap.from_list = str",
        "plt.cm.colors.LinearSegmentedColormap.from_list.__globals__",
        "plt.cm.colors.ColorConverter()",
        "plt.cm.colors.LinearSegmentedColormap.from_list.extra()",
    ],
)
def test_colormap_compatibility_does_not_allow_module_or_callable_escapes(statement):
    code = f'''def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    {statement}
    return data
'''
    with pytest.raises(GenerationError):
        validate_code(code, "matplotlib")
