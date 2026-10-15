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
