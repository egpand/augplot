from io import BytesIO
from types import SimpleNamespace

import IPython
import IPython.display
import matplotlib.pyplot as plt
import pytest
from PIL import Image as PillowImage

from himalia import ConfigurationError, Visualizer


@pytest.fixture
def notebook_display(monkeypatch):
    # Initialize pyplot before replacing get_ipython with a minimal notebook stand-in.
    plt.switch_backend("Agg")
    outputs = []
    monkeypatch.setattr(IPython, "get_ipython", lambda: SimpleNamespace(kernel=object()))
    monkeypatch.setattr(IPython.display, "display", outputs.append)
    return outputs


@pytest.mark.parametrize("format_name", ["retina", "png", "svg"])
def test_explicit_display_preserves_figure_and_notebook_settings(format_name, notebook_display):
    from matplotlib_inline.config import InlineBackend

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot([1, 3, 2])
    before_formats = set(InlineBackend.instance().figure_formats)
    before_dpi = fig.dpi
    before_size = fig.get_size_inches().copy()
    try:
        Visualizer(display_format=format_name)._display(fig)
        assert len(notebook_display) == 1
        output = notebook_display[0]
        if format_name == "svg":
            assert isinstance(output, IPython.display.SVG)
            assert "<svg" in output.data
        else:
            assert isinstance(output, IPython.display.Image)
            width, height = PillowImage.open(BytesIO(output.data)).size
            if format_name == "retina":
                assert output.width == width // 2
                assert output.height == height // 2
                assert width > 600
            else:
                assert output.width is None
                assert output.height is None
                assert width < 500
        assert fig.dpi == before_dpi
        assert (fig.get_size_inches() == before_size).all()
        assert set(InlineBackend.instance().figure_formats) == before_formats
    finally:
        plt.close(fig)


def test_plotly_keeps_its_renderer(notebook_display):
    go = pytest.importorskip("plotly.graph_objects")
    fig = go.Figure(go.Scatter(y=[1, 2]))
    Visualizer(backend="plotly")._display(fig)
    assert notebook_display == [fig]


def test_script_does_not_render(monkeypatch):
    monkeypatch.setattr(IPython, "get_ipython", lambda: None)

    def unexpected_display(*args, **kwargs):
        pytest.fail("Display should not run outside a notebook kernel")

    monkeypatch.setattr(IPython.display, "display", unexpected_display)
    Visualizer()._display(object())


def test_invalid_format_rejected():
    with pytest.raises(ConfigurationError, match="display_format"):
        Visualizer(display_format="jpeg")
