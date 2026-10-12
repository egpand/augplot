import os

import pytest

from augplot import models, plot


@pytest.mark.live
@pytest.mark.skipif(os.getenv("AUGPLOT_LIVE_TEST") != "1", reason="Live inference is opt-in")
def test_live_cv_smoke(cv_data):
    viz = plot(cv_data, prompt="Compare mean R² and show individual fold scores", show=False)
    assert viz.figure.axes
    assert viz.code
    assert viz.explanation


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("AUGPLOT_LIVE_CHATGPT_TEST") != "1", reason="ChatGPT plan inference is opt-in",
)
def test_live_chatgpt_generation_and_refinement(cv_data):
    """Use a connection signed in beforehand; never launch authentication from pytest."""
    catalog = models()
    assert not catalog.empty
    viz = plot(
        cv_data, auth="chatgpt", prompt="Compare mean R² and show individual fold scores",
        show=False, cache_dir=None,
    )
    selected = viz.model
    assert viz.figure.axes and viz.code and viz.explanation
    viz.refine("Use horizontal bars and label each observed mean", show=False)
    assert viz.model == selected
    assert viz.figure.axes and viz.code and viz.explanation
