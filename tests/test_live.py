import os

import pytest

from himalia import plot


@pytest.mark.live
@pytest.mark.skipif(os.getenv("HIMALIA_LIVE_TEST") != "1", reason="Live inference is opt-in")
def test_live_cv_smoke(cv_data):
    viz = plot(cv_data, prompt="Compare mean R² and show individual fold scores", show=False)
    assert viz.figure.axes
    assert viz.code
    assert viz.explanation
