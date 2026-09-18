import json

import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path, monkeypatch):
    """Keep persistent plot history and exports isolated from other tests/projects."""
    monkeypatch.chdir(tmp_path)

CV_CODE = """def plot_data(data, *, title=None, figsize=None):
    import numpy as np
    import matplotlib.pyplot as plt
    names = list(data)
    means = [np.mean(data[name]["r2"]) for name in names]
    fig, ax = plt.subplots(figsize=figsize or (8, 4))
    ax.bar(names, means)
    ax.set_ylabel("Mean R² across folds")
    ax.set_title(title or "Model comparison")
    fig.tight_layout()
    return fig
"""

ARRAY_CODE = """def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize or (8, 4))
    ax.plot(data)
    ax.set_title(title or "Values")
    return fig
"""

DF_CODE = """def plot_data(data, *, title=None, figsize=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=figsize or (8, 4))
    sns.scatterplot(data=data, x="spend", y="revenue", hue="channel", ax=ax)
    ax.set_title(title or "Spend and revenue")
    fig.tight_layout()
    return fig
"""

def response(code, explanation="Compare the observed values."):
    return json.dumps({"code": code, "explanation": explanation})


@pytest.fixture
def cv_data():
    return {"ridge": {"r2": [0.71, 0.75, 0.73]}, "forest": {"r2": [0.8, 0.82, 0.81]}}


@pytest.fixture
def fake_model(monkeypatch):
    from augplot import provider

    calls = []

    def install(*responses):
        sequence = iter(responses)

        def complete(**kwargs):
            calls.append(kwargs)
            result = next(sequence)
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(provider, "complete", complete)
        monkeypatch.setenv("AUGPLOT_MODEL", "test/fake-model")
        return calls

    return install
