import pandas as pd
import pytest

import augplot as ap

EXPECTED_DATASETS = (
    "anagrams",
    "anscombe",
    "attention",
    "brain_networks",
    "car_crashes",
    "diamonds",
    "dots",
    "dowjones",
    "exercise",
    "flights",
    "fmri",
    "geyser",
    "glue",
    "healthexp",
    "iris",
    "mpg",
    "penguins",
    "planets",
    "seaice",
    "taxis",
    "tips",
    "titanic",
)


def test_sns_dataset_catalog_is_public_and_complete():
    assert ap.SNS_DATASETS == EXPECTED_DATASETS


def test_load_sns_dataset_delegates_to_seaborn(monkeypatch):
    expected = pd.DataFrame({"species": ["Adelie"], "body_mass_g": [3700]})
    calls = []

    def fake_load_dataset(name, **kwargs):
        calls.append((name, kwargs))
        return expected

    monkeypatch.setattr("augplot.datasets.sns.load_dataset", fake_load_dataset)

    actual = ap.load_sns_dataset(
        "penguins", cache=False, data_home="/tmp/seaborn-test", na_values=["missing"]
    )

    assert actual is expected
    assert calls == [
        (
            "penguins",
            {"cache": False, "data_home": "/tmp/seaborn-test", "na_values": ["missing"]},
        )
    ]


def test_load_sns_dataset_rejects_unknown_names_without_network(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Seaborn should not be called for an unknown dataset")

    monkeypatch.setattr("augplot.datasets.sns.load_dataset", fail_if_called)

    with pytest.raises(ValueError, match="Unknown Seaborn dataset.*Available datasets"):
        ap.load_sns_dataset("not-a-dataset")
