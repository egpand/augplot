import json

import pytest
from conftest import ARRAY_CODE, response

from augplot import GenerationError, ScopeError
from augplot.execution import parse_response, validate_code


@pytest.mark.parametrize(
    "statement",
    [
        "import os as os",
        "import pathlib as paths",
        "import requests as requests",
        "import numpy as _np",
        "from pandas import read_csv",
        'open("/tmp/private")',
        "data.__class__",
        'getattr(data, "shape")',
        'pd.read_csv("https://example.com")',
        'np.load("data.npy")',
        'ax.figure.savefig("out.png")',
        'exec("print(1)")',
        "while True:\n        pass",
        "plot_data(data)",
        'plt.rcParams["font.size"] = 99',
        'plt.style.use("dark_background")',
    ],
)
def test_rejects_prohibited_constructs(statement):
    code = ARRAY_CODE.replace("ax.plot(data)", statement)
    with pytest.raises(GenerationError):
        validate_code(code, "auto")


@pytest.mark.parametrize(
    "code",
    [
        "print('side effect')\n" + ARRAY_CODE,
        "@decorator\n" + ARRAY_CODE,
        ARRAY_CODE.replace("figsize=None", "figsize=func()"),
        ARRAY_CODE.replace("plot_data(data,", "plot_data(data: func(),"),
        ARRAY_CODE + "\ndef another():\n    pass\n",
        ARRAY_CODE.replace("ax.plot(data)", "def nested():\n        pass"),
    ],
)
def test_rejects_executable_definition_metadata_and_extra_definitions(code):
    with pytest.raises(GenerationError):
        validate_code(code, "auto")


def test_backend_import_enforced():
    with pytest.raises(GenerationError):
        validate_code(ARRAY_CODE.replace("ax.plot(data)", "import seaborn as sns"), "matplotlib")


def test_accepts_single_json_fence():
    code, explanation = parse_response("```json\n" + response(ARRAY_CODE) + "\n```")
    assert code == ARRAY_CODE
    assert explanation


@pytest.mark.parametrize(
    "payload",
    [
        {"code": ARRAY_CODE, "explanation": "Missing status"},
        {"status": "unknown", "code": ARRAY_CODE, "explanation": "Bad status"},
        {"status": "ok", "code": "", "explanation": "Missing code"},
        {"status": "ok", "code": ARRAY_CODE, "explanation": "Fine", "extra": True},
        {"status": "out_of_scope", "code": ARRAY_CODE, "explanation": "Must be empty"},
    ],
)
def test_rejects_responses_outside_unified_schema(payload):
    with pytest.raises(GenerationError):
        parse_response(json.dumps(payload))


def test_parses_unified_out_of_scope_response():
    payload = {
        "status": "out_of_scope",
        "code": "",
        "explanation": "Provide upstream predictions.",
    }
    with pytest.raises(ScopeError, match="upstream predictions"):
        parse_response(json.dumps(payload))
