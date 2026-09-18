from augplot.prompts import PROMPT_VERSION, SYSTEM_PROMPT


def test_prompt_allows_chart_native_statistics_but_not_predictive_modeling():
    prompt = " ".join(SYSTEM_PROMPT.split())

    assert PROMPT_VERSION == "4"
    assert "everything the selected visualization backend can do" in prompt
    assert "The boundary is the figure" in prompt
    assert "regression or smoothing trend lines" in prompt
    assert "Backend-native transformations and statistical layers are visualization" in prompt
    assert "do not produce a fitted model, transformed dataset, predictions" in prompt
    assert "extrapolate trends or forecasts beyond supplied observations" in prompt
