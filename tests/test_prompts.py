from augplot.prompts import (
    CROSS_VALIDATION_GUIDANCE,
    RESPONSE_FORMAT,
    SYSTEM_PROMPT,
    system_prompt_for,
)


def test_prompt_keeps_a_bounded_workflow_and_helper_contract():
    assert len(SYSTEM_PROMPT) < 5_000
    assert "plot_data(data, *, title=None, figsize=None)" in SYSTEM_PROMPT
    assert "apx.smooth_trend" in SYSTEM_PROMPT
    assert "apx.label_curves" in SYSTEM_PROMPT
    assert "untrusted" in SYSTEM_PROMPT.lower()


def test_response_format_is_a_strict_unified_schema():
    assert RESPONSE_FORMAT["type"] == "json_schema"
    contract = RESPONSE_FORMAT["json_schema"]
    assert contract["strict"] is True
    schema = contract["schema"]
    assert set(schema["required"]) == {"status", "explanation", "code"}
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]["status"]["enum"]) == {
        "ok", "needs_input", "out_of_scope",
    }


def test_cross_validation_guidance_is_short_and_conditional():
    ordinary = system_prompt_for(request="Plot revenue", profile={"data": {"type": "DataFrame"}})
    cv_prompt = system_prompt_for(
        request="Compare the models",
        profile={"data": {"items": [{"key": "r2", "value": {"type": "list"}}]}},
    )

    assert len(CROSS_VALIDATION_GUIDANCE) < 700
    assert CROSS_VALIDATION_GUIDANCE not in ordinary
    assert CROSS_VALIDATION_GUIDANCE in cv_prompt
