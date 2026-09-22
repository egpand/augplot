from augplot.prompts import (
    CROSS_VALIDATION_GUIDANCE,
    PROMPT_VERSION,
    RESPONSE_FORMAT,
    SYSTEM_PROMPT,
    system_prompt_for,
)


def test_prompt_has_explicit_sections_and_scope_boundary():
    prompt = " ".join(SYSTEM_PROMPT.split())

    assert PROMPT_VERSION == "10"
    for heading in (
        "# Core role and scope",
        "# Trust boundaries",
        "# Output contract",
        "# Code-generation contract",
        "# Deterministic-validator compatibility",
        "# Backend rules",
        "# General visual-quality rubric",
    ):
        assert heading in SYSTEM_PROMPT
    assert "everything the selected visualization backend can do" in prompt
    assert "The boundary is the figure" in prompt
    assert "regression or smoothing trend lines" in prompt
    assert "do not produce a fitted model, transformed dataset, predictions" in prompt
    assert "extrapolate trends or forecasts beyond supplied observations" in prompt


def test_prompt_describes_validator_compatibility_without_a_bypass():
    prompt = " ".join(SYSTEM_PROMPT.split())

    assert "independent default-deny validator" in prompt
    assert "must not attempt to bypass validation" in prompt
    assert "Never call Pandas `plot` or `hist`" in prompt
    assert "`apply`, `agg`, `aggregate`, `map`, or `transform`" in prompt
    assert "Do not use generic `set` methods or indirect call targets" in prompt
    assert "backend, file or path, URL, font-file, picker, `usetex`" in prompt
    assert "`zeros`, `ones`, `full`, or `repeat`" in prompt
    assert "concatenate or stack collections" in prompt
    assert "loop may iterate over the bounded Axes sequence" in prompt
    assert "columns selected from data explicitly capped with `head(N)`" in prompt
    assert "or `tail(N)`" in prompt
    assert "where `N` is at most 200" in prompt
    assert "put the Axes sequence first when styling panels" in prompt
    assert "Nested loops and nested comprehensions are not allowed" in prompt
    assert "return `out_of_scope`" in prompt


def test_response_format_is_a_strict_unified_schema():
    assert RESPONSE_FORMAT["type"] == "json_schema"
    contract = RESPONSE_FORMAT["json_schema"]
    assert contract["strict"] is True
    schema = contract["schema"]
    assert schema["required"] == ["status", "explanation", "code"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["status"]["enum"] == ["ok", "out_of_scope"]


def test_prompt_coordinates_emphasis_and_layout_across_chart_types():
    prompt = " ".join(SYSTEM_PROMPT.split())

    assert "Plan emphasis and layout together for every chart type" in prompt
    assert "points, lines, bars, cells, or regions" in prompt
    assert "do not obscure marks or rely on color alone" in prompt
    assert "individual observation, a category, or an aggregate" in prompt
    assert "state the aggregation when applicable" in prompt
    assert "prefer a border, marker, or connector" in prompt
    assert "legends for repeated categorical encodings, not one-off callouts" in prompt
    assert "adapt figure size, text size, and label formatting" in prompt
    assert "Across single and multi-panel figures" in prompt
    assert "must not overlap or be clipped" in prompt
    assert "allocate a dedicated layout region" in prompt


def test_prompt_keeps_tick_labels_individually_readable():
    prompt = " ".join(SYSTEM_PROMPT.split())

    assert "keep them out of axis-title and tick-label regions" in prompt
    assert "Tick labels must remain individually distinguishable" in prompt
    assert "must not visually merge" in prompt
    assert "prefer horizontal labels when short labels fit" in prompt
    assert "reduce tick frequency without removing data" in prompt


def test_cross_validation_guidance_is_conditional():
    ordinary = system_prompt_for(request="Plot revenue", profile={"data": {"type": "DataFrame"}})
    cv_prompt = system_prompt_for(
        request="Compare the models",
        profile={"data": {"items": [{"key": "r2", "value": {"type": "list"}}]}},
    )

    assert CROSS_VALIDATION_GUIDANCE not in SYSTEM_PROMPT
    assert CROSS_VALIDATION_GUIDANCE not in ordinary
    assert CROSS_VALIDATION_GUIDANCE in cv_prompt
    assert "does not establish statistical significance" in cv_prompt
