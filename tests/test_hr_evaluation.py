import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt


def test_observations_are_json_safe_with_overlapping_annotations_and_log_values():
    path = Path(__file__).parents[1] / "examples" / "evaluate_hr.py"
    spec = importlib.util.spec_from_file_location("hr_evaluation", path)
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    fig, ax = plt.subplots()
    ax.scatter([1, 2], [-1, 1])
    ax.text(1, 0, "First label")
    ax.text(1, 0, "Second label")
    ax.twinx()
    try:
        result = evaluation.observations(fig, 2)
        json.dumps(result, allow_nan=False)
        assert result["annotation_box_overlaps"] == 1
        assert not result["native_log_y_axis"]
        assert all(result["checks"].values())
    finally:
        plt.close(fig)
