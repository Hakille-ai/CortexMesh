from __future__ import annotations

from examples.inspect_memory import run as run_inspect_memory
from examples.save_and_reload import run as run_save_and_reload


def test_inspect_memory_reports_read_weights_and_memory_stats() -> None:
    report = run_inspect_memory(steps=1, seed=23)

    assert report["prompt"] == "a=3;b=7;c=1;?b="
    assert report["read_weights_shape"] == (report["token_count"], 8)
    assert report["memory_shape"] == (8, 24)
    assert 0.999 <= report["read_weight_row_sum_min"] <= 1.001
    assert 0.999 <= report["read_weight_row_sum_max"] <= 1.001
    assert report["memory_l2_max"] > 0
    assert len(report["top_reads"]) == report["token_count"]


def test_save_and_reload_example_smoke() -> None:
    report = run_save_and_reload(steps=1, seed=29)

    assert report["trained_steps"] == 1
    assert "before" in report
    assert report["status"] in {"ok", "unavailable"}
    if report["status"] == "ok":
        assert report["max_logit_delta"] < 1e-5
        assert "after" in report
    else:
        assert "No CortexMesh save/reload API" in report["message"]
