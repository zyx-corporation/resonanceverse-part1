"""Phase 3 M5: 抽出型 QA（デモ・オフライン）。"""

import torch

from experiments.squad_span import run_demo


def test_run_demo_squad_span_schema():
    payload = run_demo(device=torch.device("cpu"), max_steps=5, seed=0)
    assert payload["schema_version"] == "squad_span.v1"
    assert payload["mode"] == "demo"
    assert payload["task"] == "synthetic_span"
    assert payload["max_steps"] == 5
    assert "final_loss" in payload
    assert "exact_match_eval" in payload
