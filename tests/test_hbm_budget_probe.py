"""Phase 3 M4: HBM バイト予算プローブ（デモ・オフライン）。"""

import torch

from experiments.hbm_budget_probe import TEMPLATE_ORDER, run_probe


def test_run_probe_demo_schema_and_total():
    payload = run_probe(
        demo=True,
        model_name="gpt2",
        device=torch.device("cpu"),
        seq_len=16,
        batch_size=1,
        seed=0,
    )
    assert payload["schema_version"] == "hbm_budget.v1"
    assert payload["demo"] is True
    assert len(payload["rows"]) == len(TEMPLATE_ORDER)
    assert payload["total_act_bytes_estimated"] > 0
    emb = next(r for r in payload["rows"] if r["layer"] == "Embedding")
    assert emb["act_b"] is not None and emb["act_b"] > 0
