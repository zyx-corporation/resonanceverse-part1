"""Phase3 P0: phase3_claim_run（デモ・オフライン）。"""

from experiments.phase3_claim_run import run_claim_bundle


def test_run_claim_bundle_demo():
    b = run_claim_bundle(
        seed=0,
        cpu=True,
        demo=True,
        model="gpt2",
        max_new_tokens=4,
        warmup=1,
        repeats=2,
        seq_len=32,
        batch_size=1,
        router_step_stride=None,
        with_squad=False,
        squad_demo=True,
        squad_max_steps=3,
        squad_max_train=8,
        squad_max_eval=0,
    )
    assert b["schema_version"] == "phase3_claim_bundle.v1"
    assert b["meta"]["schema_version"] == "phase3_claim_meta.v1"
    assert b["two_tier_sweep"]["schema_version"] == "two_tier_sweep.v1"
    assert b["hbm_budget"]["schema_version"] == "hbm_budget.v1"


def test_run_claim_bundle_with_squad_demo():
    """P2: squad_span 同梱（合成 run_demo・オフライン）。"""
    b = run_claim_bundle(
        seed=0,
        cpu=True,
        demo=True,
        model="gpt2",
        max_new_tokens=4,
        warmup=1,
        repeats=2,
        seq_len=32,
        batch_size=1,
        router_step_stride=None,
        with_squad=True,
        squad_demo=True,
        squad_max_steps=3,
        squad_max_train=8,
        squad_max_eval=0,
    )
    assert b["schema_version"] == "phase3_claim_bundle.v1"
    assert "squad_span" in b
    assert b["squad_span"].get("schema_version") == "squad_span.v1"
