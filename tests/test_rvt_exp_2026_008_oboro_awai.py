"""RVT L3 固着指標・AwaiVector 最小テスト。"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.rvt_exp_2026_008_awai_vector import RvtExp008AwaiVector
from experiments.rvt_exp_2026_008_oboro_generate import (
    build_oboro_demo_payload,
    entropy_from_logits,
    fixation_mass_from_logits,
    run_oboro_cli,
)


def test_fixation_mass_peaked():
    x = np.array([100.0, 0.0, 0.0])
    assert fixation_mass_from_logits(x) > 0.99


def test_entropy_from_logits_uniform_small_vocab():
    x = np.zeros(4, dtype=np.float64)
    ent = entropy_from_logits(x)
    assert ent > 1.35


def test_run_oboro_cli_rejects_bad_max_tokens():
    assert run_oboro_cli(["--max-new-tokens", "0"]) == 2


def test_build_oboro_demo_payload_lite_and_full():
    lite = build_oboro_demo_payload(profile="lite")
    assert lite["schema_version"] == "rvt_exp_008_oboro_lite.v1"
    assert lite.get("demo_synthetic") is True
    assert "logit_entropy" not in lite
    full = build_oboro_demo_payload(profile="full")
    assert full["schema_version"] == "rvt_exp_008_oboro.v2"
    assert len(full["stall_reason"]) == len(full["logit_fixation"])


def test_run_oboro_cli_demo_exits_zero(capsys):
    rc = run_oboro_cli(["--demo", "--profile", "full"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "rvt_exp_008_oboro.v2" in out
    assert "standalone_scope_ja" in out


def test_awai_vector_stack():
    acc = RvtExp008AwaiVector()
    acc.append(np.arange(6, dtype=np.float64))
    acc.append(np.ones(6))
    w = acc.stack()
    assert w.shape == (2, 6)
    with pytest.raises(ValueError):
        acc.append(np.ones(5))
