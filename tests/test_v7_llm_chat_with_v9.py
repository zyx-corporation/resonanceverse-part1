"""tools.v7_llm_chat_with_v9 — ログヘルパと CLI --help。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.v7_llm_chat_with_v9 import append_v9_turn_jsonl


def test_append_v9_turn_jsonl_skips_when_disabled(tmp_path):
    p = tmp_path / "x.jsonl"
    append_v9_turn_jsonl(
        str(p),
        session_id="s",
        turn_index=0,
        reply="a",
        v9_enabled=False,
        v9_payload={"schema_version": "v9_rhythm_log.v1"},
    )
    assert not p.exists()


def test_append_v9_turn_jsonl_skips_empty_path():
    append_v9_turn_jsonl(
        "",
        session_id="s",
        turn_index=0,
        reply="a",
        v9_enabled=True,
        v9_payload={"schema_version": "v9_rhythm_log.v1"},
    )


def test_v9_batch_chat_turn_cli_help():
    script = _ROOT / "tools" / "v9_batch_chat_turn.py"
    r = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert "v9" in r.stdout.lower() or "v9" in (r.stdout + r.stderr).lower()


def test_generate_assistant_turn_v7_requires_torch_types():
    from core.v9_rhythm_policy import V9RhythmConfig, V9RhythmState
    from tools.v7_llm_chat_with_v9 import generate_assistant_turn_v7

    class _Bad:
        pass

    reply, st, pl = generate_assistant_turn_v7(
        chat_messages=[{"role": "user", "content": "x"}],
        system_summary=None,
        base_model=_Bad(),
        tokenizer=_Bad(),
        device=pytest.importorskip("torch").device("cpu"),
        awai_model=None,
        v9_enabled=False,
        v9_state=V9RhythmState(),
        v9_cfg=V9RhythmConfig(),
        pair_projector=None,
        layer_index=-1,
        d=6,
        delay_tau=0,
        omega_window_mode="cumulative",
        omega_last_n=8,
        model_name="dummy",
        max_new_tokens=4,
    )
    assert reply.startswith("（エラー）")
    assert isinstance(pl, dict)
    assert pl.get("suspend_reason") == "disabled"


@patch("tools.v7_llm_chat_with_v9.time.sleep")
@patch("tools.v7_llm_chat_with_v9.generate_completion", return_value="ok")
@patch("tools.v7_llm_chat_with_v9.build_generation_prompt", return_value="p")
@patch(
    "tools.v7_llm_chat_with_v9.inject_system_summary",
    side_effect=lambda msgs, summary: msgs,
)
@patch(
    "tools.v7_llm_chat_with_v9.tokenizer_supports_offset_pool",
    return_value=(True, None),
)
@patch("tools.v7_llm_chat_with_v9.v9_rhythm_before_chat_completion")
def test_generate_assistant_turn_v7_oboro_tail_jsonl_payload(
    mock_hook, _tok_pool, _inj, _build, _gen, _sleep
):
    """朧＋末尾スライス時、v9_log に oboro_history_tail_* / prompt_message_count* が載る。"""
    torch = pytest.importorskip("torch")
    from core.v9_rhythm_policy import RhythmDecision, V9RhythmConfig, V9RhythmState
    from tools.v7_llm_chat_with_v9 import generate_assistant_turn_v7

    mock_hook.return_value = (
        RhythmDecision(
            suspend_ms=0,
            suspend_reason="axes:none",
            oboro_burst=True,
            append_system_honesty=False,
            non_compliance="none",
        ),
        V9RhythmState(),
        {"schema_version": "v9_rhythm_log.v1"},
        "concat_proxy",
    )
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "user", "content": "d"},
    ]
    cfg = V9RhythmConfig(oboro_history_tail_messages=2)
    reply, _st, pl = generate_assistant_turn_v7(
        chat_messages=msgs,
        system_summary=None,
        base_model=object(),
        tokenizer=object(),
        device=torch.device("cpu"),
        awai_model=None,
        v9_enabled=True,
        v9_state=V9RhythmState(),
        v9_cfg=cfg,
        pair_projector=None,
        layer_index=-1,
        d=6,
        delay_tau=0,
        omega_window_mode="cumulative",
        omega_last_n=8,
        model_name="dummy",
        max_new_tokens=64,
    )
    assert reply == "ok"
    assert pl is not None
    assert pl.get("oboro_history_tail_messages") == 2
    assert pl.get("oboro_history_tail_sliced") is True
    assert pl.get("prompt_message_count") == 2
    assert pl.get("prompt_message_count_full") == 4
    assert pl.get("max_new_tokens_base") == 64
    assert "max_new_tokens_effective" in pl
