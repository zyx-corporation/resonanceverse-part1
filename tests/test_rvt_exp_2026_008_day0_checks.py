"""RVT Day 0 checks（windows.jsonl 検証）。"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.rvt_exp_2026_008_day0_checks import validate_windows_jsonl
from experiments.v7_mrmp_prepare import prepare_mrmp


def test_day0_checks_ok_on_prepared_mrmp(tmp_path: Path) -> None:
    corp = tmp_path / "multi_relational_multi_party_chat_corpus"
    ddir = corp / "dialogues" / "A_first_time"
    ddir.mkdir(parents=True)
    dlg = {
        "dialogue_id": "T0001",
        "dialogue_type": "First time",
        "utterances": [
            {
                "utterance_id": 0,
                "interlocutor_id": "a",
                "text": "こんにちは",
                "mention_to": [],
            },
            {
                "utterance_id": 1,
                "interlocutor_id": "b",
                "text": "どうも",
                "mention_to": [],
            },
        ],
        "evaluations": [],
    }
    payload = json.dumps(dlg, ensure_ascii=False)
    (ddir / "T0001.json").write_text(payload, encoding="utf-8")

    out = tmp_path / "out"
    prepare_mrmp(corpus_root=corp, out_dir=out, window=2, max_dialogues=None)
    win = out / "windows.jsonl"
    man = out / "manifest.json"
    r = validate_windows_jsonl(
        win,
        min_rows=2,
        expect_pair_rule="P1_prev_speaker",
        manifest_path=man,
        strict_manifest=True,
    )
    assert r["ok"] is True
    assert r["n_lines"] == 2
    assert r["manifest_n_utterance_rows"] == 2


def test_day0_checks_fails_min_rows(tmp_path: Path) -> None:
    win = tmp_path / "windows.jsonl"
    win.write_text(
        json.dumps(
            {
                "schema_version": "v7_mrmp_prepared.v1",
                "id": "x_u00000",
                "dialogue_id": "T",
                "dialogue_type": "",
                "utterance_id": 0,
                "pair_rule": "P1_prev_speaker",
                "speaker": "a",
                "speaker_src": None,
                "speaker_tgt": "a",
                "mention_to": [],
                "window_turns": 2,
                "text": "a: hi",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    r = validate_windows_jsonl(win, min_rows=5)
    assert r["ok"] is False
    assert any("min_rows" in e for e in r["errors"])


def test_day0_checks_fails_missing_key(tmp_path: Path) -> None:
    win = tmp_path / "windows.jsonl"
    one = json.dumps({"id": "only_id"}, ensure_ascii=False) + "\n"
    win.write_text(one, encoding="utf-8")
    r = validate_windows_jsonl(win)
    assert r["ok"] is False
    assert any("missing key" in e for e in r["errors"])
