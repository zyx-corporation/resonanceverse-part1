"""MRMP 整形（v7_mrmp_prepare）の最小テスト。"""

import json
from pathlib import Path

def test_v7_mrmp_prepare_minimal(tmp_path: Path) -> None:
    corp = tmp_path / "multi_relational_multi_party_chat_corpus"
    ddir = corp / "dialogues" / "A_first_time"
    ddir.mkdir(parents=True)
    dlg = {
        "dialogue_id": "T0001",
        "dialogue_type": "First time",
        "interlocutors": ["a", "b"],
        "relationship": [],
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
        "evaluations": [
            {
                "interlocutor_id": "a",
                "informativeness": 3,
                "comprehension": 3,
                "familiarity": 3,
                "interest": 3,
                "proactiveness": 3,
                "satisfaction": 3,
            },
            {
                "interlocutor_id": "b",
                "informativeness": 4,
                "comprehension": 4,
                "familiarity": 4,
                "interest": 4,
                "proactiveness": 4,
                "satisfaction": 4,
            },
        ],
    }
    (ddir / "T0001.json").write_text(json.dumps(dlg, ensure_ascii=False), encoding="utf-8")

    from experiments.v7_mrmp_prepare import prepare_mrmp

    out = tmp_path / "out"
    m = prepare_mrmp(corpus_root=corp, out_dir=out, window=2, max_dialogues=None)
    assert m.get("n_utterance_rows") == 2
    assert m.get("n_dialogues_written") == 1
    lines = (out / "windows.jsonl").read_text(encoding="utf-8").strip().splitlines()
    r0 = json.loads(lines[0])
    assert r0["speaker_src"] is None
    r1 = json.loads(lines[1])
    assert r1["speaker_src"] == "a"
    assert "a:" in r1["text"] and "b:" in r1["text"]


def test_v7_mrmp_sample_pilot_demo() -> None:
    from experiments.v7_phase1a_pilot_jsonl import MRMP_LABEL_KEYS, load_jsonl, run_pilot

    p = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "data"
        / "v7_mrmp_sample.jsonl"
    )
    rows = load_jsonl(p)
    out = run_pilot(
        rows=rows,
        demo=True,
        model_name="gpt2",
        cpu=True,
        seed=0,
        layer_index=-1,
        label_keys=MRMP_LABEL_KEYS,
    )
    assert out["n_rows"] == 3
    assert "mrmp_informativeness_01" in out["correlations_label_vs_fro"]
