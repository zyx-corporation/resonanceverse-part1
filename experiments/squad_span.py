"""
Phase 3 M5: 抽出型 QA（スパン予測）の最小プロトコル — SQuAD v1 形式。

- --demo: 合成データ + ミニモデル（transformers 不要）
- 実データ: datasets の `squad` + `AutoModelForQuestionAnswering`（distilbert 推奨）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.reproducibility import set_experiment_seed


class MiniSpanQA(nn.Module):
    """デモ用: 埋め込み + 開始／終了ロジット (B, L)。"""

    def __init__(self, vocab_size: int, hidden: int, max_len: int):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, hidden)
        self.start = nn.Linear(hidden, 1)
        self.end = nn.Linear(hidden, 1)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.emb(input_ids)
        start_logits = self.start(h).squeeze(-1)
        end_logits = self.end(h).squeeze(-1)
        return start_logits, end_logits


def _span_loss(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    start_pos: torch.Tensor,
    end_pos: torch.Tensor,
) -> torch.Tensor:
    return (
        F.cross_entropy(start_logits, start_pos)
        + F.cross_entropy(end_logits, end_pos)
    ) / 2.0


def _exact_match(pred_start: torch.Tensor, pred_end: torch.Tensor, y0: torch.Tensor, y1: torch.Tensor) -> float:
    ok = (pred_start == y0) & (pred_end == y1)
    return float(ok.float().mean().item())


def run_demo(device: torch.device, max_steps: int, seed: int) -> dict:
    set_experiment_seed(seed)
    max_len = 32
    vocab = 200
    model = MiniSpanQA(vocab, 64, max_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses: list[float] = []
    for _ in range(max_steps):
        x = torch.randint(1, vocab, (2, max_len), device=device)
        start_pos = torch.randint(1, max_len // 2, (2,), device=device)
        end_pos = torch.randint(max_len // 2, max_len - 1, (2,), device=device)
        opt.zero_grad(set_to_none=True)
        sl, el = model(x)
        loss = _span_loss(sl, el, start_pos, end_pos)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    model.eval()
    with torch.no_grad():
        x = torch.randint(1, vocab, (4, max_len), device=device)
        start_pos = torch.randint(1, max_len // 2, (4,), device=device)
        end_pos = torch.randint(max_len // 2, max_len - 1, (4,), device=device)
        sl, el = model(x)
        ps = sl.argmax(dim=-1)
        pe = el.argmax(dim=-1)
        em = _exact_match(ps, pe, start_pos, end_pos)
    return {
        "schema_version": "squad_span.v1",
        "mode": "demo",
        "task": "synthetic_span",
        "max_steps": max_steps,
        "final_loss": losses[-1] if losses else float("nan"),
        "exact_match_eval": em,
    }


def _make_prepare_fn(tokenizer, max_length: int):
    """SQuAD v1: 文脈側オフセットから開始・終了トークン index を求める。"""

    def prepare(examples):
        enc = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )
        starts: list[int] = []
        ends: list[int] = []
        for i, offsets in enumerate(enc["offset_mapping"]):
            ans = examples["answers"][i]
            if not ans["text"]:
                starts.append(0)
                ends.append(0)
                continue
            start_char = ans["answer_start"][0]
            end_char = start_char + len(ans["text"][0])
            seq_ids = enc.sequence_ids(i)
            idx_start = idx_end = 0
            found_start = False
            for idx, (off, sid) in enumerate(zip(offsets, seq_ids)):
                if sid != 1:
                    continue
                if not found_start and off[0] <= start_char < off[1]:
                    idx_start = idx
                    found_start = True
                if off[0] < end_char <= off[1]:
                    idx_end = idx
                    break
            else:
                idx_end = idx_start if found_start else 0
            if not found_start:
                idx_start = idx_end = 0
            starts.append(idx_start)
            ends.append(idx_end)
        enc["start_positions"] = starts
        enc["end_positions"] = ends
        enc.pop("offset_mapping")
        return enc

    return prepare


def run_squad_hf(
    device: torch.device,
    model_name: str,
    max_steps: int,
    max_train: int,
    max_eval: int,
    max_length: int,
    seed: int,
) -> dict:
    from datasets import load_dataset
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    set_experiment_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
    prepare = _make_prepare_fn(tok, max_length)

    ds = load_dataset("squad", split="train")
    if max_train > 0:
        ds = ds.select(range(min(max_train, len(ds))))

    ds_t = ds.map(
        prepare,
        batched=True,
        remove_columns=ds.column_names,
        desc="tokenize_train",
    )
    ds_t.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "start_positions", "end_positions"],
    )

    loader = torch.utils.data.DataLoader(ds_t, batch_size=4, shuffle=True)
    it = iter(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    model.train()
    losses: list[float] = []
    for step in range(max_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out.loss
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    em_eval: float | None = None
    if max_eval > 0:
        eval_raw = load_dataset("squad", split="validation")
        eval_raw = eval_raw.select(range(min(max_eval, len(eval_raw))))
        eval_ds = eval_raw.map(
            prepare,
            batched=True,
            remove_columns=eval_raw.column_names,
            desc="tokenize_eval",
        )
        eval_ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "start_positions", "end_positions"],
        )
        ev_loader = torch.utils.data.DataLoader(eval_ds, batch_size=8, shuffle=False)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in ev_loader:
                gold_s = batch["start_positions"].to(device)
                gold_e = batch["end_positions"].to(device)
                bsz = gold_s.size(0)
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                ps = out.start_logits.argmax(dim=-1)
                pe = out.end_logits.argmax(dim=-1)
                correct += int(((ps == gold_s) & (pe == gold_e)).sum().item())
                total += bsz
        em_eval = correct / total if total else 0.0

    return {
        "schema_version": "squad_span.v1",
        "mode": "hf",
        "task": "squad_v1",
        "model": model_name,
        "max_steps": max_steps,
        "max_train_samples": max_train,
        "max_eval_samples": max_eval,
        "final_loss": losses[-1] if losses else float("nan"),
        "exact_match_eval": em_eval,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 3 M5: SQuAD span QA minimal")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--max-train-samples", type=int, default=256)
    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    if args.demo:
        payload = run_demo(device, args.max_steps, args.seed)
    else:
        try:
            payload = run_squad_hf(
                device,
                args.model,
                args.max_steps,
                args.max_train_samples,
                args.max_eval_samples,
                args.max_length,
                args.seed,
            )
        except Exception as e:
            print("squad_span_skip:", e)
            sys.exit(0)

    print("squad_span_ok", json.dumps(payload, ensure_ascii=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
