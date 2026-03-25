"""
Phase 2: 下流タスクの固定プロトコル（入口）。

- `--task sst2`: GLUE SST-2（感情二値分類）
- `--task boolq`: BoolQ（`datasets` スタンドアロン `boolq`、Yes/No を単一シーケンスに連結して二値分類）
- `--demo`: ミニスタブ（ネットワーク不要、`transformers` 不要）
- 既定: HF エンコーダ + マスク平均プール + 線形ヘッド（baseline）または
  エンコーダ + ResonantCore + 線形ヘッド（awai）
- awai の `--awai-readout`: `narrow`（6→ラベル・従来）、`projected`（6→hidden→ラベル）、
  `dual`（encoder プールと共鳴 6 次元を連結）
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.cultural_modulation import CulturalModulationAdapter
from core.reproducibility import set_experiment_seed
from core.resonant_core import ResonantCore


def _pool_masked(h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """(B,S,D) と (B,S) マスク → (B,D) 平均プール。"""
    m = attention_mask.unsqueeze(-1).to(dtype=h.dtype)
    denom = m.sum(dim=1).clamp(min=1e-6)
    return (h * m).sum(dim=1) / denom


class BaselineClassifier(nn.Module):
    """エンコーダ最終隠れ状態のマスク平均 → num_labels。"""

    def __init__(self, encoder: nn.Module, num_labels: int):
        super().__init__()
        self.encoder = encoder
        h = int(encoder.config.hidden_size)
        self.head = nn.Linear(h, num_labels)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        pooled = _pool_masked(h, attention_mask)
        return self.head(pooled)


class AwaiClassifier(nn.Module):
    """エンコーダ隠れ状態 → ResonantCore → 読み出し → num_labels。"""

    def __init__(
        self,
        encoder: nn.Module,
        num_labels: int,
        *,
        cultural_modulation: bool = False,
        num_nodes: int = 512,
        readout: str = "narrow",
    ):
        super().__init__()
        self.encoder = encoder
        h = int(encoder.config.hidden_size)
        self.readout = readout
        self.resonance = ResonantCore(h, num_nodes=num_nodes)
        if readout == "narrow":
            self.head = nn.Linear(6, num_labels)
            self.up: nn.Linear | None = None
        elif readout == "projected":
            self.up = nn.Linear(6, h)
            self.head = nn.Linear(h, num_labels)
        elif readout == "dual":
            self.up = None
            self.head = nn.Linear(h + 6, num_labels)
        else:
            raise ValueError(f"unknown awai readout: {readout}")
        self.cultural_adapter: CulturalModulationAdapter | None
        if cultural_modulation:
            self.cultural_adapter = CulturalModulationAdapter(h)
        else:
            self.cultural_adapter = None

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        r = self.resonance(h, attention_mask=attention_mask)
        if self.cultural_adapter is not None:
            r = r * self.cultural_adapter(h)
        pooled_r = _pool_masked(r, attention_mask)
        if self.readout == "narrow":
            return self.head(pooled_r)
        if self.readout == "projected":
            assert self.up is not None
            z = self.up(pooled_r)
            return self.head(z)
        pooled_h = _pool_masked(h, attention_mask)
        z = torch.cat([pooled_h, pooled_r], dim=-1)
        return self.head(z)


class _DemoEncoder(nn.Module):
    """HF 互換の最小エンコーダ（last_hidden_state のみ）。"""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": hidden_size})()
        self.emb = nn.Embedding(vocab_size, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        h = self.emb(input_ids)
        out = type("Out", (), {})()
        out.last_hidden_state = h
        return out


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().item())


def _demo_batches(
    device: torch.device,
    batch_size: int,
    vocab_size: int,
    seq_len: int,
    n_batches: int,
):
    for _ in range(n_batches):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
        y = torch.randint(0, 2, (batch_size,), device=device)
        yield x, mask, y


def _texts_and_labels_from_glue(split, task: str) -> tuple[list[str], list[int]]:
    if task == "sst2":
        texts = [str(x) for x in split["sentence"]]
        labels = [int(x) for x in split["label"]]
    elif task == "boolq":
        texts = [
            f"{q} {p}"
            for q, p in zip(
                split["question"],
                split["passage"],
                strict=True,
            )
        ]
        # スタンドアロン boolq は "answer"（bool）、旧 GLUE 名は "label" の場合あり
        if "label" in split.column_names:
            labels = [int(x) for x in split["label"]]
        else:
            labels = [int(bool(x)) for x in split["answer"]]
    else:
        raise ValueError(f"unknown glue binary task: {task}")
    return texts, labels


def _load_glue_binary_tensors(
    tok,
    task: str,
    *,
    max_seq_len: int,
    max_train: int,
    max_eval: int,
    seed: int,
):
    from datasets import load_dataset

    # BoolQ は `datasets` の GLUE コレクションに含まれないためスタンドアロンを読む
    if task == "boolq":
        ds = load_dataset("boolq")
    else:
        ds = load_dataset("glue", task)
    train_ds = ds["train"]
    eval_ds = ds["validation"]
    if max_train > 0 and len(train_ds) > max_train:
        train_ds = train_ds.shuffle(seed=seed).select(range(max_train))
    if max_eval > 0 and len(eval_ds) > max_eval:
        eval_ds = eval_ds.select(range(max_eval))

    def _encode(split):
        texts, labels = _texts_and_labels_from_glue(split, task)
        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"], torch.tensor(
            labels, dtype=torch.long
        )

    return _encode(train_ds), _encode(eval_ds)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase 2 downstream: GLUE binary tasks (SST-2, BoolQ)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--demo", action="store_true", help="スタブのみ（ネットワーク不要）")
    p.add_argument(
        "--task",
        choices=("sst2", "boolq"),
        default="sst2",
        help="GLUE タスク: sst2=感情、boolq=Yes/No 読解 QA（質問+パッセージを連結）",
    )
    p.add_argument("--model", default="gpt2", help="HF モデル名（--demo では未使用）")
    p.add_argument(
        "--integration",
        choices=("baseline", "awai", "awai-cultural"),
        default="baseline",
        help="baseline=プール+線形、awai=ResonantCore+線形、awai-cultural=調製あり",
    )
    p.add_argument(
        "--awai-readout",
        choices=("narrow", "projected", "dual"),
        default="narrow",
        help="awai/awai-cultural のみ: narrow=6→labels、projected=6→hidden→labels、dual=encoder+共鳴の連結",
    )
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-train-samples", type=int, default=4000, help="0=訓練分割の全件")
    p.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="検証に使う最大件数（0=検証分割の全件）",
    )
    p.add_argument("--freeze-encoder", action="store_true", help="エンコーダを固定（ヘッドのみ学習）")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    set_experiment_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    if args.demo:
        vocab_size, hidden = 512, 64
        enc = _DemoEncoder(vocab_size, hidden).to(device)
        if args.integration == "baseline":
            model = BaselineClassifier(enc, 2).to(device)
        else:
            model = AwaiClassifier(
                enc,
                2,
                cultural_modulation=args.integration == "awai-cultural",
                num_nodes=32,
                readout=args.awai_readout,
            ).to(device)
        train_it = _demo_batches(device, args.batch_size, vocab_size, 16, max(1, args.max_steps))
        eval_batches = list(
            _demo_batches(device, args.batch_size, vocab_size, 16, 3)
        )
    else:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            print("slm_downstream_skip: transformers:", e)
            sys.exit(0)

        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        enc = AutoModel.from_pretrained(args.model).to(device)
        if args.integration == "baseline":
            model = BaselineClassifier(enc, 2).to(device)
        else:
            model = AwaiClassifier(
                enc,
                2,
                cultural_modulation=args.integration == "awai-cultural",
                readout=args.awai_readout,
            ).to(device)
        try:
            (train_ids, train_mask, train_y), (
                eval_ids,
                eval_mask,
                eval_y,
            ) = _load_glue_binary_tensors(
                tok,
                args.task,
                max_seq_len=args.max_seq_len,
                max_train=args.max_train_samples,
                max_eval=args.max_eval_samples,
                seed=args.seed,
            )
        except Exception as e:
            print("slm_downstream_skip: datasets/glue:", e)
            sys.exit(0)

    if args.freeze_encoder:
        for par in model.encoder.parameters():
            par.requires_grad = False

    opt = torch.optim.AdamW((x for x in model.parameters() if x.requires_grad), lr=args.lr)
    model.train()

    losses: list[float] = []
    if args.demo:
        train_gen = _demo_batches(
            device, args.batch_size, 512, 16, max(1, args.max_steps + 2)
        )
        for step in range(args.max_steps):
            try:
                x, mask, y = next(train_gen)
            except StopIteration:
                train_gen = _demo_batches(
                    device, args.batch_size, 512, 16, max(1, args.max_steps + 2)
                )
                x, mask, y = next(train_gen)
            opt.zero_grad(set_to_none=True)
            logits = model(x, mask)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
    else:
        n_train = train_ids.size(0)
        g = torch.Generator()
        g.manual_seed(args.seed)
        bs = min(args.batch_size, n_train)
        for step in range(args.max_steps):
            sel = torch.randperm(n_train, generator=g)[:bs]
            x = train_ids[sel].to(device)
            mask = train_mask[sel].to(device)
            y = train_y[sel].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x, mask)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        if args.demo:
            ev_logits = []
            ev_y = []
            for x, mask, y in eval_batches:
                ev_logits.append(model(x, mask))
                ev_y.append(y)
            el = torch.cat(ev_logits, dim=0)
            ey = torch.cat(ev_y, dim=0)
            acc_eval = _accuracy(el, ey)
            acc_train = acc_eval
        else:
            e_ids = eval_ids.to(device)
            e_mask = eval_mask.to(device)
            e_y = eval_y.to(device)
            n = e_ids.size(0)
            correct = 0
            total = 0
            for start in range(0, n, args.batch_size):
                sl = slice(start, start + args.batch_size)
                logits = model(e_ids[sl], e_mask[sl])
                pred = logits.argmax(dim=-1)
                correct += int((pred == e_y[sl]).sum().item())
                total += e_ids[sl].size(0)
            acc_eval = correct / max(total, 1)
            acc_train = float("nan")

    task_key = "synthetic" if args.demo else f"glue_{args.task}"
    acc_tr_out = acc_train
    if isinstance(acc_train, float) and math.isnan(acc_train):
        acc_tr_out = None
    final_loss = losses[-1] if losses else float("nan")
    final_loss_out = None if (isinstance(final_loss, float) and math.isnan(final_loss)) else final_loss

    payload = {
        "mode": "demo" if args.demo else "hf",
        "task": task_key,
        "glue_task": args.task,
        "integration": args.integration,
        "awai_readout": args.awai_readout
        if args.integration in ("awai", "awai-cultural")
        else None,
        "model": args.model if not args.demo else "_DemoEncoder",
        "max_steps": args.max_steps,
        "freeze_encoder": bool(args.freeze_encoder),
        "accuracy_eval": acc_eval,
        "accuracy_train": acc_tr_out,
        "final_loss": final_loss_out,
        "loss_start": losses[0] if losses else None,
        "device": str(device),
    }
    print("slm_downstream_ok", json.dumps(payload, ensure_ascii=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
