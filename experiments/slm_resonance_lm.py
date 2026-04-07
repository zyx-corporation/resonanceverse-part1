"""
Phase B（入口）: AwaiIntegratedSLM の最小学習ループ（次トークン予測・クロスエントロピー）。

- `--demo`: ミニスタブ＋ランダムトークン（ネットワーク不要）
- `--data random`: 実 HF モデルでもランダムトークンで学習（スモーク）
- `--data wikitext`: Wikitext-2-raw の先頭をトークナイズして学習（要 `datasets`、初回 DL）
- `--baseline-hf`: Resonanceverse 未統合の HF 因果 LM のみ（導入前ベースライン）
- `--cultural-modulation`: Phase 1B — `CulturalModulationAdapter` で共鳴特徴を調製（`--baseline-hf` 不可）
- 終了時に評価分割で **perplexity**（exp(mean CE)）を付記可能（`--eval-ppl`）
- `AwaiIntegratedSLM` は **隠れ状態と共鳴 6 次元を連結**して語彙ロジットを出す（`core/resonant_core.py`）
- JSON に `train_time_s` / `steps_per_sec` /（CUDA 時）`cuda_peak_memory_bytes`
- `--checkpoint-out` で学習後の ``model.state_dict()`` を保存（Streamlit 統合ロード用）
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(_ROOT / "experiments"))

import torch
import torch.nn as nn
import torch.nn.functional as F

import slm_data

from core.reproducibility import set_experiment_seed
from core.resonant_core import AwaiIntegratedSLM


class _MiniCausalStub(nn.Module):
    """HF 互換の最小因果スタブ（hidden_states[-1] のみ）。"""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": hidden_size, "vocab_size": vocab_size})()
        self.emb = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False):
        h = self.emb(input_ids)
        out = type("Out", (), {})()
        out.hidden_states = [h, h]
        return out


class HfCausalLMOnly(nn.Module):
    """Resonanceverse 未統合のベースライン: HF 因果 LM の語彙ロジットのみ（導入前比較用）。"""

    def __init__(self, base_slm_model: nn.Module):
        super().__init__()
        self.base_model = base_slm_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.base_model(input_ids).logits


def causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    v = shift_logits.size(-1)
    return F.cross_entropy(shift_logits.view(-1, v), shift_labels.view(-1))


def _freeze_base(model: AwaiIntegratedSLM) -> None:
    for p in model.base_model.parameters():
        p.requires_grad = False


@torch.no_grad()
def mean_nll_on_chunks(
    model: nn.Module,
    chunks_2d: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    """各バッチの CE（トークン平均）をバッチ数で平均。"""
    if chunks_2d.numel() == 0:
        return float("nan")
    model.eval()
    losses: list[float] = []
    for batch in slm_data.batched_chunks(chunks_2d, batch_size, device):
        if batch.size(0) == 0:
            continue
        loss = causal_lm_loss(model(batch), batch)
        losses.append(float(loss.item()))
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    p = argparse.ArgumentParser(description="AwaiIntegratedSLM minimal LM training")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--demo", action="store_true", help="ミニスタブのみ（ネットワーク不要）")
    p.add_argument("--model", default="gpt2", help="HF モデル名（--demo では未使用）")
    p.add_argument(
        "--data",
        choices=("random", "wikitext"),
        default="random",
        help="random=ランダムトークン、wikitext=Wikitext-2-raw 先頭（--demo では random のみ）",
    )
    p.add_argument("--max-chars", type=int, default=200_000, help="wikitext で連結する最大文字数")
    p.add_argument("--eval-frac", type=float, default=0.1, help="末尾を評価用に回す割合")
    p.add_argument("--eval-ppl", action="store_true", help="学習後に評価分割で perplexity を計算")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--freeze-base", action="store_true")
    p.add_argument(
        "--baseline-hf",
        action="store_true",
        help="ResonantCore を使わず HF 因果 LM のロジットのみ（導入前ベースライン）。--demo とは併用不可",
    )
    p.add_argument(
        "--cultural-modulation",
        action="store_true",
        help="Phase 1B: 隠れ状態から調製スカラーを学習し共鳴特徴に乗算（Awai のみ）。--baseline-hf 不可",
    )
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--checkpoint-out",
        type=Path,
        default=None,
        help="学習後に model.state_dict() を保存（Streamlit の統合チェックポイント欄に渡す）",
    )
    args = p.parse_args()

    if args.demo and args.baseline_hf:
        p.error("--baseline-hf は --demo と併用できません")
    if args.baseline_hf and args.cultural_modulation:
        p.error("--cultural-modulation は --baseline-hf と併用できません")

    set_experiment_seed(args.seed)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    if args.demo:
        args.data = "random"

    wikitext_train_chunks: torch.Tensor | None = None
    wikitext_eval_chunks: torch.Tensor | None = None

    if args.demo:
        vocab, hidden = 256, 64
        base = _MiniCausalStub(vocab, hidden).to(device)
        model = AwaiIntegratedSLM(base, cultural_modulation=args.cultural_modulation).to(
            device
        )
        data_iter = slm_data.random_token_batches(vocab, args.batch, args.seq_len, device)
    else:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            print("slm_resonance_lm_skip: transformers:", e)
            sys.exit(0)

        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(args.model).to(device)
        if args.baseline_hf:
            model = HfCausalLMOnly(base).to(device)
        else:
            model = AwaiIntegratedSLM(
                base, cultural_modulation=args.cultural_modulation
            ).to(device)
        vocab = int(base.config.vocab_size)

        if args.data == "random":
            data_iter = slm_data.random_token_batches(vocab, args.batch, args.seq_len, device)
        else:
            try:
                raw = slm_data.load_wikitext_token_ids(tok, max_chars=args.max_chars)
            except Exception as e:
                print("slm_resonance_lm_skip: wikitext/datasets:", e)
                sys.exit(0)
            train_1d, eval_1d = slm_data.train_eval_split(raw, eval_frac=args.eval_frac)
            wikitext_train_chunks = slm_data.token_ids_to_chunks(train_1d, args.seq_len)
            wikitext_eval_chunks = slm_data.token_ids_to_chunks(eval_1d, args.seq_len)
            if wikitext_train_chunks.numel() == 0:
                print("slm_resonance_lm_error: no train chunks; lower seq_len or raise max_chars")
                sys.exit(1)
            data_iter = slm_data.cycle_batches(wikitext_train_chunks, args.batch, device)

    if args.freeze_base:
        _freeze_base(model)
        if args.baseline_hf:
            print(
                "slm_resonance_lm_warn: --baseline-hf では base_model が全体のため "
                "--freeze-base で学習可能パラメータが残りません（比較は損失がほぼ不変になります）",
                file=sys.stderr,
            )

    opt = torch.optim.AdamW((x for x in model.parameters() if x.requires_grad), lr=args.lr)
    model.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(args.max_steps):
        x = next(data_iter)
        opt.zero_grad(set_to_none=True)
        loss = causal_lm_loss(model(x), x)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    train_time_s = time.perf_counter() - t0

    cuda_peak_bytes = None
    if device.type == "cuda":
        cuda_peak_bytes = int(torch.cuda.max_memory_allocated(device))

    final = losses[-1] if losses else float("nan")

    ppl_train = None
    ppl_eval = None
    if args.eval_ppl and not args.demo and args.data == "wikitext":
        if wikitext_eval_chunks is not None and wikitext_eval_chunks.numel() > 0:
            nll_e = mean_nll_on_chunks(model, wikitext_eval_chunks, args.batch, device)
            ppl_eval = math.exp(nll_e)
        if wikitext_train_chunks is not None and wikitext_train_chunks.numel() > 0:
            nll_t = mean_nll_on_chunks(model, wikitext_train_chunks, args.batch, device)
            ppl_train = math.exp(nll_t)

    integration = "demo_stub" if args.demo else ("hf_baseline" if args.baseline_hf else "awai_resonance")
    payload = {
        "mode": "demo" if args.demo else "hf",
        "integration": integration,
        "data": args.data if not args.demo else "random",
        "model": args.model if not args.demo else "_MiniCausalStub",
        "max_steps": args.max_steps,
        "freeze_base": bool(args.freeze_base),
        "cultural_modulation": bool(args.cultural_modulation),
        "final_loss": final,
        "loss_start": losses[0] if losses else None,
        "device": str(device),
        "train_time_s": train_time_s,
        "steps_per_sec": (args.max_steps / train_time_s) if train_time_s > 0 else None,
        "cuda_peak_memory_bytes": cuda_peak_bytes,
        "perplexity_train": ppl_train,
        "perplexity_eval": ppl_eval,
    }
    print("slm_resonance_lm_ok", json.dumps(payload, ensure_ascii=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint_out)
        print(
            "slm_resonance_lm_checkpoint_saved",
            json.dumps({"path": str(args.checkpoint_out)}, ensure_ascii=False),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
