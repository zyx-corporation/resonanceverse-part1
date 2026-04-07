"""
RVT-EXP-2026-008 L2（最小実装）: **eager** 注意の ``eager_attention_forward`` 事後ブレンド。

``transformers`` の各種 ``modeling_*`` にある ``eager_attention_forward`` をラップし、
softmax 済み確率に対して SYM / WASYM を適用してから ``value``（GQA 時は repeat_kv 後）
との積を再計算する。

**前提**: ``attn_implementation=\"eager\"``。**SDPA / Flash では L2 不可**（重みが表に出ない）。

**セッション管理（厳密）**: ``gpt2_rvt_inject_session`` は入れ子対応の参照カウント。
深さ 0→1 で既知アーキの ``eager_attention_forward`` をパッチし、未登録は ``apply`` 時に遅延パッチ。
1→0 で **登録済みをすべて** 原関数へ復帰。
"""

from __future__ import annotations

import importlib
import inspect
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_INJECT_SESSION_LOCK = threading.Lock()
_INJECT_SESSION_DEPTH = 0

# (python_module, original eager_attention_forward)
_PATCH_STACK: list[tuple[Any, Any]] = []
_PATCHED_MODULE_IDS: set[int] = set()

_PREFETCH_MODELING_NAMES: tuple[str, ...] = (
    "gpt2",
    "llama",
    "mistral",
    "gemma2",
    "qwen2",
    "phi3",
)


def _blend_post_softmax(
    p: Any,
    *,
    mode: str,
    eps: float,
) -> Any:
    import torch

    if mode == "base":
        return p
    if p.dim() < 2:
        return p
    tr = p.transpose(-1, -2)
    if mode == "sym":
        q = 0.5 * (p + tr)
    elif mode == "wasym":
        q = p + float(eps) * (p - tr)
        q = torch.clamp(q, min=0.0)
    else:
        raise ValueError(f"unknown mode {mode!r} (use base|sym|wasym)")
    s = q.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return q / s


def _recompute_attn_output_from_weights(
    module: Any,
    w2: Any,
    value: Any,
) -> Any:
    import torch

    if hasattr(module, "num_key_value_groups"):
        pymod = sys.modules[type(module).__module__]
        repeat_kv = getattr(pymod, "repeat_kv", None)
        if repeat_kv is None:
            raise RuntimeError(
                "rvt L2: GQA attention but repeat_kv missing in "
                f"{type(module).__module__}",
            )
        vs = repeat_kv(value, module.num_key_value_groups)
        return torch.matmul(w2, vs).transpose(1, 2).contiguous()
    return torch.matmul(w2, value).transpose(1, 2).contiguous()


def _rvt_unified_eager_wrapper(orig_fn: Any) -> Any:
    sig = inspect.signature(orig_fn)

    def wrapped(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        attn_output, attn_weights = orig_fn(*args, **kwargs)
        module = args[0]
        cfg = getattr(module, "_rvt_inject", None)
        if cfg is None:
            return attn_output, attn_weights
        mode = str(cfg.get("mode", "base"))
        if mode == "base":
            return attn_output, attn_weights
        eps = float(cfg.get("eps", 0.05))
        w2 = _blend_post_softmax(attn_weights, mode=mode, eps=eps)
        ba = sig.bind(*args, **kwargs)
        ba.apply_defaults()
        value = ba.arguments["value"]
        attn_output2 = _recompute_attn_output_from_weights(module, w2, value)
        return attn_output2, w2

    return wrapped


def _register_patch_for_modeling_module(pymod: Any) -> None:
    if id(pymod) in _PATCHED_MODULE_IDS:
        return
    if not hasattr(pymod, "eager_attention_forward"):
        return
    orig = pymod.eager_attention_forward
    pymod.eager_attention_forward = _rvt_unified_eager_wrapper(orig)
    _PATCH_STACK.append((pymod, orig))
    _PATCHED_MODULE_IDS.add(id(pymod))


def _prefetch_known_eager_modules() -> None:
    for name in _PREFETCH_MODELING_NAMES:
        try:
            pymod = importlib.import_module(
                f"transformers.models.{name}.modeling_{name}",
            )
        except ImportError:
            continue
        _register_patch_for_modeling_module(pymod)


def _restore_all_eager_patches() -> None:
    for pymod, orig in reversed(_PATCH_STACK):
        pymod.eager_attention_forward = orig
    _PATCH_STACK.clear()
    _PATCHED_MODULE_IDS.clear()


def ensure_rvt_eager_patch_for_attention(attn_module: Any) -> None:
    """当該注意モジュールの ``modeling_*`` にラップ未適用なら適用（セッション内のみ）。"""
    pymod = sys.modules[type(attn_module).__module__]
    if not hasattr(pymod, "eager_attention_forward"):
        return
    with _INJECT_SESSION_LOCK:
        if _INJECT_SESSION_DEPTH == 0:
            raise RuntimeError(
                "rvt ensure_rvt_eager_patch_for_attention: "
                "not inside gpt2_rvt_inject_session",
            )
        _register_patch_for_modeling_module(pymod)


def gpt2_rvt_inject_session_depth() -> int:
    """デバッグ・テスト用: 現在の入れ子深さ（ロック外スナップショット）。"""
    return _INJECT_SESSION_DEPTH


@contextmanager
def gpt2_rvt_inject_session() -> Generator[None, None, None]:
    """
    既知＋遅延対象の ``eager_attention_forward`` を RVT ラップに差し替えるコンテキスト。

    入れ子は深さのみ増減し、内側の終了で原関数に戻さない。
    """
    global _INJECT_SESSION_DEPTH

    with _INJECT_SESSION_LOCK:
        if _INJECT_SESSION_DEPTH == 0:
            _prefetch_known_eager_modules()
        _INJECT_SESSION_DEPTH += 1
    try:
        yield
    finally:
        with _INJECT_SESSION_LOCK:
            _INJECT_SESSION_DEPTH -= 1
            if _INJECT_SESSION_DEPTH < 0:
                raise RuntimeError(
                    "rvt gpt2_rvt_inject_session: depth underflow (bug)",
                )
            if _INJECT_SESSION_DEPTH == 0:
                _restore_all_eager_patches()


def iter_causal_lm_decoder_layers(model: Any) -> list[Any]:
    """``GPT2LMHeadModel`` / ``LlamaForCausalLM`` 等のデコーダ層ブロック列。"""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    return []


def apply_rvt_inject_to_causal_lm(
    model: Any,
    *,
    layer_indices: list[int] | None,
    mode: str,
    eps: float,
) -> None:
    layers = iter_causal_lm_decoder_layers(model)
    if not layers:
        return
    n_layers = len(layers)
    if layer_indices is None:
        use = set(range(n_layers))
    else:
        use = {int(i) for i in layer_indices}
    for li, block in enumerate(layers):
        attn = getattr(block, "self_attn", None) or getattr(
            block,
            "attn",
            None,
        )
        if attn is None:
            continue
        ensure_rvt_eager_patch_for_attention(attn)
        if li in use:
            attn._rvt_inject = {  # noqa: SLF001
                "mode": mode,
                "eps": float(eps),
            }
        elif hasattr(attn, "_rvt_inject"):
            delattr(attn, "_rvt_inject")


def apply_rvt_inject_to_gpt2_attention_modules(
    model: Any,
    *,
    layer_indices: list[int] | None,
    mode: str,
    eps: float,
) -> None:
    """
    GPT2 互換名。内部は :func:`apply_rvt_inject_to_causal_lm`。
    """
    apply_rvt_inject_to_causal_lm(
        model,
        layer_indices=layer_indices,
        mode=mode,
        eps=eps,
    )


def clear_rvt_inject_from_causal_lm(model: Any) -> None:
    for block in iter_causal_lm_decoder_layers(model):
        attn = getattr(block, "self_attn", None) or getattr(
            block,
            "attn",
            None,
        )
        if attn is not None and hasattr(attn, "_rvt_inject"):
            delattr(attn, "_rvt_inject")


def clear_rvt_inject_from_gpt2(model: Any) -> None:
    clear_rvt_inject_from_causal_lm(model)


def _attn_implementation_config(model: Any) -> str | None:
    """
    HF の版差吸収: 公開 ``attn_implementation`` が無い構成でも
    ``_attn_implementation`` / internal を参照する。
    """
    cfg = model.config
    impl = getattr(cfg, "attn_implementation", None)
    if impl is not None:
        return str(impl)
    internal = getattr(cfg, "_attn_implementation", None)
    if internal is not None:
        return str(internal)
    i2 = getattr(cfg, "_attn_implementation_internal", None)
    if i2 is not None:
        return str(i2)
    return None


def resolved_attn_implementation(model: Any) -> str | None:
    """ログ・エラー用: モデル設定から実効 ``attn_implementation`` を返す。"""
    return _attn_implementation_config(model)


def model_supports_rvt_l2_inject(model: Any) -> bool:
    """
    L2 可能: **eager** でロード済みかつ、デコーダ層・注意が取れ、
    当該 ``modeling_*`` に ``eager_attention_forward`` がある。
    """
    impl = _attn_implementation_config(model)
    if impl != "eager":
        return False
    layers = iter_causal_lm_decoder_layers(model)
    if not layers:
        return False
    first = layers[0]
    attn = getattr(first, "self_attn", None) or getattr(first, "attn", None)
    if attn is None:
        return False
    pymod = sys.modules.get(type(attn).__module__)
    if pymod is None or not hasattr(pymod, "eager_attention_forward"):
        return False
    return True


def model_supports_gpt2_rvt_inject(model: Any) -> bool:
    """GPT2LMHeadModel か（後方互換・テスト用）。"""
    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
    except ImportError:
        return False
    return isinstance(model, GPT2LMHeadModel)


def hf_forward_attention_with_rvt_l2(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    text: str,
    layer_index: int,
    mode: str = "base",
    eps: float = 0.05,
    all_layers: bool = False,
) -> tuple[Any, Any, int]:
    """
    Phase II-A 本線用: 層平均注意 ``(L,L)`` を 1 回 forward で取得。

    ``mode=base`` は無介入。
    ``sym`` / ``wasym`` は eager ``eager_attention_forward`` の事後ブレンド。
    """
    from experiments.v7_phase1a_phi_correlation import (
        hf_forward_attention_layer_matrix,
    )

    if mode == "base":
        return hf_forward_attention_layer_matrix(
            model=model,
            tokenizer=tokenizer,
            device=device,
            text=text,
            layer_index=layer_index,
        )
    impl = _attn_implementation_config(model)
    if impl != "eager":
        return None, {
            "schema_version": "v7_phase1a.v1",
            "error": "rvt_l2_requires_attn_implementation_eager",
            "attn_implementation": impl,
            "skipped": True,
        }, 0
    if not model_supports_rvt_l2_inject(model):
        return None, {
            "schema_version": "v7_phase1a.v1",
            "error": "rvt_l2_unsupported_model_structure",
            "skipped": True,
        }, 0
    layers = iter_causal_lm_decoder_layers(model)
    n_layers = len(layers)
    li = layer_index if layer_index >= 0 else n_layers - 1
    li = max(0, min(int(li), n_layers - 1))
    layer_indices: list[int] | None = None if all_layers else [li]
    with gpt2_rvt_inject_session():
        apply_rvt_inject_to_causal_lm(
            model,
            layer_indices=layer_indices,
            mode=mode,
            eps=float(eps),
        )
        try:
            return hf_forward_attention_layer_matrix(
                model=model,
                tokenizer=tokenizer,
                device=device,
                text=text,
                layer_index=layer_index,
            )
        finally:
            clear_rvt_inject_from_causal_lm(model)


def hf_forward_attention_layer_heads_with_rvt_l2(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    text: str,
    layer_index: int,
    mode: str = "base",
    eps: float = 0.05,
    all_layers: bool = False,
) -> tuple[Any, Any, int]:
    """
    指定層の注意を **ヘッド別** ``(H, L, L)`` で取得（L2 は ``sym`` / ``wasym``）。

    ``mode=base`` は ``hf_forward_attention_layer_heads_numpy``
    （`rvt_exp_2026_008_attention`）と同等。
    """
    from experiments.rvt_exp_2026_008_attention import (
        hf_forward_attention_layer_heads_numpy,
    )

    if mode == "base":
        return hf_forward_attention_layer_heads_numpy(
            model=model,
            tokenizer=tokenizer,
            device=device,
            text=text,
            layer_index=layer_index,
        )
    impl = _attn_implementation_config(model)
    if impl != "eager":
        return None, {
            "schema_version": "rvt_exp_2026_008.v1",
            "error": "rvt_l2_requires_attn_implementation_eager",
            "attn_implementation": impl,
            "skipped": True,
        }, 0
    if not model_supports_rvt_l2_inject(model):
        return None, {
            "schema_version": "rvt_exp_2026_008.v1",
            "error": "rvt_l2_unsupported_model_structure",
            "skipped": True,
        }, 0
    layers = iter_causal_lm_decoder_layers(model)
    n_layers = len(layers)
    li = layer_index if layer_index >= 0 else n_layers - 1
    li = max(0, min(int(li), n_layers - 1))
    layer_indices: list[int] | None = None if all_layers else [li]
    with gpt2_rvt_inject_session():
        apply_rvt_inject_to_causal_lm(
            model,
            layer_indices=layer_indices,
            mode=mode,
            eps=float(eps),
        )
        try:
            return hf_forward_attention_layer_heads_numpy(
                model=model,
                tokenizer=tokenizer,
                device=device,
                text=text,
                layer_index=layer_index,
            )
        finally:
            clear_rvt_inject_from_causal_lm(model)
