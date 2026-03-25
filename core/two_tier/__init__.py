"""Phase 3: 二階建てスタブと品質 τ フック。"""

from .causal_lm_layers import get_decoder_module_list
from .gpt2_identity_skip import (
    IdentityGPT2Block,
    decoder_layer_skip_context,
    gpt2_layer_skip_context,
)
from .quality import QualityTauResult, check_quality_tau, quality_report_jsonable
from .stubs import BlockRouterStub, SequenceControllerStub, router_keep_fraction

__all__ = [
    "BlockRouterStub",
    "SequenceControllerStub",
    "router_keep_fraction",
    "IdentityGPT2Block",
    "decoder_layer_skip_context",
    "get_decoder_module_list",
    "gpt2_layer_skip_context",
    "QualityTauResult",
    "check_quality_tau",
    "quality_report_jsonable",
]
