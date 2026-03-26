"""Resonanceverse コア（PyTorch）。

理論の正本: ``docs/v7/Resonanceverse_Theory_v7.0.md``（有向遅延共鳴場 DDRF）。
本パッケージは段階的実装であり、厳密な DDE・写像 φ・遅延監視のすべてが
揃っているとは限らない。索引: ``docs/theory/resonanceverse_theory.md``。
"""
from .autopoiesis import AutopoieticInference
from .cultural_modulation import CulturalModulationAdapter, awai_pressure_from_embeddings
from .config_utils import autopoietic_kwargs, load_yaml_config, resonant_core_kwargs
from .lightweight_resonance import LightweightResonanceFacade
from .instrumentation import StageTimer
from .reproducibility import set_experiment_seed
from .resonance import ResonanceEngine
from .resonant_core import AwaiIntegratedSLM, ResonantCore
from .roi_selector import DynamicROISelector

__all__ = [
    "AutopoieticInference",
    "AwaiIntegratedSLM",
    "CulturalModulationAdapter",
    "awai_pressure_from_embeddings",
    "DynamicROISelector",
    "LightweightResonanceFacade",
    "ResonanceEngine",
    "ResonantCore",
    "StageTimer",
    "set_experiment_seed",
    "autopoietic_kwargs",
    "load_yaml_config",
    "resonant_core_kwargs",
]
