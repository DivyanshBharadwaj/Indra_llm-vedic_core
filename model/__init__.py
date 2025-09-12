"""
Model module for INDRA - Vedic-aligned Decoder-only Transformer LLM
(c) Divyansh Bharadwaj
"""

from .transformer import INDRATransformer
from .attention import GroupedQueryAttention, FlashAttentionLayer
from .moe import MixtureOfExperts, Expert
from .vedic_core import VedicExpert, VedicMemoryAdapter, VedicRewardModel
from .kv_cache import KVCache, KVCacheManager
from .hierarchical_reasoning import (
    HierarchicalReasoningModule, 
    VedicReasoningCoordinator,
    add_hierarchical_reasoning_to_model
)

__all__ = [
    'INDRATransformer',
    'GroupedQueryAttention',
    'FlashAttentionLayer',
    'MixtureOfExperts',
    'Expert',
    'VedicExpert',
    'VedicMemoryAdapter', 
    'VedicRewardModel',
    'KVCache',
    'KVCacheManager',
    'HierarchicalReasoningModule',
    'VedicReasoningCoordinator',
    'add_hierarchical_reasoning_to_model'
]
