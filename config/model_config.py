"""
Model configuration classes for INDRA LLM
(c) Divyansh Bharadwaj
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class GQAConfig:
    """Grouped Query Attention configuration"""
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    group_size: int = 4  # num_heads / num_kv_heads

@dataclass
class MOEConfig:
    """Mixture of Experts configuration"""
    num_experts: int = 8
    expert_capacity: int = 2
    top_k: int = 2
    jitter_noise: float = 0.1
    auxiliary_loss_weight: float = 0.01
    expert_hidden_size: Optional[int] = None

@dataclass
class VedicConfig:
    """Vedic integration configuration"""
    use_vedic_core: bool = True
    vedic_memory_size: int = 10000
    vedic_retrieval_top_k: int = 5
    vedic_expert_frozen: bool = True
    vedic_expert_id: int = 0
    vedic_routing_weight: float = 1.5
    dharmic_alignment_weight: float = 0.1

@dataclass
class ModelConfig:
    """Main model configuration"""
    # Model architecture
    vocab_size: int = 262154 ## <============= ORignal: 50257
    n_positions: int = 2048
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    intermediate_size: Optional[int] = None
    
    # MoE configuration
    use_moe: bool = True
    moe: MOEConfig = None
    
    # GQA configuration
    gqa: GQAConfig = None
    
    # Vedic integration
    vedic: VedicConfig = None
    
    # Hierarchical reasoning
    use_hierarchical_reasoning: bool = False
    reasoning_levels: int = 5
    
    # Efficiency features
    use_flash_attention: bool = True
    use_kv_cache: bool = True
    gradient_checkpointing: bool = False
    
    # Activation and normalization
    activation_function: str = "gelu"
    layer_norm_eps: float = 1e-5
    use_rms_norm: bool = True
    
    # Regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    
    # Initialization
    initializer_range: float = 0.02
    
    # Rope settings
    use_rope: bool = True
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # KV cache settings
    max_cache_length: int = 4096
    
    def __post_init__(self):
        if self.moe is None:
            self.moe = MOEConfig()
        if self.gqa is None:
            self.gqa = GQAConfig(num_heads=self.n_head, num_kv_heads=self.n_kv_head)
        if self.vedic is None:
            self.vedic = VedicConfig()
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.n_embd
        
        # Validate configurations
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.gqa.group_size = self.n_head // self.n_kv_head
        self.gqa.head_dim = self.n_embd // self.n_head
        
        if self.moe.expert_hidden_size is None:
            self.moe.expert_hidden_size = self.intermediate_size
