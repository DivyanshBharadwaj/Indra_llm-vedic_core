"""
Attention mechanisms for INDRA LLM including FlashAttention and Grouped Query Attention
(c) Divyansh Bharadwaj
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torch import Tensor

def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
    """Applies Rotary Position Embedding to query and key tensors."""
    cos = cos[position_ids].unsqueeze(1)  # [seq_len, 1, head_dim]
    sin = sin[position_ids].unsqueeze(1)  # [seq_len, 1, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryPositionalEmbedding(nn.Module):
    """RoPE implementation for positional encoding."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    def forward(self, seq_len: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Generate cos and sin embeddings."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention implementation with FlashAttention support."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.group_size = self.n_head // self.n_kv_head
        
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        self.use_flash_attention = config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
        
        if config.use_rope:
            self.rotary_emb = RotaryPositionalEmbedding(
                self.head_dim,
                config.n_positions,
                config.rope_theta
            )
        else:
            self.rotary_emb = None
            
    def repeat_kv(self, hidden_states: Tensor) -> Tensor:
        """Repeat k/v heads to match number of query heads."""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if num_key_value_heads == self.n_head:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, self.group_size, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * self.group_size, slen, head_dim)
        
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to q, k, v
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if self.rotary_emb is not None:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=hidden_states.device)
                
            # --- START OF FIX ---
            # Determine the past sequence length from the cache to create a correctly sized RoPE table.
            past_seq_len = 0
            if past_key_value is not None:
                # cache_k has shape [batch, num_kv_heads, seq_len, head_dim]
                past_seq_len = past_key_value[0].shape[2]
            
            # Generate embeddings for the full sequence length (cache + new tokens)
            total_seq_len = past_seq_len + seq_len
            cos, sin = self.rotary_emb(total_seq_len, hidden_states.device)
            # --- END OF FIX ---
            
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # Handle KV cache
        if past_key_value is not None:
            cache_k, cache_v = past_key_value
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)
        
        if use_cache:
            present = (k, v)
        else:
            present = None
            
        # Repeat k/v for grouped attention
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)
        
        # Compute attention
        if self.use_flash_attention:
            # Use PyTorch's flash attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                is_causal=attention_mask is None
            )
        else:
            # Standard attention computation
            attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = self.dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.n_head * self.head_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present

class FlashAttentionLayer(nn.Module):
    """Standalone FlashAttention layer wrapper."""
    
    def __init__(self, config):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        
    def forward(self, *args, **kwargs):
        return self.attention(*args, **kwargs)
