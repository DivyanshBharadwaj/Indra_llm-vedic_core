"""
KV caching system for efficient inference in INDRA LLM
(c) Divyansh Bharadwaj
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from torch import Tensor

class KVCache:
    """Efficient key-value cache for single sequence."""
    
    def __init__(
        self, 
        max_batch_size: int,
        max_seq_length: int, 
        num_heads: int,
        head_dim: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        
        # Initialize cache tensors
        self.keys = torch.zeros(
            (num_layers, max_batch_size, num_heads, max_seq_length, head_dim),
            device=device,
            dtype=dtype
        )
        self.values = torch.zeros(
            (num_layers, max_batch_size, num_heads, max_seq_length, head_dim),
            device=device,
            dtype=dtype
        )
        
        # Track current sequence lengths for each batch item
        self.seq_lengths = torch.zeros(max_batch_size, dtype=torch.long, device=device)
        
    def update(
        self, 
        layer_idx: int,
        batch_idx: int, 
        keys: Tensor, 
        values: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Update cache with new key-value pairs.
        
        Args:
            layer_idx: Layer index
            batch_idx: Batch index
            keys: New keys [1, num_heads, seq_len, head_dim]
            values: New values [1, num_heads, seq_len, head_dim]
            
        Returns:
            - cached_keys: All keys up to current position
            - cached_values: All values up to current position
        """
        seq_len = keys.size(2)
        start_pos = self.seq_lengths[batch_idx].item()
        end_pos = start_pos + seq_len
        
        if end_pos > self.max_seq_length:
            raise ValueError(f"Sequence length {end_pos} exceeds maximum {self.max_seq_length}")
        
        # Update cache
        self.keys[layer_idx, batch_idx, :, start_pos:end_pos, :] = keys[0]
        self.values[layer_idx, batch_idx, :, start_pos:end_pos, :] = values[0]
        
        # Update sequence length
        self.seq_lengths[batch_idx] = end_pos
        
        # Return all cached keys and values up to current position
        cached_keys = self.keys[layer_idx, batch_idx:batch_idx+1, :, :end_pos, :]
        cached_values = self.values[layer_idx, batch_idx:batch_idx+1, :, :end_pos, :]
        
        return cached_keys, cached_values
    
    def get(self, layer_idx: int, batch_idx: int) -> Tuple[Tensor, Tensor]:
        """Get cached keys and values for a specific layer and batch item."""
        seq_len = self.seq_lengths[batch_idx].item()
        keys = self.keys[layer_idx, batch_idx:batch_idx+1, :, :seq_len, :]
        values = self.values[layer_idx, batch_idx:batch_idx+1, :, :seq_len, :]
        return keys, values
    
    def clear(self, batch_idx: Optional[int] = None):
        """Clear cache for specific batch item or entire cache."""
        if batch_idx is not None:
            self.keys[:, batch_idx, :, :, :].zero_()
            self.values[:, batch_idx, :, :, :].zero_()
            self.seq_lengths[batch_idx] = 0
        else:
            self.keys.zero_()
            self.values.zero_()
            self.seq_lengths.zero_()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        total_elements = self.keys.numel() + self.values.numel()
        element_size = self.keys.element_size()  # bytes per element
        total_bytes = total_elements * element_size
        
        return {
            'total_mb': total_bytes / (1024 * 1024),
            'keys_mb': self.keys.numel() * element_size / (1024 * 1024),
            'values_mb': self.values.numel() * element_size / (1024 * 1024),
            'utilization': (self.seq_lengths.sum().item() / (self.max_batch_size * self.max_seq_length)).item()
        }

class KVCacheManager:
    """Manages multiple KV caches and provides batch processing."""
    
    def __init__(self, config):
        self.config = config
        self.max_cache_length = config.max_cache_length
        self.caches: Dict[str, KVCache] = {}
        self.active_batches: Dict[str, int] = {}
        
    def create_cache(
        self,
        cache_id: str,
        max_batch_size: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16
    ) -> KVCache:
        """Create a new KV cache."""
        cache = KVCache(
            max_batch_size=max_batch_size,
            max_seq_length=self.max_cache_length,
            num_heads=num_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            device=device,
            dtype=dtype
        )
        self.caches[cache_id] = cache
        self.active_batches[cache_id] = 0
        return cache
    
    def get_cache(self, cache_id: str) -> Optional[KVCache]:
        """Get existing cache by ID."""
        return self.caches.get(cache_id)
    
    def update_cache(
        self,
        cache_id: str,
        layer_idx: int,
        keys: Tensor,
        values: Tensor,
        batch_indices: Optional[List[int]] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Update cache with new key-value pairs for multiple batch items.
        
        Args:
            cache_id: Cache identifier
            layer_idx: Layer index
            keys: New keys [batch_size, num_heads, seq_len, head_dim]
            values: New values [batch_size, num_heads, seq_len, head_dim]
            batch_indices: Specific batch indices to update
            
        Returns:
            - all_keys: Concatenated keys including cache
            - all_values: Concatenated values including cache
        """
        cache = self.get_cache(cache_id)
        if cache is None:
            raise ValueError(f"Cache {cache_id} not found")
        
        batch_size = keys.size(0)
        if batch_indices is None:
            batch_indices = list(range(batch_size))
        
        all_keys = []
        all_values = []
        
        for i, batch_idx in enumerate(batch_indices):
            # Get keys and values for this batch item
            batch_keys = keys[i:i+1]  # [1, num_heads, seq_len, head_dim]
            batch_values = values[i:i+1]  # [1, num_heads, seq_len, head_dim]
            
            # Update cache and get all cached keys/values
            cached_keys, cached_values = cache.update(
                layer_idx, batch_idx, batch_keys, batch_values
            )
            
            all_keys.append(cached_keys)
            all_values.append(cached_values)
        
        # Concatenate all keys and values
        # Note: sequences might have different lengths, so we handle this carefully
        max_seq_len = max(k.size(2) for k in all_keys)
        
        # Pad sequences to same length
        padded_keys = []
        padded_values = []
        
        for k, v in zip(all_keys, all_values):
            if k.size(2) < max_seq_len:
                pad_length = max_seq_len - k.size(2)
                k = torch.cat([k, torch.zeros_like(k[:, :, :pad_length, :])], dim=2)
                v = torch.cat([v, torch.zeros_like(v[:, :, :pad_length, :])], dim=2)
            padded_keys.append(k)
            padded_values.append(v)
        
        final_keys = torch.cat(padded_keys, dim=0)
        final_values = torch.cat(padded_values, dim=0)
        
        return final_keys, final_values
    
    def clear_cache(self, cache_id: str, batch_indices: Optional[List[int]] = None):
        """Clear specific cache or batch items."""
        cache = self.get_cache(cache_id)
        if cache is None:
            return
        
        if batch_indices is None:
            cache.clear()
        else:
            for batch_idx in batch_indices:
                cache.clear(batch_idx)
    
    def get_total_memory_usage(self) -> Dict[str, float]:
        """Get total memory usage across all caches."""
        total_usage = {
            'total_mb': 0.0,
            'keys_mb': 0.0,
            'values_mb': 0.0,
            'avg_utilization': 0.0,
            'num_caches': len(self.caches)
        }
        
        if not self.caches:
            return total_usage
        
        utilizations = []
        for cache in self.caches.values():
            usage = cache.get_memory_usage()
            total_usage['total_mb'] += usage['total_mb']
            total_usage['keys_mb'] += usage['keys_mb']
            total_usage['values_mb'] += usage['values_mb']
            utilizations.append(usage['utilization'])
        
        total_usage['avg_utilization'] = sum(utilizations) / len(utilizations)
        
        return total_usage
    
    def cleanup_unused_caches(self):
        """Remove caches that are no longer needed."""
        to_remove = []
        for cache_id, cache in self.caches.items():
            usage = cache.get_memory_usage()
            if usage['utilization'] == 0.0:  # No active sequences
                to_remove.append(cache_id)
        
        for cache_id in to_remove:
            del self.caches[cache_id]
            if cache_id in self.active_batches:
                del self.active_batches[cache_id]

# Global cache manager instance
_global_cache_manager: Optional[KVCacheManager] = None

def get_cache_manager(config=None) -> KVCacheManager:
    """Get global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        if config is None:
            raise ValueError("Config must be provided for first initialization")
        _global_cache_manager = KVCacheManager(config)
    return _global_cache_manager

def clear_global_cache():
    """Clear global cache manager."""
    global _global_cache_manager
    if _global_cache_manager:
        for cache_id in list(_global_cache_manager.caches.keys()):
            _global_cache_manager.clear_cache(cache_id)
    _global_cache_manager = None
