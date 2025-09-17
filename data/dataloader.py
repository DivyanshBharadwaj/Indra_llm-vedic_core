"""
Efficient data loading for INDRA LLM with batching and collation
(c) Divyansh Bharadwaj
"""

import torch
from torch.utils.data import DataLoader, Sampler
from typing import List, Dict, Optional, Union, Any
import random
import logging

class INDRADataLoader:
    """Custom data loader with advanced batching strategies."""
    
    def __init__(
        self,
        dataset,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory=torch.cuda.is_available(),
        drop_last: bool = True,
        collate_fn: Optional[callable] = None,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Sampler] = None,
        persistent_workers: bool = False,
    ):
        """
        Initialize INDRA data loader.
        
        Args:
            dataset: Dataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
            collate_fn: Custom collation function
            sampler: Custom sampler
            batch_sampler: Custom batch sampler
            persistent_workers: Whether to keep workers alive
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Use custom collate function if not provided
        if collate_fn is None:
            collate_fn = self._default_collate_fn
        
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
            sampler=sampler,
            batch_sampler=batch_sampler,
            persistent_workers=persistent_workers,
        )
    
    # REPLACE the old _default_collate_fn method in dataloader.py with this one

    def _default_collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Default collation function for batching that handles inconsistent keys."""
        if not batch:
            return {}
        
        # Get a union of all keys present in the batch
        all_keys = set()
        for item in batch:
            all_keys.update(item.keys())
            
        result = {}
        
        for key in all_keys:
            # For tensor keys, we need to handle missing items robustly
            if key in ['input_ids', 'attention_mask', 'labels', 'vedic_weights']:
                # Find the first item that has the key to determine the shape and dtype
                present_item = next((item for item in batch if key in item), None)
                if present_item is None: continue # Skip if key is not in any item
                
                example_tensor = present_item[key]
                default_tensor = torch.zeros_like(example_tensor)
                
                values = [item.get(key, default_tensor) for item in batch]
                result[key] = torch.stack(values)
            
            # For other non-tensor metadata
            else:
                values = [item.get(key, None) for item in batch]
                result[key] = values

        # Ensure essential keys are present
        for essential_key in ['input_ids', 'attention_mask', 'labels']:
            if essential_key not in result:
                dummy_tensor = torch.zeros(len(batch), self.dataset.max_length, dtype=torch.long)
                result[essential_key] = dummy_tensor

        return result
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)

class LanguageAwareSampler(Sampler):
    """Sampler that ensures balanced language representation in batches."""
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        language_weights: Optional[Dict[str, float]] = None,
        shuffle: bool = True,
    ):
        """
        Initialize language-aware sampler.
        
        Args:
            dataset: MultiLanguageDataset instance
            batch_size: Batch size
            language_weights: Weights for different languages
            shuffle: Whether to shuffle within languages
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.language_weights = language_weights or {}
        self.shuffle = shuffle
        
        # Check if dataset has language indices
        if not hasattr(dataset, 'language_indices'):
            raise ValueError("Dataset must have language_indices attribute")
        
        self.language_indices = dataset.language_indices
        self.languages = list(self.language_indices.keys())
        
    def __iter__(self):
        # Get sampling weights
        weights = []
        for lang in self.languages:
            weight = self.language_weights.get(lang, 1.0)
            weights.append(weight)
        
        # Generate batches
        num_batches = len(self.dataset) // self.batch_size
        
        for _ in range(num_batches):
            batch_indices = []
            
            # Sample languages for this batch
            for _ in range(self.batch_size):
                # Choose language based on weights
                chosen_lang = random.choices(self.languages, weights=weights, k=1)[0]
                
                # Choose random sample from chosen language
                lang_indices = self.language_indices[chosen_lang]
                batch_indices.append(random.choice(lang_indices))
            
            if self.shuffle:
                random.shuffle(batch_indices)
            
            yield batch_indices
    
    def __len__(self):
        return len(self.dataset) // self.batch_size

# class VedicPrioritySampler(Sampler):
#     """Sampler that prioritizes Vedic content during curriculum learning."""
    
#     def __init__(
#         self,
#         dataset,
#         batch_size: int,
#         vedic_ratio: float = 0.5,
#         phase: str = 'mixed',  # 'vedic', 'general', 'mixed'
#         shuffle: bool = True,
#     ):
#         """
#         Initialize Vedic priority sampler.
        
#         Args:
#             dataset: Dataset instance
#             batch_size: Batch size
#             vedic_ratio: Ratio of Vedic content in mixed phase
#             phase: Training phase ('vedic', 'general', 'mixed')
#             shuffle: Whether to shuffle indices
#         """
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.vedic_ratio = vedic_ratio
#         self.phase = phase
#         self.shuffle = shuffle
        
#         # Identify Vedic vs general content
#         self.vedic_indices = []
#         self.general_indices = []
        
#         for i in range(len(dataset)):
#             example = dataset.examples[i]
#             if self._is_vedic_content(example):
#                 self.vedic_indices.append(i)
#             else:
#                 self.general_indices.append(i)
        
#         logging.info(f"VedicPrioritySampler: {len(self.vedic_indices)} Vedic, {len(self.general_indices)} general examples")
    
#     def _is_vedic_content(self, example: Dict) -> bool:
#         """Determine if example is Vedic content."""
#         # Check for Vedic markers in text
#         text = example.get('text', '').lower()
        
#         vedic_keywords = [
#             'dharma', 'karma', 'moksha', 'brahman', 'atman',
#             'veda', 'upanishad', 'rigveda', 'yajurveda', 'samaveda', 'atharvaveda',
#             'sanskrit', 'mantra', 'sloka', 'yoga', 'ahimsa', 'satya'
#         ]
        
#         for keyword in vedic_keywords:
#             if keyword in text:
#                 return True
        
#         # Check for language markers
#         if hasattr(example, 'language') and example.get('language') == 'sanskrit':
#             return True
        
#         # Check for is_vedic flag
#         if example.get('is_vedic', False):
#             return True
        
#         return False
    
#     def __iter__(self):
#         if self.phase == 'vedic':
#             # Only Vedic content
#             indices = self.vedic_indices.copy()
#         elif self.phase == 'general':
#             # Only general content
#             indices = self.general_indices.copy()
#         else:
#             # Mixed content with specified ratio
#             num_vedic = int(len(self.dataset) * self.vedic_ratio)
#             num_general = len(self.dataset) - num_vedic
            
#             vedic_samples = random.choices(self.vedic_indices, k=min(num_vedic, len(self.vedic_indices)))
#             general_samples = random.choices(self.general_indices, k=min(num_general, len(self.general_indices)))
            
#             indices = vedic_samples + general_samples
        
#         if self.shuffle:
#             random.shuffle(indices)
        
#         # Yield batches
#         for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
#             yield indices[i:i + self.batch_size]
    
#     def __len__(self):
#         if self.phase == 'vedic':
#             return len(self.vedic_indices) // self.batch_size
#         elif self.phase == 'general':
#             return len(self.general_indices) // self.batch_size
#         else:
#             return len(self.dataset) // self.batch_size

class VedicPrioritySampler(Sampler):
    """Sampler that prioritizes Vedic content during curriculum learning."""
    
    def __init__(
        self,
        dataset,
        vedic_ratio: float = 0.5,
        phase: str = 'mixed',  # 'vedic', 'general', 'mixed'
        shuffle: bool = True,
    ):
        """
        Initialize Vedic priority sampler.
        
        Args:
            dataset: Dataset instance
            vedic_ratio: Ratio of Vedic content in mixed phase
            phase: Training phase ('vedic', 'general', 'mixed')
            shuffle: Whether to shuffle indices
        """
        self.dataset = dataset
        self.vedic_ratio = vedic_ratio
        self.phase = phase
        self.shuffle = shuffle
        
        # Identify Vedic vs general content
        self.vedic_indices = []
        self.general_indices = []
        
        for i in range(len(dataset)):
            example = dataset.examples[i]
            if self._is_vedic_content(example):
                self.vedic_indices.append(i)
            else:
                self.general_indices.append(i)
        
        logging.info(f"VedicPrioritySampler: {len(self.vedic_indices)} Vedic, {len(self.general_indices)} general examples")
    
    def _is_vedic_content(self, example: dict) -> bool:
        """Determine if example is Vedic content."""
        # Check for Vedic markers in text
        text = example.get('text', '').lower()
        
        vedic_keywords = [
            'dharma', 'karma', 'moksha', 'brahman', 'atman',
            'veda', 'upanishad', 'rigveda', 'yajurveda', 'samaveda', 'atharvaveda',
            'sanskrit', 'mantra', 'sloka', 'yoga', 'ahimsa', 'satya'
        ]
        
        for keyword in vedic_keywords:
            if keyword in text:
                return True
        
        # Check for language markers
        if hasattr(example, 'language') and example.get('language') == 'sanskrit':
            return True
        
        # Check for is_vedic flag
        if example.get('is_vedic', False):
            return True
        
        return False
    
    def __iter__(self):
        """Yield individual indices, not batches."""
        if self.phase == 'vedic':
            # Only Vedic content
            indices = self.vedic_indices.copy()
        elif self.phase == 'general':
            # Only general content
            indices = self.general_indices.copy()
        else:
            # Mixed content with specified ratio
            total_samples = len(self.dataset)
            num_vedic = int(total_samples * self.vedic_ratio)
            num_general = total_samples - num_vedic
            
            # Sample with replacement if needed
            vedic_samples = random.choices(
                self.vedic_indices, 
                k=min(num_vedic, len(self.vedic_indices) * 10)  # Allow oversampling
            )[:num_vedic]
            
            general_samples = random.choices(
                self.general_indices, 
                k=min(num_general, len(self.general_indices) * 10)  # Allow oversampling
            )[:num_general]
            
            indices = vedic_samples + general_samples
        
        if self.shuffle:
            random.shuffle(indices)
        
        # Yield individual indices
        for idx in indices:
            yield idx
    
    def __len__(self):
        return len(self.dataset)

class DynamicBatchSampler(Sampler):
    """Dynamic batch sampler that groups samples by sequence length."""
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        max_tokens_per_batch: Optional[int] = None,
        drop_last: bool = True,
        shuffle: bool = True,
    ):
        """
        Initialize dynamic batch sampler.
        
        Args:
            dataset: Dataset instance
            batch_size: Maximum batch size
            max_tokens_per_batch: Maximum tokens per batch
            drop_last: Whether to drop last incomplete batch
            shuffle: Whether to shuffle batches
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # Pre-compute sequence lengths
        self.lengths = []
        for i in range(len(dataset)):
            example = dataset[i]
            if isinstance(example['input_ids'], torch.Tensor):
                length = (example['input_ids'] != dataset.tokenizer.pad_token_id).sum().item()
            else:
                length = len(example['input_ids'])
            self.lengths.append((i, length))
        
        # Sort by length for efficient batching
        self.lengths.sort(key=lambda x: x[1])
    
    def __iter__(self):
        # Create batches
        batches = []
        current_batch = []
        current_tokens = 0
        
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)
        
        for idx in indices:
            sample_idx, sample_length = self.lengths[idx]
            
            # Check if adding this sample would exceed limits
            would_exceed_size = len(current_batch) >= self.batch_size
            would_exceed_tokens = (
                self.max_tokens_per_batch and 
                current_tokens + sample_length > self.max_tokens_per_batch
            )
            
            if would_exceed_size or (would_exceed_tokens and current_batch):
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [sample_idx]
                current_tokens = sample_length
            else:
                # Add to current batch
                current_batch.append(sample_idx)
                current_tokens += sample_length
        
        # Handle last batch
        if current_batch and not self.drop_last:
            batches.append(current_batch)
        
        # Shuffle batches if requested
        if self.shuffle:
            random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        # Approximate number of batches
        total_samples = len(self.dataset)
        if self.max_tokens_per_batch:
            # Estimate based on average sequence length
            avg_length = sum(length for _, length in self.lengths) / len(self.lengths)
            samples_per_batch = min(self.batch_size, self.max_tokens_per_batch // avg_length)
            return total_samples // max(1, int(samples_per_batch))
        else:
            return total_samples // self.batch_size

def create_dataloader(
    dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    sampler_type: str = 'default',
    **kwargs
) -> INDRADataLoader:
    """
    Create appropriate data loader based on dataset type.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        sampler_type: Type of sampler ('default', 'language_aware', 'vedic_priority', 'dynamic')
        **kwargs: Additional arguments for sampler
        
    Returns:
        Configured data loader
    """
    sampler = None
    
    if sampler_type == 'language_aware' and hasattr(dataset, 'language_indices'):
        sampler = LanguageAwareSampler(
            dataset, 
            batch_size, 
            language_weights=kwargs.get('language_weights'),
            shuffle=shuffle
        )
        shuffle = False  # Sampler handles shuffling
    
    elif sampler_type == 'vedic_priority':
        sampler = VedicPrioritySampler(
            dataset=dataset,
            vedic_ratio=kwargs.get('vedic_ratio', 0.5),
            phase=kwargs.get('phase', 'mixed'),
            shuffle=shuffle
        )
        shuffle = False
    
    elif sampler_type == 'dynamic':
        sampler = DynamicBatchSampler(
            dataset,
            batch_size,
            max_tokens_per_batch=kwargs.get('max_tokens_per_batch'),
            drop_last=kwargs.get('drop_last', True),
            shuffle=shuffle
        )
        shuffle = False
        batch_size = 1  # DynamicBatchSampler handles batching
    
    return INDRADataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        **{k: v for k, v in kwargs.items() if k not in ['language_weights', 'vedic_ratio', 'phase', 'max_tokens_per_batch']}
    )
