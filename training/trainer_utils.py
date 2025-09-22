"""
Training utilities for INDRA LLM with streaming dataset support
(c) Divyansh Bharadwaj
"""

import os
import json
import logging
import math
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, LinearLR
import wandb

class MetricsTracker:
    """Track training metrics and statistics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}
        self.step_counts = {}
    
    def update(self, metrics: Dict[str, Any], step: int):
        """Update metrics for current step."""
        for key, value in metrics.items():
            # Only track numeric values in metrics history
            if isinstance(value, (int, float)):
                if key not in self.metrics:
                    self.metrics[key] = []
                    self.step_counts[key] = []
                
                self.metrics[key].append(value)
                self.step_counts[key].append(step)
                
                # Keep only recent values
                if len(self.metrics[key]) > self.window_size:
                    self.metrics[key] = self.metrics[key][-self.window_size:]
                    self.step_counts[key] = self.step_counts[key][-self.window_size:]
    
    def get_average(self, key: str, steps: Optional[int] = None) -> float:
        """Get average value for a metric."""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        
        values = self.metrics[key]
        if steps is not None:
            values = values[-steps:]
        
        return sum(values) / len(values)
    
    def get_latest(self, key: str) -> float:
        """Get latest value for a metric."""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return self.metrics[key][-1]
    
    def get_trend(self, key: str, steps: int = 10) -> str:
        """Get trend direction for a metric."""
        if key not in self.metrics or len(self.metrics[key]) < steps:
            return "insufficient_data"
        
        values = self.metrics[key][-steps:]
        first_half = values[:steps//2]
        second_half = values[steps//2:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        diff_ratio = (avg_second - avg_first) / abs(avg_first) if avg_first != 0 else 0
        
        if diff_ratio > 0.01:
            return "increasing"
        elif diff_ratio < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics."""
        summary = {}
        for key in self.metrics:
            if self.metrics[key]:
                summary[key] = {
                    'current': self.get_latest(key),
                    'average': self.get_average(key),
                    'min': min(self.metrics[key]),
                    'max': max(self.metrics[key]),
                    'count': len(self.metrics[key])
                }
        return summary

class StreamingMetricsTracker(MetricsTracker):
    """Enhanced metrics tracker for streaming datasets."""
    
    def __init__(self, window_size: int = 100):
        super().__init__(window_size)
        self.streaming_stats = {
            'examples_processed': 0,
            'batches_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'buffer_overflows': 0
        }
    
    def update_streaming_stats(self, **kwargs):
        """Update streaming-specific statistics."""
        for key, value in kwargs.items():
            if key in self.streaming_stats:
                self.streaming_stats[key] += value
    
    def get_streaming_summary(self) -> Dict[str, Any]:
        """Get summary of streaming statistics."""
        summary = self.get_summary()
        summary['streaming'] = self.streaming_stats.copy()
        
        # Calculate derived metrics
        if self.streaming_stats['batches_processed'] > 0:
            summary['streaming']['avg_examples_per_batch'] = (
                self.streaming_stats['examples_processed'] / 
                self.streaming_stats['batches_processed']
            )
        
        if (self.streaming_stats['cache_hits'] + self.streaming_stats['cache_misses']) > 0:
            summary['streaming']['cache_hit_rate'] = (
                self.streaming_stats['cache_hits'] / 
                (self.streaming_stats['cache_hits'] + self.streaming_stats['cache_misses'])
            )
        
        return summary

class TrainerUtils:
    """Utility functions for training with streaming dataset support."""
    
    @staticmethod
    def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
        """Setup logging configuration."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        handlers = [logging.StreamHandler()]
        if log_file:
            handlers.append(logging.FileHandler(log_file))
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=handlers
        )
    
    @staticmethod
    def setup_wandb(config: Dict[str, Any], project: str, run_name: Optional[str] = None):
        """Setup Weights & Biases logging."""
        if not config.get('use_wandb', False):
            return None
        
        try:
            wandb.init(
                project=project,
                name=run_name,
                config=config
            )
            return wandb
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {e}")
            return None
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated
            }
        return {}
    
    @staticmethod
    def get_streaming_dataset_info(dataset) -> Dict[str, Any]:
        """Get information about streaming dataset."""
        info = {}
        
        if hasattr(dataset, 'total_files'):
            info['total_files'] = dataset.total_files
        if hasattr(dataset, 'batch_size_mb'):
            info['batch_size_mb'] = dataset.batch_size_mb
        if hasattr(dataset, 'examples_per_batch'):
            info['examples_per_batch'] = dataset.examples_per_batch
        if hasattr(dataset, 'shuffle_buffer_size'):
            info['shuffle_buffer_size'] = dataset.shuffle_buffer_size
        if hasattr(dataset, '_shuffle_buffer'):
            info['current_buffer_size'] = len(dataset._shuffle_buffer)
        if hasattr(dataset, 'cache_dir'):
            info['cache_dir'] = dataset.cache_dir
        
        return info
    
    @staticmethod
    def config_to_dict(config) -> Dict[str, Any]:
        """Convert config object to dictionary for JSON serialization."""
        if hasattr(config, '__dict__'):
            # Handle custom config objects
            config_dict = {}
            for key, value in config.__dict__.items():
                if hasattr(value, '__dict__'):
                    # Recursively convert nested config objects
                    config_dict[key] = TrainerUtils.config_to_dict(value)
                elif isinstance(value, (str, int, float, bool, type(None))):
                    # Basic types that are JSON serializable
                    config_dict[key] = value
                elif isinstance(value, (list, tuple)):
                    # Handle lists/tuples of basic types
                    config_dict[key] = [
                        TrainerUtils.config_to_dict(item) if hasattr(item, '__dict__') else item
                        for item in value
                    ]
                elif isinstance(value, dict):
                    # Handle dictionaries
                    config_dict[key] = {
                        k: TrainerUtils.config_to_dict(v) if hasattr(v, '__dict__') else v
                        for k, v in value.items()
                    }
                else:
                    # Convert other types to string representation
                    config_dict[key] = str(value)
            return config_dict
        elif isinstance(config, dict):
            # Already a dictionary, but check nested values
            return {
                k: TrainerUtils.config_to_dict(v) if hasattr(v, '__dict__') else v
                for k, v in config.items()
            }
        else:
            # Return as-is for basic types
            return config
    
    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        step: int,
        loss: float,
        checkpoint_dir: str,
        config: Dict[str, Any],
        save_format: str = "both",  # "pt", "safetensors", "both"
        streaming_dataset_info: Optional[Dict[str, Any]] = None
    ):
        """Save model checkpoint with streaming dataset info."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert config to JSON-serializable format
        serializable_config = TrainerUtils.config_to_dict(config)
        
        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'step': step,
            'loss': loss,
            'config': serializable_config,
        }
        
        # Add streaming dataset info if available
        if streaming_dataset_info:
            checkpoint['streaming_info'] = streaming_dataset_info
        
        # Save in PyTorch format
        if save_format in ["pt", "both"]:
            pt_path = checkpoint_dir / f"checkpoint-step-{step}.pt"
            torch.save(checkpoint, pt_path)
            logging.info(f"Saved PyTorch checkpoint to {pt_path}")
        
        # Save in SafeTensors format
        if save_format in ["safetensors", "both"]:
            try:
                from safetensors.torch import save_file
                
                # Flatten state dicts for safetensors
                tensors = {}
                
                # Model state dict
                for key, tensor in model.state_dict().items():
                    tensors[f"model.{key}"] = tensor
                
                # Optimizer state dict (only tensors)
                for key, value in optimizer.state_dict().items():
                    if isinstance(value, torch.Tensor):
                        tensors[f"optimizer.{key}"] = value
                    elif isinstance(value, dict):
                        for subkey, subtensor in value.items():
                            if isinstance(subtensor, torch.Tensor):
                                tensors[f"optimizer.{key}.{subkey}"] = subtensor
                
                # Metadata
                metadata = {
                    'step': str(step),
                    'loss': str(loss),
                    'config': json.dumps(serializable_config)
                }
                
                # Add streaming info to metadata
                if streaming_dataset_info:
                    metadata['streaming_info'] = json.dumps(streaming_dataset_info)
                
                safetensors_path = checkpoint_dir / f"checkpoint-step-{step}.safetensors"
                save_file(tensors, safetensors_path, metadata=metadata)
                logging.info(f"Saved SafeTensors checkpoint to {safetensors_path}")
                
            except ImportError:
                logging.warning("SafeTensors not available, skipping safetensors save")
    
    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.suffix == '.safetensors':
            return TrainerUtils._load_safetensors_checkpoint(
                checkpoint_path, model, optimizer, scheduler, device
            )
        else:
            return TrainerUtils._load_pytorch_checkpoint(
                checkpoint_path, model, optimizer, scheduler, device
            )
    
    @staticmethod
    def _load_pytorch_checkpoint(
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: Optional[torch.device]
    ) -> Dict[str, Any]:
        """Load PyTorch checkpoint."""
        map_location = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return {
            'step': checkpoint.get('step', 0),
            'loss': checkpoint.get('loss', float('inf')),
            'config': checkpoint.get('config', {}),
            'streaming_info': checkpoint.get('streaming_info', {})
        }
    
    @staticmethod
    def _load_safetensors_checkpoint(
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: Optional[torch.device]
    ) -> Dict[str, Any]:
        """Load SafeTensors checkpoint."""
        try:
            from safetensors.torch import load_file
            from safetensors import safe_open
            
            tensors = load_file(checkpoint_path)
            
            # Extract model state dict
            model_state_dict = {}
            optimizer_state_dict = {}
            
            for key, tensor in tensors.items():
                if key.startswith('model.'):
                    model_key = key[6:]  # Remove 'model.' prefix
                    model_state_dict[model_key] = tensor
                elif key.startswith('optimizer.'):
                    opt_key = key[10:]  # Remove 'optimizer.' prefix
                    optimizer_state_dict[opt_key] = tensor
            
            # Load model state
            model.load_state_dict(model_state_dict)
            
            # Load optimizer state (basic implementation)
            if optimizer and optimizer_state_dict:
                # This is a simplified version - full implementation would need 
                # proper reconstruction of optimizer state dict structure
                try:
                    optimizer.load_state_dict(optimizer_state_dict)
                except Exception as e:
                    logging.warning(f"Could not load optimizer state from safetensors: {e}")
            
            # Extract metadata
            metadata = {}
            with safe_open(checkpoint_path, framework="pt") as f:
                metadata = f.metadata()
            
            logging.info(f"Loaded SafeTensors checkpoint from {checkpoint_path}")
            
            return {
                'step': int(metadata.get('step', 0)) if 'step' in metadata else 0,
                'loss': float(metadata.get('loss', float('inf'))) if 'loss' in metadata else float('inf'),
                'config': json.loads(metadata.get('config', '{}')) if 'config' in metadata else {},
                'streaming_info': json.loads(metadata.get('streaming_info', '{}')) if 'streaming_info' in metadata else {}
            }
            
        except ImportError:
            logging.error("SafeTensors not available for loading checkpoint")
            raise
    
    @staticmethod
    def cleanup_checkpoints(checkpoint_dir: str, keep_latest: int = 5):
        """Clean up old checkpoints, keeping only the latest ones."""
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return
        
        # Find all checkpoint files (both .pt and .safetensors)
        checkpoints = {}  # step -> list of paths
        for ext in ['.pt', '.safetensors']:
            pattern = f"checkpoint-step-*{ext}"
            for path in checkpoint_dir.glob(pattern):
                try:
                    step_str = path.stem.split('-step-')[1]
                    step = int(step_str)
                    if step not in checkpoints:
                        checkpoints[step] = []
                    checkpoints[step].append(path)
                except (IndexError, ValueError):
                    continue
        
        # Sort by step number
        sorted_steps = sorted(checkpoints.keys())
        
        # Remove old checkpoints (keep only the latest N)
        if len(sorted_steps) > keep_latest:
            steps_to_remove = sorted_steps[:-keep_latest]
            for step in steps_to_remove:
                for path in checkpoints[step]:
                    try:
                        path.unlink()
                        logging.info(f"Removed old checkpoint: {path}")
                    except Exception as e:
                        logging.warning(f"Could not remove checkpoint {path}: {e}")
    
    @staticmethod
    def estimate_tokens_per_second(
        batch_size: int,
        sequence_length: int,
        time_elapsed: float,
        gradient_accumulation_steps: int = 1
    ) -> float:
        """Estimate tokens processed per second."""
        total_tokens = batch_size * sequence_length * gradient_accumulation_steps
        return total_tokens / time_elapsed if time_elapsed > 0 else 0.0
    
    @staticmethod
    def estimate_streaming_throughput(
        examples_processed: int,
        time_elapsed: float,
        avg_sequence_length: int = 2048
    ) -> Dict[str, float]:
        """Estimate streaming dataset throughput."""
        examples_per_sec = examples_processed / time_elapsed if time_elapsed > 0 else 0.0
        tokens_per_sec = examples_per_sec * avg_sequence_length
        
        return {
            'examples_per_second': examples_per_sec,
            'tokens_per_second': tokens_per_sec,
            'total_examples': examples_processed,
            'total_time': time_elapsed
        }
    
    @staticmethod
    def calculate_flops_per_token(
        model_params: int,
        sequence_length: int,
        vocab_size: int
    ) -> int:
        """Estimate FLOPs per token for transformer model."""
        # Simplified FLOP calculation for transformer
        # Forward pass: 2 * params (for matrix multiplications)
        # Attention: 4 * n_layers * seq_len^2 * d_model
        # This is a rough approximation
        forward_flops = 2 * model_params
        
        # Add attention computation (rough estimate)
        # Assuming typical transformer with d_model = sqrt(model_params / (6 * n_layers))
        n_layers_est = max(1, int(math.log2(model_params / 1000000)))  # Rough estimate
        d_model_est = int(math.sqrt(model_params / (6 * n_layers_est)))
        attention_flops = 4 * n_layers_est * sequence_length * sequence_length * d_model_est
        
        return forward_flops + attention_flops

def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    **kwargs
) -> torch.optim.Optimizer:
    """Get optimizer for model."""
    
    # Separate parameters with and without weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Don't apply weight decay to biases, layer norms, embeddings
        if any(nd in name.lower() for nd in ['bias', 'norm', 'embed']):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    if optimizer_name.lower() == "adamw":
        return AdamW(
            param_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            **kwargs
        )
    elif optimizer_name.lower() == "sgd":
        return SGD(
            param_groups,
            lr=learning_rate,
            momentum=kwargs.get('momentum', 0.9),
            **{k: v for k, v in kwargs.items() if k != 'momentum'}
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "cosine",
    warmup_steps: int = 1000,
    max_steps: int = 100000,
    min_lr_ratio: float = 0.1,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler."""
    
    if scheduler_name.lower() == "cosine":
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # Linear warmup
                return current_step / warmup_steps
            else:
                # Cosine annealing
                progress = (current_step - warmup_steps) / (max_steps - warmup_steps)
                progress = min(1.0, progress)
                cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
                return min_lr_ratio + (1 - min_lr_ratio) * cosine_factor
        
        return LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_name.lower() == "linear":
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / warmup_steps
            else:
                progress = (current_step - warmup_steps) / (max_steps - warmup_steps)
                progress = min(1.0, progress)
                return min_lr_ratio + (1 - min_lr_ratio) * (1 - progress)
        
        return LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_name.lower() == "constant":
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / warmup_steps
            else:
                return 1.0
        
        return LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_name.lower() == "polynomial":
        power = kwargs.get('power', 2.0)
        
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / warmup_steps
            else:
                progress = (current_step - warmup_steps) / (max_steps - warmup_steps)
                progress = min(1.0, progress)
                decay_factor = (1 - progress) ** power
                return min_lr_ratio + (1 - min_lr_ratio) * decay_factor
        
        return LambdaLR(optimizer, lr_lambda)
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
