"""
Checkpoint management utilities for INDRA LLM
(c) Divyansh Bharadwaj
"""

import os
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

@dataclass
class CheckpointMetadata:
    """Metadata for model checkpoints."""
    model_name: str
    step: int
    epoch: int
    loss: float
    timestamp: str
    config: Dict[str, Any]
    model_size_mb: float
    training_phase: str
    vedic_alignment_score: Optional[float] = None
    notes: Optional[str] = None

class CheckpointManager:
    """Advanced checkpoint management with versioning and metadata."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        compression: bool = True,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
            compression: Whether to compress checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.compression = compression
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.checkpoint_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Checkpoint tracking
        self.checkpoints = self._load_checkpoint_registry()
        
        logging.info(f"CheckpointManager initialized at {checkpoint_dir}")
    
    def save_checkpoint(
        self,
        model: nn.Module,
        step: int,
        epoch: int,
        loss: float,
        config: Dict[str, Any],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        training_phase: str = "training",
        vedic_alignment_score: Optional[float] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Save model checkpoint with comprehensive metadata.
        
        Args:
            model: Model to save
            step: Training step
            epoch: Training epoch
            loss: Current loss
            config: Model configuration
            optimizer: Optimizer state (optional)
            scheduler: Scheduler state (optional)
            training_phase: Phase of training (pretrain, sft, rlhf)
            vedic_alignment_score: Vedic alignment score (if available)
            notes: Additional notes
            tags: Tags for checkpoint categorization
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().isoformat()
        checkpoint_name = f"checkpoint-step-{step}-{timestamp.split('T')[0]}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'step': step,
            'epoch': epoch,
            'loss': loss,
            'config': config,
            'timestamp': timestamp,
            'training_phase': training_phase,
            'vedic_alignment_score': vedic_alignment_score,
        }
        
        # Add optimizer state if requested and available
        if self.save_optimizer and optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add scheduler state if requested and available
        if self.save_scheduler and scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save checkpoint
        if self.compression:
            # Save with compression
            torch.save(checkpoint_data, checkpoint_path, _use_new_zipfile_serialization=True)
        else:
            torch.save(checkpoint_data, checkpoint_path)
        
        # Calculate file size
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        
        # Create metadata
        metadata = CheckpointMetadata(
            model_name="INDRA",
            step=step,
            epoch=epoch,
            loss=loss,
            timestamp=timestamp,
            config=config,
            model_size_mb=file_size_mb,
            training_phase=training_phase,
            vedic_alignment_score=vedic_alignment_score,
            notes=notes
        )
        
        # Save metadata
        metadata_path = self.metadata_dir / f"{checkpoint_name}.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # Update registry
        self.checkpoints.append({
            'name': checkpoint_name,
            'path': str(checkpoint_path),
            'metadata_path': str(metadata_path),
            'step': step,
            'timestamp': timestamp,
            'tags': tags or [],
            'file_size_mb': file_size_mb
        })
        
        # Sort by step
        self.checkpoints.sort(key=lambda x: x['step'])
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        # Save updated registry
        self._save_checkpoint_registry()
        
        logging.info(f"Saved checkpoint: {checkpoint_path} ({file_size_mb:.1f} MB)")
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load checkpoint on
            strict: Whether to strictly enforce state dict matching
            
        Returns:
            Dictionary with checkpoint information
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        else:
            # Fallback for older checkpoint formats
            model.load_state_dict(checkpoint, strict=strict)
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                logging.warning(f"Could not load optimizer state: {e}")
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                logging.warning(f"Could not load scheduler state: {e}")
        
        # Extract information
        info = {
            'step': checkpoint.get('step', 0),
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', float('inf')),
            'config': checkpoint.get('config', {}),
            'timestamp': checkpoint.get('timestamp', ''),
            'training_phase': checkpoint.get('training_phase', 'unknown'),
            'vedic_alignment_score': checkpoint.get('vedic_alignment_score')
        }
        
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
        logging.info(f"Step: {info['step']}, Loss: {info['loss']:.4f}")
        
        return info
    
    def list_checkpoints(self, training_phase: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available checkpoints.
        
        Args:
            training_phase: Filter by training phase
            
        Returns:
            List of checkpoint information
        """
        checkpoints = self.checkpoints.copy()
        
        if training_phase:
            # Load metadata to filter by training phase
            filtered_checkpoints = []
            for cp in checkpoints:
                try:
                    with open(cp['metadata_path'], 'r') as f:
                        metadata = json.load(f)
                    if metadata.get('training_phase') == training_phase:
                        filtered_checkpoints.append(cp)
                except Exception:
                    continue
            checkpoints = filtered_checkpoints
        
        return checkpoints
    
    def get_best_checkpoint(self, metric: str = "loss", minimize: bool = True) -> Optional[str]:
        """
        Get best checkpoint based on a metric.
        
        Args:
            metric: Metric to optimize for
            minimize: Whether to minimize the metric
            
        Returns:
            Path to best checkpoint
        """
        if not self.checkpoints:
            return None
        
        best_checkpoint = None
        best_value = float('inf') if minimize else float('-inf')
        
        for cp in self.checkpoints:
            try:
                with open(cp['metadata_path'], 'r') as f:
                    metadata = json.load(f)
                
                value = metadata.get(metric)
                if value is None:
                    continue
                
                if minimize and value < best_value:
                    best_value = value
                    best_checkpoint = cp['path']
                elif not minimize and value > best_value:
                    best_value = value
                    best_checkpoint = cp['path']
                    
            except Exception:
                continue
        
        return best_checkpoint
    
    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """
        Delete a specific checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint to delete
            
        Returns:
            True if deleted successfully
        """
        for i, cp in enumerate(self.checkpoints):
            if cp['name'] == checkpoint_name:
                # Delete files
                try:
                    Path(cp['path']).unlink(missing_ok=True)
                    Path(cp['metadata_path']).unlink(missing_ok=True)
                    
                    # Remove from registry
                    del self.checkpoints[i]
                    self._save_checkpoint_registry()
                    
                    logging.info(f"Deleted checkpoint: {checkpoint_name}")
                    return True
                except Exception as e:
                    logging.error(f"Error deleting checkpoint {checkpoint_name}: {e}")
                    return False
        
        return False
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints to maintain max_checkpoints limit."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Keep the most recent checkpoints
        to_remove = self.checkpoints[:-self.max_checkpoints]
        
        for cp in to_remove:
            try:
                Path(cp['path']).unlink(missing_ok=True)
                Path(cp['metadata_path']).unlink(missing_ok=True)
                logging.info(f"Cleaned up old checkpoint: {cp['name']}")
            except Exception as e:
                logging.warning(f"Could not remove checkpoint {cp['name']}: {e}")
        
        # Update registry
        self.checkpoints = self.checkpoints[-self.max_checkpoints:]
    
    def _load_checkpoint_registry(self) -> List[Dict[str, Any]]:
        """Load checkpoint registry."""
        registry_path = self.checkpoint_dir / "checkpoint_registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Could not load checkpoint registry: {e}")
        
        return []
    
    def _save_checkpoint_registry(self):
        """Save checkpoint registry."""
        registry_path = self.checkpoint_dir / "checkpoint_registry.json"
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.checkpoints, f, indent=2)
        except Exception as e:
            logging.error(f"Could not save checkpoint registry: {e}")

class ModelSaver:
    """Utility for saving models in various formats."""
    
    @staticmethod
    def save_for_huggingface(
        model: nn.Module,
        tokenizer,
        save_directory: str,
        model_name: str = "indra-llm",
        push_to_hub: bool = False,
        **kwargs
    ):
        """
        Save model in HuggingFace format.
        
        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            save_directory: Directory to save model
            model_name: Name for the model
            push_to_hub: Whether to push to HuggingFace Hub
            **kwargs: Additional arguments for push_to_hub
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save_pretrained(save_path)
        
        # Save tokenizer
        if hasattr(tokenizer, 'save_pretrained'):
            tokenizer.save_pretrained(save_path)
        
        # Create model card
        model_card_content = f"""---
language: en
tags:
- pytorch
- transformers
- vedic
- philosophy
- indra
license: custom
---

# {model_name}

INDRA is a Vedic-aligned Large Language Model that combines cutting-edge transformer architecture with ancient philosophical wisdom.

## Features

- Decoder-only transformer with FlashAttention and Grouped Query Attention
- Mixture of Experts (MoE) architecture
- Vedic philosophical integration with dharmic alignment
- Sanskrit language support
- Multi-language capability (Sanskrit, Hindi, English)

## Usage

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
```

## Training

This model was trained with a curriculum learning approach:
1. Vedic corpus pre-training
2. General domain adaptation
3. Instruction fine-tuning with dharmic alignment
4. RLHF with Vedic-aligned reward model

## Citation

```bibtex
@software{{indra_llm_2024,
  title={{INDRA LLM: Vedic-aligned Decoder-only Transformer}},
  author={{Bharadwaj, Divyansh}},
  year={{2024}},
  url={{https://github.com/your-username/indra-llm}}
}}
```
"""
        
        with open(save_path / "README.md", 'w') as f:
            f.write(model_card_content)
        
        logging.info(f"Model saved to {save_path}")
        
        if push_to_hub:
            try:
                from huggingface_hub import create_repo, upload_folder
                
                # Create repository
                create_repo(model_name, exist_ok=True, **kwargs)
                
                # Upload folder
                upload_folder(
                    folder_path=str(save_path),
                    repo_id=model_name,
                    **kwargs
                )
                
                logging.info(f"Model pushed to HuggingFace Hub: {model_name}")
                
            except ImportError:
                logging.error("huggingface_hub not installed. Cannot push to hub.")
            except Exception as e:
                logging.error(f"Error pushing to hub: {e}")
    
    @staticmethod
    def save_onnx(
        model: nn.Module,
        save_path: str,
        input_shape: tuple = (1, 512),
        opset_version: int = 14,
        **kwargs
    ):
        """
        Save model in ONNX format for deployment.
        
        Args:
            model: PyTorch model
            save_path: Path to save ONNX model
            input_shape: Input shape for export
            opset_version: ONNX opset version
            **kwargs: Additional arguments for torch.onnx.export
        """
        try:
            import torch.onnx
            
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randint(0, model.config.vocab_size, input_shape)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                save_path,
                export_params=True,
                opset_version=opset_version,
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'logits': {0: 'batch_size', 1: 'sequence'}
                },
                **kwargs
            )
            
            logging.info(f"Model exported to ONNX: {save_path}")
            
        except ImportError:
            logging.error("ONNX export requires torch and onnx packages")
        except Exception as e:
            logging.error(f"Error exporting to ONNX: {e}")
    
    @staticmethod
    def save_safetensors(
        model: nn.Module,
        save_path: str,
        metadata: Optional[Dict[str, str]] = None
    ):
        """
        Save model weights in SafeTensors format.
        
        Args:
            model: PyTorch model
            save_path: Path to save SafeTensors file
            metadata: Optional metadata dictionary
        """
        try:
            from safetensors.torch import save_file
            
            state_dict = model.state_dict()
            
            # Convert metadata to strings
            if metadata:
                metadata = {k: str(v) for k, v in metadata.items()}
            
            save_file(state_dict, save_path, metadata=metadata)
            
            logging.info(f"Model saved in SafeTensors format: {save_path}")
            
        except ImportError:
            logging.error("SafeTensors export requires safetensors package")
        except Exception as e:
            logging.error(f"Error saving SafeTensors: {e}")
    
    @staticmethod
    def create_model_archive(
        model_dir: str,
        archive_path: str,
        include_checkpoints: bool = False,
        compression_level: int = 6
    ):
        """
        Create compressed archive of model directory.
        
        Args:
            model_dir: Directory containing model files
            archive_path: Path for output archive
            include_checkpoints: Whether to include checkpoint files
            compression_level: Compression level (0-9)
        """
        import tarfile
        
        model_path = Path(model_dir)
        
        with tarfile.open(archive_path, 'w:gz', compresslevel=compression_level) as tar:
            for file_path in model_path.rglob('*'):
                if file_path.is_file():
                    # Skip checkpoints if not requested
                    if not include_checkpoints and 'checkpoint' in file_path.name:
                        continue
                    
                    # Add file to archive
                    arcname = file_path.relative_to(model_path)
                    tar.add(file_path, arcname=arcname)
        
        logging.info(f"Model archive created: {archive_path}")

def backup_training_state(
    checkpoint_manager: CheckpointManager,
    backup_dir: str,
    max_backups: int = 3
):
    """
    Create backup of training state.
    
    Args:
        checkpoint_manager: CheckpointManager instance
        backup_dir: Directory for backups
        max_backups: Maximum number of backups to keep
    """
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"training_backup_{timestamp}.tar.gz"
    backup_file = backup_path / backup_name
    
    # Create archive
    ModelSaver.create_model_archive(
        str(checkpoint_manager.checkpoint_dir),
        str(backup_file),
        include_checkpoints=True
    )
    
    # Cleanup old backups
    backups = sorted(backup_path.glob("training_backup_*.tar.gz"))
    if len(backups) > max_backups:
        for old_backup in backups[:-max_backups]:
            old_backup.unlink()
            logging.info(f"Removed old backup: {old_backup}")
    
    logging.info(f"Training state backed up to: {backup_file}")
