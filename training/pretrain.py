"""
Pre-training implementation for INDRA LLM with Vedic curriculum learning
Modified to use streaming datasets for memory efficiency
(c) Divyansh Bharadwaj
"""

import os
import glob
import time
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path
import psutil
import GPUtil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from .trainer_utils import TrainerUtils, get_optimizer, get_scheduler, MetricsTracker, StreamingMetricsTracker
from data import StreamingINDRADataset, StreamingVedicDataset
from model import INDRATransformer


class PretrainTrainer:
    """Pre-training trainer with Vedic curriculum learning and streaming datasets."""
    
    def __init__(
        self,
        model: INDRATransformer,
        config,
        train_dataset,  # Now expects StreamingINDRADataset or StreamingVedicDataset
        val_dataset=None,
        tokenizer=None,
    ):
        """
        Initialize pre-training trainer.
        
        Args:
            model: INDRA transformer model
            config: Training configuration
            train_dataset: Streaming training dataset
            val_dataset: Streaming validation dataset
            tokenizer: Tokenizer instance
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        
        # Setup device and distributed training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_distributed = config.use_ddp or config.use_fsdp
        
        if self.is_distributed:
            self.model = self._setup_distributed_model()
        else:
            self.model = self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = get_optimizer(
            self.model,
            optimizer_name="adamw",
            learning_rate=config.pretrain.max_lr,
            weight_decay=config.pretrain.weight_decay,
            beta1=config.pretrain.beta1,
            beta2=config.pretrain.beta2,
        )
        
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_name="cosine",
            warmup_steps=config.pretrain.warmup_steps,
            max_steps=config.max_steps,
            min_lr_ratio=config.pretrain.min_lr / config.pretrain.max_lr,
        )
        
        # Setup mixed precision
        self.scaler = GradScaler(enabled=config.use_fp16 or config.use_bf16)
        self.use_amp = config.use_fp16 or config.use_bf16
        self.amp_dtype = torch.float16 if config.use_fp16 else torch.bfloat16

        # Vedic curriculum phases
        self.current_phase = "vedic"  # vedic -> general -> mixed
        self.phase_steps = {
            "vedic": config.pretrain.vedic_phase_steps,
            "general": config.pretrain.general_phase_steps,
            "mixed": config.max_steps - config.pretrain.vedic_phase_steps - config.pretrain.general_phase_steps
        }
        
        # Setup data loaders - streaming datasets handle batching internally
        self.train_loader = self._create_train_loader()
        self.val_loader = self._create_val_loader() if val_dataset else None
        
        # Tracking and logging
        self.metrics_tracker = MetricsTracker(window_size=config.logging_steps)
        self.global_step = 0
        self.epoch = 0
        
        # Setup logging
        TrainerUtils.setup_logging()
        self.wandb = TrainerUtils.setup_wandb(
            vars(config), config.wandb_project, config.wandb_run_name
        )
        
        # Log model info
        param_counts = TrainerUtils.count_parameters(self.model)
        logging.info(f"Model parameters: {param_counts}")
        logging.info(f"Using streaming dataset with {train_dataset.total_files} files")
        
    def _setup_distributed_model(self) -> nn.Module:
        """Setup model for distributed training."""
        if self.config.use_fsdp:
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
                from .model.transformer import TransformerBlock
                
                auto_wrap_policy = transformer_auto_wrap_policy({TransformerBlock})
                model = FSDP(
                    self.model,
                    auto_wrap_policy=auto_wrap_policy,
                    mixed_precision=None,  # Handle separately
                )
                return model
            except ImportError:
                logging.warning("FSDP not available, falling back to DDP")
        
        if self.config.use_ddp:
            model = DDP(self.model, device_ids=[self.config.local_rank])
            return model
        
        return self.model.to(self.device)
    
    def _create_train_loader(self) -> DataLoader:
        """Create training data loader for streaming datasets."""
        # Define safe collate function locally
        def safe_collate_fn(batch):
            """Custom collate function that handles inconsistent dict keys safely."""
            if not batch:
                return {}
            
            # Get all possible keys from all examples
            all_keys = set()
            for example in batch:
                if isinstance(example, dict):
                    all_keys.update(example.keys())
            
            # Create the collated batch
            collated = {}
            
            for key in all_keys:
                values = []
                for example in batch:
                    if isinstance(example, dict) and key in example:
                        values.append(example[key])
                
                if values:
                    # Only collate if we have values
                    try:
                        if isinstance(values[0], torch.Tensor):
                            collated[key] = torch.stack(values)
                        elif isinstance(values[0], (int, float)):
                            collated[key] = torch.tensor(values)
                        elif isinstance(values[0], str):
                            collated[key] = values  # Keep as list for strings
                        else:
                            collated[key] = values  # Keep as list for other types
                    except Exception as e:
                        # If collation fails, keep as list
                        collated[key] = values
            
            return collated
        
        # For streaming datasets, we use a simpler DataLoader setup
        # The dataset itself handles the streaming and batching logic
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.dataloader_num_workers if hasattr(self.config, 'dataloader_num_workers') else 0,
            pin_memory=self.config.pin_memory if hasattr(self.config, 'pin_memory') else True,
            collate_fn=safe_collate_fn,  # Use safe collate function
            # Note: shuffle=False because streaming dataset handles shuffling internally
        )
    
    def _create_val_loader(self) -> Optional[DataLoader]:
        """Create validation data loader for streaming datasets."""
        if not self.val_dataset:
            return None
        
        # Define safe collate function locally
        def safe_collate_fn(batch):
            """Custom collate function that handles inconsistent dict keys safely."""
            if not batch:
                return {}
            
            # Get all possible keys from all examples
            all_keys = set()
            for example in batch:
                if isinstance(example, dict):
                    all_keys.update(example.keys())
            
            # Create the collated batch
            collated = {}
            
            for key in all_keys:
                values = []
                for example in batch:
                    if isinstance(example, dict) and key in example:
                        values.append(example[key])
                
                if values:
                    # Only collate if we have values
                    try:
                        if isinstance(values[0], torch.Tensor):
                            collated[key] = torch.stack(values)
                        elif isinstance(values[0], (int, float)):
                            collated[key] = torch.tensor(values)
                        elif isinstance(values[0], str):
                            collated[key] = values  # Keep as list for strings
                        else:
                            collated[key] = values  # Keep as list for other types
                    except Exception as e:
                        # If collation fails, keep as list
                        collated[key] = values
            
            return collated
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.eval_batch_size,
            num_workers=self.config.dataloader_num_workers if hasattr(self.config, 'dataloader_num_workers') else 0,
            pin_memory=self.config.pin_memory if hasattr(self.config, 'pin_memory') else True,
            collate_fn=safe_collate_fn,  # Use safe collate function
        )
    
    def _update_curriculum_phase(self):
        """Update curriculum learning phase based on current step."""
        if self.global_step < self.phase_steps["vedic"]:
            new_phase = "vedic"
        elif self.global_step < self.phase_steps["vedic"] + self.phase_steps["general"]:
            new_phase = "general"
        else:
            new_phase = "mixed"
        
        if new_phase != self.current_phase:
            logging.info(f"Curriculum phase transition: {self.current_phase} -> {new_phase}")
            self.current_phase = new_phase
            
            # For streaming datasets, we don't need to recreate the loader
            # The dataset handles phase transitions internally
    
    def train(self) -> Dict[str, Any]:
        """Main training loop with streaming datasets."""
        logging.info(f"Starting pre-training for {self.config.max_steps} steps")
        logging.info(f"Using streaming dataset with memory budget: {self.train_dataset.batch_size_mb}MB")
        
        self.model.train()
        start_time = time.time()
        
        # Training loop - streaming datasets provide infinite iteration
        try:
            epoch_loss = self._train_epoch()
            
            # Final validation and save
            if self.val_loader:
                final_metrics = self._validate()
                logging.info(f"Final validation metrics: {final_metrics}")
            
            self._save_checkpoint(final=True)
            
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            self._save_checkpoint(final=True)
        
        except Exception as e:
            logging.error(f"Training failed with error: {e}")
            # Save emergency checkpoint
            try:
                self._save_checkpoint(final=True)
            except:
                pass
            raise
        
        total_time = time.time() - start_time
        logging.info(f"Pre-training completed in {total_time:.2f} seconds")
        
        return {
            'final_step': self.global_step,
            'total_time': total_time,
            'final_metrics': self.metrics_tracker.get_summary()
        }
    
    def _train_epoch(self) -> float:
        """Train with streaming dataset - runs until max_steps reached."""
        epoch_loss = 0.0
        num_batches = 0
        
        # Streaming datasets provide infinite iteration until we break
        for batch_idx, batch in enumerate(self.train_loader):
            if self.global_step >= self.config.max_steps:
                logging.info(f"Reached max steps ({self.config.max_steps}), stopping training")
                break
            
            try:
                loss = self._train_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_metrics(loss)
                    self.log_system_stats()
                
                
                # Update curriculum phase
                self._update_curriculum_phase()

                # Validation
                if self.val_loader and self.global_step > 0 and self.global_step % self.config.eval_steps == 0:
                    val_metrics = self._validate()
                    self.metrics_tracker.update(val_metrics, self.global_step)
                    
                    logging.info(f"Step {self.global_step}: val_loss={val_metrics['val_loss']:.4f}")
                    
                    if self.wandb:
                        self.wandb.log(val_metrics, step=self.global_step)
                
                # Save checkpoint
                if self.global_step > 0 and self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()
                
                self.global_step += 1
                
            except Exception as e:
                logging.error(f"Error in training step {batch_idx}: {e}")
                # For streaming, we can continue with next batch
                continue
        
        return epoch_loss / max(num_batches, 1)
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step with streaming dataset batch."""
        try:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            use_autocast = (self.use_amp and torch.cuda.is_available())
            
            if use_autocast:
                with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        labels=batch.get('labels'),
                        compute_vedic_rewards=True,
                    )
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            else:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch.get('labels'),
                    compute_vedic_rewards=True,
                )
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.pretrain.grad_clip
                )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            return loss.item() * self.config.gradient_accumulation_steps

        except Exception as e:
            logging.error(f"Error in training step: {e}")
            logging.error(f"Batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'Not a dict'}")
            
            # Log batch information for debugging
            if isinstance(batch, dict):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        logging.error(f"  {k}: shape {v.shape}, dtype {v.dtype}, device {v.device}")
                    else:
                        logging.error(f"  {k}: {type(v)}")
            
            # For streaming, we can skip this batch and continue
            return 0.0
    
    def _validate(self) -> Dict[str, float]:
        """Run validation on streaming validation dataset with proper memory management."""
        if not self.val_loader:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_aux_loss = 0.0
        total_vedic_loss = 0.0
        num_examples = 0
        max_val_batches = 20  # Limit validation batches for streaming
        max_examples_per_batch = 4  # Process only small chunks at a time
        
        with torch.no_grad():
            for batch_idx, streaming_batch in enumerate(self.val_loader):
                if batch_idx >= max_val_batches:
                    break
                
                try:
                    # Handle streaming batch - it should be a dictionary from the collate function
                    if not isinstance(streaming_batch, dict):
                        logging.warning(f"Unexpected batch type: {type(streaming_batch)}")
                        continue
                    
                    if not streaming_batch:
                        logging.warning("Empty batch received")
                        continue
                    
                    # Get batch size from one of the tensor fields
                    batch_size = 0
                    for key, values in streaming_batch.items():
                        if isinstance(values, torch.Tensor) and len(values.shape) > 0:
                            batch_size = values.shape[0]
                            break
                        elif isinstance(values, list):
                            batch_size = len(values)
                            break
                    
                    if batch_size == 0:
                        logging.warning("Could not determine batch size")
                        continue
                    
                    # Process examples in small chunks to avoid OOM
                    for chunk_start in range(0, batch_size, max_examples_per_batch):
                        chunk_end = min(chunk_start + max_examples_per_batch, batch_size)
                        
                        # Extract chunk from the batch
                        chunk_batch = {}
                        for key, values in streaming_batch.items():
                            if isinstance(values, torch.Tensor):
                                chunk_batch[key] = values[chunk_start:chunk_end]
                            elif isinstance(values, list):
                                chunk_batch[key] = values[chunk_start:chunk_end]
                            else:
                                chunk_batch[key] = values
                        
                        if not chunk_batch or not any(isinstance(v, torch.Tensor) for v in chunk_batch.values()):
                            continue
                        
                        # Move chunk to device
                        try:
                            chunk_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                         for k, v in chunk_batch.items()}
                        except Exception as e:
                            logging.warning(f"Error moving chunk to device: {e}")
                            continue
                        
                        try:
                            # Forward pass with memory management
                            torch.cuda.empty_cache()  # Clear cache before validation
                            
                            with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                                outputs = self.model(
                                    input_ids=chunk_batch['input_ids'],
                                    attention_mask=chunk_batch.get('attention_mask'),
                                    labels=chunk_batch.get('labels'),
                                    compute_vedic_rewards=True
                                )
                            
                            chunk_size = chunk_batch['input_ids'].size(0)
                            total_loss += outputs['loss'].item() * chunk_size
                            
                            # Safely get auxiliary losses
                            aux_losses = outputs.get('aux_losses', {})
                            if isinstance(aux_losses, dict):
                                total_aux_loss += aux_losses.get('total_aux_loss', 0.0) * chunk_size
                                total_vedic_loss += aux_losses.get('vedic_alignment_loss', 0.0) * chunk_size
                            
                            num_examples += chunk_size
                            
                            # Clear memory after each chunk
                            del outputs, chunk_batch
                            torch.cuda.empty_cache()
                            
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                logging.warning(f"Validation chunk OOM, trying single examples")
                                torch.cuda.empty_cache()
                                
                                # Try processing one example at a time
                                for single_idx in range(chunk_start, chunk_end):
                                    if single_idx >= batch_size:
                                        break
                                    
                                    try:
                                        single_batch = {}
                                        for key, values in streaming_batch.items():
                                            if isinstance(values, torch.Tensor):
                                                single_batch[key] = values[single_idx:single_idx+1]
                                            elif isinstance(values, list):
                                                single_batch[key] = [values[single_idx]]
                                            else:
                                                single_batch[key] = values
                                        
                                        single_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                                      for k, v in single_batch.items()}
                                        
                                        with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                                            outputs = self.model(
                                                input_ids=single_batch['input_ids'],
                                                attention_mask=single_batch.get('attention_mask'),
                                                labels=single_batch.get('labels'),
                                                compute_vedic_rewards=True
                                            )
                                        
                                        total_loss += outputs['loss'].item()
                                        aux_losses = outputs.get('aux_losses', {})
                                        if isinstance(aux_losses, dict):
                                            total_aux_loss += aux_losses.get('total_aux_loss', 0.0)
                                            total_vedic_loss += aux_losses.get('vedic_alignment_loss', 0.0)
                                        num_examples += 1
                                        
                                        del outputs, single_batch
                                        torch.cuda.empty_cache()
                                        
                                    except Exception as single_e:
                                        logging.warning(f"Single example validation failed: {single_e}")
                                        continue
                            else:
                                logging.warning(f"Validation error: {e}")
                                torch.cuda.empty_cache()
                        
                        # Stop if we've processed enough examples
                        if num_examples >= 100:  # Reasonable validation set size
                            break
                    
                    if num_examples >= 100:
                        break
                        
                except Exception as e:
                    logging.warning(f"Error in validation batch {batch_idx}: {e}")
                    torch.cuda.empty_cache()
                    continue
        
        # Clear cache and switch back to train mode
        torch.cuda.empty_cache()
        self.model.train()
        
        if num_examples == 0:
            logging.warning("No validation examples processed successfully")
            return {
                'val_loss': 0.0,
                'val_aux_loss': 0.0,
                'val_vedic_loss': 0.0,
            }
        
        return {
            'val_loss': total_loss / num_examples,
            'val_aux_loss': total_aux_loss / num_examples,
            'val_vedic_loss': total_vedic_loss / num_examples,
        }
    
    def _collate_chunk(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Safely collate a small chunk of examples."""
        if not examples:
            return {}
        
        # Get all possible keys from all examples
        all_keys = set()
        for example in examples:
            if isinstance(example, dict):
                all_keys.update(example.keys())
        
        # Create the collated batch
        collated = {}
        
        for key in all_keys:
            values = []
            for example in examples:
                if isinstance(example, dict) and key in example:
                    values.append(example[key])
            
            if values:
                try:
                    if isinstance(values[0], torch.Tensor):
                        # Handle different tensor shapes gracefully
                        if all(v.shape == values[0].shape for v in values):
                            collated[key] = torch.stack(values)
                        else:
                            # Skip tensors with mismatched shapes
                            continue
                    elif isinstance(values[0], (int, float)):
                        collated[key] = torch.tensor(values)
                    else:
                        collated[key] = values  # Keep as list for other types
                except Exception as e:
                    # If collation fails, skip this key
                    logging.debug(f"Failed to collate key {key}: {e}")
                    continue
        
        return collated
    
    def _log_metrics(self, loss: float):
        """Log training metrics with streaming dataset info."""
        # Calculate tokens per second
        current_time = time.time()
        if not hasattr(self, '_last_log_time'):
            self._last_log_time = current_time
            self._last_log_step = self.global_step
            return
        
        time_diff = current_time - self._last_log_time
        step_diff = self.global_step - self._last_log_step
        
        if time_diff > 0 and step_diff > 0:
            tokens_per_sec = TrainerUtils.estimate_tokens_per_second(
                self.config.batch_size,
                self.config.max_seq_length,
                time_diff / step_diff,
                self.config.gradient_accumulation_steps
            )
        else:
            tokens_per_sec = 0.0
        
        # Get memory usage
        memory_stats = TrainerUtils.get_memory_usage()
        
        # Current learning rate
        current_lr = self.scheduler.get_last_lr()[0]
        
        # Add streaming-specific metrics
        streaming_stats = {}
        if hasattr(self.train_dataset, '_shuffle_buffer'):
            streaming_stats['buffer_size'] = len(self.train_dataset._shuffle_buffer)
        
        metrics = {
            'train_loss': loss,
            'learning_rate': current_lr,
            'tokens_per_second': tokens_per_sec,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'curriculum_phase': self.current_phase,
            'memory_batch_mb': self.train_dataset.batch_size_mb,
        }
        metrics.update(memory_stats)
        metrics.update(streaming_stats)
        
        # Update tracker
        self.metrics_tracker.update(metrics, self.global_step)
        
        # Log to console
        logging.info(
            f"Step {self.global_step}: loss={loss:.4f}, lr={current_lr:.2e}, "
            f"tokens/s={tokens_per_sec:.0f}, phase={self.current_phase}, "
            f"mem_batch={self.train_dataset.batch_size_mb}MB"
        )
        
        # Log to wandb
        if self.wandb:
            self.wandb.log(metrics, step=self.global_step)
        
        self._last_log_time = current_time
        self._last_log_step = self.global_step

    # def log_system_stats():
    #     # CPU and Memory
    #     cpu_percent = psutil.cpu_percent()
    #     memory = psutil.virtual_memory()
        
    #     # GPU
    #     gpus = GPUtil.getGPUs()
    #     for gpu in gpus:
    #         print(f"GPU {gpu.id}: {gpu.memoryUsed}/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
        
    #     print(f"CPU: {cpu_percent}%, RAM: {memory.percent}%, Available: {memory.available/1024**3:.1f}GB")
    
    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        suffix = "final" if final else f"step-{self.global_step}"
        
        TrainerUtils.save_checkpoint(
            model=self.model.module if isinstance(self.model, (DDP,)) else self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            loss=self.metrics_tracker.get_latest('train_loss'),
            checkpoint_dir=self.config.output_dir,
            config=vars(self.config),
            save_format="both"
        )
        
        # Cleanup old checkpoints
        if not final:
            TrainerUtils.cleanup_checkpoints(
                self.config.output_dir, 
                keep_latest=self.config.save_total_limit
            )
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        logging.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint_info = TrainerUtils.load_checkpoint(
            checkpoint_path,
            model=self.model.module if isinstance(self.model, DDP) else self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )
        
        self.global_step = checkpoint_info.get('step', 0)
        self.epoch = checkpoint_info.get('epoch', 0)
        
        logging.info(f"Resumed from step {self.global_step}")
        
        return checkpoint_info



