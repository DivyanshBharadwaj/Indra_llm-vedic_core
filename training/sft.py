"""
Supervised Fine-tuning (SFT) trainer for INDRA LLM with instruction following
Modified to use streaming datasets for memory efficiency
(c) Divyansh Bharadwaj
"""

import os
import time
import logging
from typing import Dict, Optional, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from .trainer_utils import TrainerUtils, get_optimizer, get_scheduler, MetricsTracker
from data import StreamingInstructionDataset
from model import INDRATransformer

class SFTTrainer:
    """Supervised Fine-tuning trainer for instruction following and task-specific training with streaming datasets."""
    
    def __init__(
        self,
        model: INDRATransformer,
        config,
        train_dataset,  # Now expects StreamingInstructionDataset
        val_dataset=None,
        tokenizer=None,
    ):
        """
        Initialize SFT trainer.
        
        Args:
            model: Pre-trained INDRA transformer model
            config: Training configuration
            train_dataset: Streaming instruction training dataset
            val_dataset: Streaming validation dataset
            tokenizer: Tokenizer instance
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Setup optimizer and scheduler for SFT
        self.optimizer = get_optimizer(
            self.model,
            optimizer_name="adamw",
            learning_rate=config.sft.max_lr,
            weight_decay=config.sft.weight_decay,
            beta1=config.sft.beta1,
            beta2=config.sft.beta2,
        )
        
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_name="cosine",
            warmup_steps=config.sft.warmup_steps,
            max_steps=config.max_steps,
            min_lr_ratio=config.sft.min_lr / config.sft.max_lr,
        )
        
        # Setup mixed precision
        self.scaler = GradScaler(enabled=config.use_fp16 or config.use_bf16)
        self.use_amp = config.use_fp16 or config.use_bf16
        self.amp_dtype = torch.float16 if config.use_fp16 else torch.bfloat16
        
        # Setup data loaders for streaming datasets
        self.train_loader = self._create_train_loader()
        self.val_loader = self._create_val_loader() if val_dataset else None
        
        # Tracking and logging
        self.metrics_tracker = MetricsTracker(window_size=config.logging_steps)
        self.global_step = 0
        self.epoch = 0
        
        # SFT specific settings
        self.response_loss_only = config.sft.response_loss_only
        self.max_seq_length = config.sft.max_seq_length
        
        # Setup logging
        TrainerUtils.setup_logging()
        self.wandb = TrainerUtils.setup_wandb(
            vars(config), config.wandb_project, 
            config.wandb_run_name or "indra-sft"
        )
        
        # Log model info
        param_counts = TrainerUtils.count_parameters(self.model)
        logging.info(f"SFT Model parameters: {param_counts}")
        logging.info(f"Response loss only: {self.response_loss_only}")
        logging.info(f"Using streaming dataset with {train_dataset.total_files} files")
        logging.info(f"Memory batch size: {train_dataset.batch_size_mb}MB")
    
    def _create_train_loader(self) -> DataLoader:
        """Create training data loader for streaming instruction data."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.dataloader_num_workers if hasattr(self.config, 'dataloader_num_workers') else 0,
            pin_memory=self.config.pin_memory if hasattr(self.config, 'pin_memory') else True,
        )
    
    def _create_val_loader(self) -> Optional[DataLoader]:
        """Create validation data loader for streaming datasets."""
        if not self.val_dataset:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.eval_batch_size,
            num_workers=self.config.dataloader_num_workers if hasattr(self.config, 'dataloader_num_workers') else 0,
            pin_memory=self.config.pin_memory if hasattr(self.config, 'pin_memory') else True,
        )
    
    def train(self) -> Dict[str, Any]:
        """Main SFT training loop with streaming datasets."""
        logging.info(f"Starting supervised fine-tuning for {self.config.max_steps} steps")
        logging.info(f"Using streaming dataset with memory budget: {self.train_dataset.batch_size_mb}MB")
        
        self.model.train()
        start_time = time.time()
        
        # Training loop with streaming datasets
        try:
            epoch_loss = self._train_epoch()
            
            # Final validation and save
            if self.val_loader:
                final_metrics = self._validate()
                logging.info(f"Final SFT validation metrics: {final_metrics}")
            
            self._save_checkpoint(final=True)
            
        except KeyboardInterrupt:
            logging.info("SFT training interrupted by user")
            self._save_checkpoint(final=True)
        
        except Exception as e:
            logging.error(f"SFT training failed with error: {e}")
            # Save emergency checkpoint
            try:
                self._save_checkpoint(final=True)
            except:
                pass
            raise
        
        total_time = time.time() - start_time
        logging.info(f"SFT training completed in {total_time:.2f} seconds")
        
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
                logging.info(f"Reached max steps ({self.config.max_steps}), stopping SFT training")
                break
            
            try:
                loss = self._train_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_metrics(loss)
                
                # Validation
                if self.val_loader and self.global_step > 0 and self.global_step % self.config.eval_steps == 0:
                    val_metrics = self._validate()
                    self.metrics_tracker.update(val_metrics, self.global_step)
                    
                    logging.info(f"Step {self.global_step}: val_loss={val_metrics.get('val_loss', 0.0):.4f}")
                    
                    if self.wandb:
                        self.wandb.log(val_metrics, step=self.global_step)
                
                # Save checkpoint
                if self.global_step > 0 and self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()
                
                self.global_step += 1
                
            except Exception as e:
                logging.error(f"Error in SFT training batch {batch_idx}: {e}")
                # For streaming, we can continue with next batch
                continue
        
        return epoch_loss / max(num_batches, 1)
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single SFT training step with instruction-response format and streaming dataset batch."""
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
                        compute_vedic_rewards=True  # Keep Vedic alignment during SFT
                    )
                    
                    loss = outputs['loss']
            else:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch.get('labels'),
                    compute_vedic_rewards=True  # Keep Vedic alignment during SFT
                )
                
                loss = outputs['loss']
            
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
                    self.config.sft.grad_clip
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
            logging.error(f"Error in SFT training step: {e}")
            # For streaming, we can skip this batch and continue
            return 0.0
    
    def _validate(self) -> Dict[str, float]:
        """Run validation on instruction-following tasks with streaming datasets."""
        if not self.val_loader:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_instruction_loss = 0.0
        total_response_loss = 0.0
        correct_responses = 0
        total_responses = 0
        num_batches = 0
        max_val_batches = 50  # Limit validation batches for streaming
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= max_val_batches:
                    break
                
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass
                    with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch.get('attention_mask'),
                            labels=batch.get('labels'),
                            compute_vedic_rewards=True
                        )
                    
                    total_loss += outputs['loss'].item()
                    
                    # Calculate instruction vs response accuracy
                    if 'instruction' in batch and 'response' in batch:
                        response_accuracy = self._calculate_response_accuracy(
                            outputs['logits'], 
                            batch['labels'],
                            batch.get('attention_mask')
                        )
                        correct_responses += response_accuracy['correct']
                        total_responses += response_accuracy['total']
                    
                    num_batches += 1
                    
                except Exception as e:
                    logging.warning(f"Error in SFT validation batch {batch_idx}: {e}")
                    continue
        
        self.model.train()
        
        metrics = {
            'val_loss': total_loss / max(num_batches, 1),
            'val_response_accuracy': correct_responses / max(total_responses, 1),
        }
        
        return metrics
    
    def _calculate_response_accuracy(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, int]:
        """Calculate accuracy for response tokens only."""
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        
        # Mask for valid tokens (not -100 and not padding)
        valid_mask = (labels != -100)
        if attention_mask is not None:
            valid_mask = valid_mask & (attention_mask.bool())
        
        # Calculate accuracy on valid tokens
        correct = (predictions == labels) & valid_mask
        
        return {
            'correct': correct.sum().item(),
            'total': valid_mask.sum().item()
        }
    
    def _log_metrics(self, loss: float):
        """Log SFT training metrics with streaming dataset info."""
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
            'sft_loss': loss,
            'learning_rate': current_lr,
            'tokens_per_second': tokens_per_sec,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'memory_batch_mb': self.train_dataset.batch_size_mb,
        }
        metrics.update(memory_stats)
        metrics.update(streaming_stats)
        
        # Update tracker
        self.metrics_tracker.update(metrics, self.global_step)
        
        # Log to console
        logging.info(
            f"SFT Step {self.global_step}: loss={loss:.4f}, lr={current_lr:.2e}, "
            f"tokens/s={tokens_per_sec:.0f}, mem_batch={self.train_dataset.batch_size_mb}MB"
        )
        
        # Log to wandb
        if self.wandb:
            self.wandb.log(metrics, step=self.global_step)
        
        self._last_log_time = current_time
        self._last_log_step = self.global_step
    
    def _save_checkpoint(self, final: bool = False):
        """Save SFT model checkpoint."""
        suffix = "final" if final else f"step-{self.global_step}"
        
        TrainerUtils.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            loss=self.metrics_tracker.get_latest('sft_loss'),
            checkpoint_dir=os.path.join(self.config.output_dir, "sft"),
            config=vars(self.config),
            save_format="both"
        )
        
        # Cleanup old checkpoints
        if not final:
            TrainerUtils.cleanup_checkpoints(
                os.path.join(self.config.output_dir, "sft"),
                keep_latest=self.config.save_total_limit
            )
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume SFT training from checkpoint."""
        logging.info(f"Resuming SFT from checkpoint: {checkpoint_path}")
        
        checkpoint_info = TrainerUtils.load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )
        
        self.global_step = checkpoint_info.get('step', 0)
        self.epoch = checkpoint_info.get('epoch', 0)
        
        logging.info(f"Resumed SFT from step {self.global_step}")
        
        return checkpoint_info
    
    def generate_sample(
        self, 
        instruction: str, 
        max_length: int = 200,
        temperature: float = 0.8,
        do_sample: bool = True
    ) -> str:
        """Generate a sample response for evaluation."""
        self.model.eval()
        
        # Format instruction
        if hasattr(self.train_dataset, 'instruction_template'):
            formatted_input = self.train_dataset.instruction_template.format(
                instruction=instruction,
                response=""
            ).rstrip()
        else:
            formatted_input = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_input,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_seq_length - max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        self.model.train()
        return generated_text.strip()
    
    def evaluate_instruction_following(
        self, 
        test_instructions: List[str],
        max_samples: int = 50
    ) -> Dict[str, float]:
        """Evaluate instruction following capabilities."""
        logging.info("Evaluating instruction following...")
        
        results = {
            'avg_response_length': 0.0,
            'non_empty_responses': 0,
            'total_samples': 0
        }
        
        response_lengths = []
        non_empty_count = 0
        
        for i, instruction in enumerate(test_instructions[:max_samples]):
            try:
                response = self.generate_sample(instruction)
                response_length = len(response.split())
                
                response_lengths.append(response_length)
                if len(response.strip()) > 0:
                    non_empty_count += 1
                
                # Log sample responses
                if i < 5:
                    logging.info(f"Sample {i+1}:")
                    logging.info(f"Instruction: {instruction}")
                    logging.info(f"Response: {response[:200]}...")
                    logging.info("-" * 50)
                
            except Exception as e:
                logging.warning(f"Error generating response for instruction {i}: {e}")
        
        results['avg_response_length'] = sum(response_lengths) / max(len(response_lengths), 1)
        results['non_empty_responses'] = non_empty_count
        results['total_samples'] = len(response_lengths)
        results['response_rate'] = non_empty_count / max(len(response_lengths), 1)
        
        return results
