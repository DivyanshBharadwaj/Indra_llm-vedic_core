"""
Hybrid pre-training with direct knowledge transfer from teacher model (Qwen1.5-MoE)
Modified to use streaming datasets for memory efficiency
(c) Divyansh Bharadwaj
"""

import os
import time
import logging
from typing import Dict, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoModel, AutoTokenizer, AutoConfig

from .trainer_utils import TrainerUtils, get_optimizer, get_scheduler, MetricsTracker
from .pretrain import PretrainTrainer
from model import INDRATransformer

class HybridPretrainTrainer(PretrainTrainer):
    """
    Hybrid pre-training trainer combining:
    1. Direct Knowledge Transfer: Initializing from pre-trained teacher model
    2. Knowledge Distillation: Training to mimic teacher's output distribution
    Modified to use streaming datasets for memory efficiency
    """
    
    def __init__(
        self,
        model: INDRATransformer,
        config,
        train_dataset,  # Now expects StreamingINDRADataset or StreamingVedicDataset
        val_dataset=None,
        tokenizer=None,
        teacher_model_path: str = "Qwen/Qwen1.5-MoE-A2.7B",
    ):
        """
        Initialize hybrid pre-training trainer.
        
        Args:
            model: Student INDRA model
            config: Training configuration
            train_dataset: Streaming training dataset
            val_dataset: Streaming validation dataset
            tokenizer: Tokenizer instance
            teacher_model_path: Path/name of teacher model
        """
        # Teacher model configuration (setup before parent init)
        self.teacher_model_path = teacher_model_path
        self.use_direct_transfer = config.use_direct_transfer
        self.distillation_alpha = config.distillation_alpha
        self.distillation_temperature = config.distillation_temperature
        
        # Initialize teacher model
        self.teacher_model = None
        self.teacher_tokenizer = None
        self.weight_mapping = {}
        
        # Setup teacher model and knowledge transfer
        self._setup_teacher_model()
        
        # Initialize parent trainer
        super().__init__(model, config, train_dataset, val_dataset, tokenizer)
        
        if self.use_direct_transfer:
            self._perform_direct_transfer()
        
        # Additional metrics tracking
        self.distillation_loss_weight = 1.0 - self.distillation_alpha
        
        logging.info(f"Hybrid trainer initialized with teacher: {teacher_model_path}")
        logging.info(f"Direct transfer: {self.use_direct_transfer}")
        logging.info(f"Distillation alpha: {self.distillation_alpha}")
        logging.info(f"Using streaming dataset with {train_dataset.total_files} files")
    
    def _setup_teacher_model(self):
        """Load and setup teacher model."""
        try:
            logging.info(f"Loading teacher model: {self.teacher_model_path}")
            
            # Load teacher model configuration
            teacher_config = AutoConfig.from_pretrained(self.teacher_model_path)
            
            # Load teacher model
            self.teacher_model = AutoModel.from_pretrained(
                self.teacher_model_path,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.device_count() > 1 else None,
                trust_remote_code=True,
            )
            
            # Load teacher tokenizer
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                self.teacher_model_path,
                trust_remote_code=True,
            )
            
            # Move teacher to device and set to eval mode
            if torch.cuda.device_count() <= 1:
                self.teacher_model = self.teacher_model.to(self.device if hasattr(self, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
            self.teacher_model.eval()
            
            # Freeze teacher model
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            
            # Analyze architectures for weight mapping
            self._create_weight_mapping(teacher_config)
            
            logging.info(f"Teacher model loaded successfully")
            logging.info(f"Teacher vocab size: {self.teacher_tokenizer.vocab_size}")
            
        except Exception as e:
            logging.error(f"Failed to load teacher model {self.teacher_model_path}: {e}")
            raise
    
    def _create_weight_mapping(self, teacher_config):
        """Create mapping between teacher and student model weights."""
        student_config = self.model.config
        
        # Create mapping for transferable weights
        self.weight_mapping = {}
        
        # Embedding layers
        if hasattr(teacher_config, 'vocab_size') and hasattr(student_config, 'vocab_size'):
            self.weight_mapping['embed_tokens'] = {
                'teacher_shape': (teacher_config.vocab_size, getattr(teacher_config, 'hidden_size', 4096)),
                'student_shape': (student_config.vocab_size, student_config.n_embd),
                'transfer_method': 'vocab_transfer'
            }
        
        # Transformer layers
        teacher_layers = getattr(teacher_config, 'num_hidden_layers', 32)
        student_layers = student_config.n_layer
        
        # Create layer mapping (may involve layer dropping/repeating)
        layer_mapping = self._create_layer_mapping(teacher_layers, student_layers)
        self.weight_mapping['layers'] = layer_mapping
        
        # Output/LM head
        self.weight_mapping['lm_head'] = {
            'teacher_shape': (teacher_config.vocab_size, getattr(teacher_config, 'hidden_size', 4096)),
            'student_shape': (student_config.vocab_size, student_config.n_embd),
            'transfer_method': 'vocab_transfer'
        }
        
        logging.info(f"Weight mapping created: {len(self.weight_mapping)} components")
    
    def _create_layer_mapping(self, teacher_layers: int, student_layers: int) -> Dict[int, int]:
        """Create mapping between teacher and student layers."""
        if teacher_layers == student_layers:
            # Direct 1:1 mapping
            return {i: i for i in range(student_layers)}
        elif teacher_layers > student_layers:
            # Teacher has more layers - sample evenly
            step = teacher_layers / student_layers
            return {i: int(i * step) for i in range(student_layers)}
        else:
            # Student has more layers - repeat teacher layers
            repeat_factor = student_layers / teacher_layers
            mapping = {}
            for i in range(student_layers):
                teacher_idx = min(int(i / repeat_factor), teacher_layers - 1)
                mapping[i] = teacher_idx
            return mapping
    
    def _perform_direct_transfer(self):
        """Perform direct knowledge transfer from teacher to student."""
        logging.info("Performing direct knowledge transfer...")
        
        try:
            student_state_dict = self.model.state_dict()
            teacher_state_dict = self.teacher_model.state_dict()
            
            transferred_weights = 0
            total_weights = len(student_state_dict)
            
            # Transfer embedding weights
            transferred_weights += self._transfer_embeddings(
                teacher_state_dict, student_state_dict
            )
            
            # Transfer transformer layer weights
            transferred_weights += self._transfer_transformer_layers(
                teacher_state_dict, student_state_dict
            )
            
            # Transfer output head weights
            transferred_weights += self._transfer_output_head(
                teacher_state_dict, student_state_dict
            )
            
            # Load updated state dict
            self.model.load_state_dict(student_state_dict)
            
            logging.info(f"Direct transfer completed: {transferred_weights}/{total_weights} weights transferred")
            
        except Exception as e:
            logging.warning(f"Direct transfer failed: {e}. Continuing with random initialization.")
    
    def _transfer_embeddings(self, teacher_state_dict: Dict, student_state_dict: Dict) -> int:
        """Transfer embedding layer weights."""
        transferred = 0
        
        # Find teacher embedding keys
        teacher_embed_keys = [k for k in teacher_state_dict.keys() if 'embed' in k.lower()]
        student_embed_keys = [k for k in student_state_dict.keys() if 'embed' in k.lower()]
        
        for student_key in student_embed_keys:
            # Try to find matching teacher key
            teacher_key = None
            for tk in teacher_embed_keys:
                if 'token' in tk.lower() and 'token' in student_key.lower():
                    teacher_key = tk
                    break
                elif 'position' in tk.lower() and 'position' in student_key.lower():
                    teacher_key = tk
                    break
            
            if teacher_key:
                transferred += self._transfer_weight_with_vocab_handling(
                    teacher_state_dict[teacher_key],
                    student_state_dict,
                    student_key
                )
        
        return transferred
    
    def _transfer_transformer_layers(self, teacher_state_dict: Dict, student_state_dict: Dict) -> int:
        """Transfer transformer layer weights."""
        transferred = 0
        layer_mapping = self.weight_mapping.get('layers', {})
        
        for student_layer_idx, teacher_layer_idx in layer_mapping.items():
            # Find all weights for this layer
            student_layer_prefix = f"layers.{student_layer_idx}."
            teacher_layer_prefix = f"layers.{teacher_layer_idx}."
            
            # Alternative prefixes for different model architectures
            alt_prefixes = [
                (f"h.{student_layer_idx}.", f"h.{teacher_layer_idx}."),
                (f"decoder.layers.{student_layer_idx}.", f"decoder.layers.{teacher_layer_idx}."),
                (f"transformer.layers.{student_layer_idx}.", f"transformer.layers.{teacher_layer_idx}.")
            ]
            
            # Try different prefix combinations
            for student_prefix, teacher_prefix in [(student_layer_prefix, teacher_layer_prefix)] + alt_prefixes:
                student_keys = [k for k in student_state_dict.keys() if k.startswith(student_prefix)]
                
                if student_keys:
                    for student_key in student_keys:
                        # Generate corresponding teacher key
                        relative_key = student_key[len(student_prefix):]
                        teacher_key = teacher_prefix + relative_key
                        
                        # Try alternative naming conventions
                        if teacher_key not in teacher_state_dict:
                            teacher_key = self._find_alternative_teacher_key(
                                teacher_state_dict, teacher_prefix, relative_key
                            )
                        
                        if teacher_key and teacher_key in teacher_state_dict:
                            transferred += self._transfer_weight_with_shape_handling(
                                teacher_state_dict[teacher_key],
                                student_state_dict,
                                student_key
                            )
                    break
        
        return transferred
    
    def _find_alternative_teacher_key(self, teacher_state_dict: Dict, prefix: str, relative_key: str) -> Optional[str]:
        """Find alternative teacher key with different naming conventions."""
        # Common naming variations
        alternatives = [
            relative_key.replace('self_attn', 'attn'),
            relative_key.replace('attn', 'self_attn'),
            relative_key.replace('mlp', 'feed_forward'),
            relative_key.replace('feed_forward', 'mlp'),
            relative_key.replace('q_proj', 'query'),
            relative_key.replace('k_proj', 'key'),
            relative_key.replace('v_proj', 'value'),
            relative_key.replace('o_proj', 'out_proj'),
            relative_key.replace('gate_proj', 'w1'),
            relative_key.replace('up_proj', 'w3'),
            relative_key.replace('down_proj', 'w2'),
        ]
        
        for alt_key in alternatives:
            full_alt_key = prefix + alt_key
            if full_alt_key in teacher_state_dict:
                return full_alt_key
        
        return None
    
    def _transfer_output_head(self, teacher_state_dict: Dict, student_state_dict: Dict) -> int:
        """Transfer output/LM head weights."""
        transferred = 0
        
        # Find output head keys
        teacher_head_keys = [k for k in teacher_state_dict.keys() 
                           if any(term in k.lower() for term in ['lm_head', 'output', 'head'])]
        student_head_keys = [k for k in student_state_dict.keys() 
                           if any(term in k.lower() for term in ['lm_head', 'output', 'head'])]
        
        for student_key in student_head_keys:
            for teacher_key in teacher_head_keys:
                if 'weight' in student_key and 'weight' in teacher_key:
                    transferred += self._transfer_weight_with_vocab_handling(
                        teacher_state_dict[teacher_key],
                        student_state_dict,
                        student_key
                    )
                    break
        
        return transferred
    
    def _transfer_weight_with_vocab_handling(
        self, 
        teacher_weight: torch.Tensor, 
        student_state_dict: Dict, 
        student_key: str
    ) -> int:
        """Transfer weight with vocabulary size handling."""
        student_weight = student_state_dict[student_key]
        
        if teacher_weight.shape == student_weight.shape:
            # Direct transfer
            student_state_dict[student_key] = teacher_weight.clone()
            return 1
        
        # Handle vocabulary size differences
        if len(teacher_weight.shape) == 2 and len(student_weight.shape) == 2:
            # Matrix weight (e.g., embedding, lm_head)
            teacher_vocab, teacher_dim = teacher_weight.shape
            student_vocab, student_dim = student_weight.shape
            
            # Transfer common dimensions
            min_vocab = min(teacher_vocab, student_vocab)
            min_dim = min(teacher_dim, student_dim)
            
            # Initialize with teacher weights for common part
            student_state_dict[student_key][:min_vocab, :min_dim] = teacher_weight[:min_vocab, :min_dim]
            
            # For remaining dimensions, use scaled versions or random init
            if student_dim > teacher_dim:
                # Replicate teacher dimensions
                repeat_factor = student_dim // teacher_dim
                remainder = student_dim % teacher_dim
                
                for i in range(repeat_factor):
                    start_idx = teacher_dim + i * teacher_dim
                    end_idx = start_idx + teacher_dim
                    student_state_dict[student_key][:min_vocab, start_idx:end_idx] = teacher_weight[:min_vocab, :]
                
                if remainder > 0:
                    start_idx = teacher_dim + repeat_factor * teacher_dim
                    student_state_dict[student_key][:min_vocab, start_idx:start_idx + remainder] = \
                        teacher_weight[:min_vocab, :remainder]
            
            return 1
        
        return 0
    
    def _transfer_weight_with_shape_handling(
        self, 
        teacher_weight: torch.Tensor, 
        student_state_dict: Dict, 
        student_key: str
    ) -> int:
        """Transfer weight with general shape handling."""
        student_weight = student_state_dict[student_key]
        
        if teacher_weight.shape == student_weight.shape:
            # Direct transfer
            student_state_dict[student_key] = teacher_weight.clone()
            return 1
        
        # Handle dimension mismatches
        if teacher_weight.dim() == student_weight.dim():
            # Same number of dimensions - transfer compatible parts
            min_dims = [min(t, s) for t, s in zip(teacher_weight.shape, student_weight.shape)]
            
            # Create slices for each dimension
            teacher_slices = tuple(slice(0, dim) for dim in min_dims)
            student_slices = tuple(slice(0, dim) for dim in min_dims)
            
            student_state_dict[student_key][student_slices] = teacher_weight[teacher_slices]
            return 1
        
        return 0
    
    def _compute_distillation_loss(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute knowledge distillation loss."""
        # Temperature scaling
        student_log_probs = F.log_softmax(
            student_logits / self.distillation_temperature, dim=-1
        )
        teacher_probs = F.softmax(
            teacher_logits / self.distillation_temperature, dim=-1
        )
        
        # KL divergence loss
        distillation_loss = F.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction='batchmean'
        )
        
        # Scale by temperature squared (standard practice)
        distillation_loss *= self.distillation_temperature ** 2
        
        return distillation_loss
    
    def _get_teacher_outputs(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get teacher model outputs for distillation."""
        with torch.no_grad():
            # Convert student tokens to teacher tokens if vocabularies differ
            teacher_input_ids = self._convert_tokens_for_teacher(batch['input_ids'])
            
            # Get teacher outputs
            teacher_outputs = self.teacher_model(
                input_ids=teacher_input_ids,
                attention_mask=batch.get('attention_mask'),
                return_dict=True
            )
            
            # Extract logits
            if hasattr(teacher_outputs, 'logits'):
                teacher_logits = teacher_outputs.logits
            elif hasattr(teacher_outputs, 'last_hidden_state'):
                # If no direct logits, apply teacher's lm_head if available
                hidden_states = teacher_outputs.last_hidden_state
                if hasattr(self.teacher_model, 'lm_head'):
                    teacher_logits = self.teacher_model.lm_head(hidden_states)
                else:
                    # Fallback: create dummy logits
                    batch_size, seq_len, hidden_size = hidden_states.shape
                    teacher_logits = torch.zeros(
                        batch_size, seq_len, self.config.vocab_size,
                        device=hidden_states.device,
                        dtype=hidden_states.dtype
                    )
            else:
                # Fallback: create dummy logits
                batch_size, seq_len = teacher_input_ids.shape
                teacher_logits = torch.zeros(
                    batch_size, seq_len, self.config.vocab_size,
                    device=teacher_input_ids.device,
                    dtype=torch.float32
                )
            
            return teacher_logits
    
    def _convert_tokens_for_teacher(self, student_input_ids: torch.Tensor) -> torch.Tensor:
        """Convert student tokens to teacher vocabulary."""
        if self.teacher_tokenizer is None or self.tokenizer is None:
            return student_input_ids
        
        # For now, return as-is (assuming compatible vocabularies)
        # In practice, you might need vocabulary mapping
        return student_input_ids
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step with hybrid loss and streaming dataset batch."""
        try:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Get teacher outputs for distillation
            teacher_logits = None
            if self.distillation_alpha < 1.0 and self.teacher_model is not None:
                try:
                    teacher_logits = self._get_teacher_outputs(batch)
                except Exception as e:
                    logging.warning(f"Failed to get teacher outputs: {e}")
            
            # Forward pass with mixed precision
            use_autocast = (self.use_amp and torch.cuda.is_available())
            
            if use_autocast:
                with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        labels=batch.get('labels'),
                        compute_vedic_rewards=True
                    )
                    
                    # Standard language modeling loss
                    lm_loss = outputs['loss']
                    
                    # Compute distillation loss
                    distillation_loss = 0.0
                    if teacher_logits is not None:
                        student_logits = outputs['logits']
                        
                        # Align dimensions if necessary
                        if student_logits.shape[-1] != teacher_logits.shape[-1]:
                            # Truncate to smaller vocabulary
                            min_vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
                            student_logits = student_logits[..., :min_vocab]
                            teacher_logits = teacher_logits[..., :min_vocab]
                        
                        distillation_loss = self._compute_distillation_loss(
                            student_logits, teacher_logits
                        )
                    
                    # Combine losses
                    total_loss = (
                        self.distillation_alpha * lm_loss + 
                        self.distillation_loss_weight * distillation_loss
                    )
            else:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch.get('labels'),
                    compute_vedic_rewards=True
                )
                
                # Standard language modeling loss
                lm_loss = outputs['loss']
                
                # Compute distillation loss
                distillation_loss = 0.0
                if teacher_logits is not None:
                    student_logits = outputs['logits']
                    
                    # Align dimensions if necessary
                    if student_logits.shape[-1] != teacher_logits.shape[-1]:
                        # Truncate to smaller vocabulary
                        min_vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
                        student_logits = student_logits[..., :min_vocab]
                        teacher_logits = teacher_logits[..., :min_vocab]
                    
                    distillation_loss = self._compute_distillation_loss(
                        student_logits, teacher_logits
                    )
                
                # Combine losses
                total_loss = (
                    self.distillation_alpha * lm_loss + 
                    self.distillation_loss_weight * distillation_loss
                )
            
            # Scale loss for gradient accumulation
            total_loss = total_loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
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
            
            # Return individual loss components for logging
            return {
                'total_loss': total_loss.item() * self.config.gradient_accumulation_steps,
                'lm_loss': lm_loss.item() if isinstance(lm_loss, torch.Tensor) else lm_loss,
                'distillation_loss': distillation_loss.item() if isinstance(distillation_loss, torch.Tensor) else distillation_loss
            }

        except Exception as e:
            logging.error(f"Error in hybrid training step: {e}")
            # For streaming, we can skip this batch and continue
            return {
                'total_loss': 0.0,
                'lm_loss': 0.0,
                'distillation_loss': 0.0
            }
    
    def _train_epoch(self) -> float:
        """Train for one epoch with hybrid losses and streaming datasets."""
        epoch_losses = {'total_loss': 0.0, 'lm_loss': 0.0, 'distillation_loss': 0.0}
        num_batches = 0
        
        # Streaming datasets provide infinite iteration until we break
        for batch_idx, batch in enumerate(self.train_loader):
            if self.global_step >= self.config.max_steps:
                logging.info(f"Reached max steps ({self.config.max_steps}), stopping training")
                break
            
            try:
                losses = self._train_step(batch)
                
                for key, value in losses.items():
                    epoch_losses[key] += value
                
                num_batches += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_hybrid_metrics(losses)
                
                # Update curriculum phase
                self._update_curriculum_phase()

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
                logging.error(f"Error in training batch {batch_idx}: {e}")
                # For streaming, we can continue with next batch
                continue
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses['total_loss']
    
    def _log_hybrid_metrics(self, losses: Dict[str, float]):
        """Log hybrid training metrics with streaming dataset info."""
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
            'train_total_loss': losses['total_loss'],
            'train_lm_loss': losses['lm_loss'],
            'train_distillation_loss': losses['distillation_loss'],
            'learning_rate': current_lr,
            'tokens_per_second': tokens_per_sec,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'curriculum_phase': self.current_phase,
            'distillation_alpha': self.distillation_alpha,
            'memory_batch_mb': self.train_dataset.batch_size_mb,
        }
        metrics.update(memory_stats)
        metrics.update(streaming_stats)
        
        # Update tracker
        self.metrics_tracker.update(metrics, self.global_step)
        
        # Log to console
        logging.info(
            f"Step {self.global_step}: total_loss={losses['total_loss']:.4f}, "
            f"lm_loss={losses['lm_loss']:.4f}, distill_loss={losses['distillation_loss']:.4f}, "
            f"lr={current_lr:.2e}, tokens/s={tokens_per_sec:.0f}, phase={self.current_phase}, "
            f"mem_batch={self.train_dataset.batch_size_mb}MB"
        )
        
        # Log to wandb
        if self.wandb:
            self.wandb.log(metrics, step=self.global_step)
        
        self._last_log_time = current_time
        self._last_log_step = self.global_step
