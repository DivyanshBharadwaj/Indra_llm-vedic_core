"""
Reinforcement Learning from Human Feedback (RLHF) trainer for INDRA LLM
with Vedic-aligned reward model and PPO training
(c) Divyansh Bharadwaj
"""

import os
import time
import logging
import math
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.distributions import Categorical

from .trainer_utils import TrainerUtils, get_optimizer, MetricsTracker
from model import INDRATransformer
from model.vedic_core import VedicRewardModel

@dataclass
class PPOBatch:
    """Batch data structure for PPO training."""
    query_tensors: torch.Tensor
    response_tensors: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    masks: torch.Tensor

class RLHFTrainer:
    """RLHF trainer with PPO and Vedic-aligned rewards."""
    
    def __init__(
        self,
        model: INDRATransformer,
        ref_model: INDRATransformer,
        reward_model: VedicRewardModel,
        config,
        dataset,
        tokenizer=None,
    ):
        """
        Initialize RLHF trainer.
        
        Args:
            model: Policy model (student model to train)
            ref_model: Reference model (frozen copy of initial policy)
            reward_model: Vedic-aligned reward model
            config: Training configuration
            dataset: Prompt dataset for RLHF
            tokenizer: Tokenizer instance
        """
        self.model = model  # Policy model
        self.ref_model = ref_model  # Reference model (frozen)
        self.reward_model = reward_model
        self.config = config
        self.dataset = dataset
        self.tokenizer = tokenizer
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.ref_model = self.ref_model.to(self.device)
        self.reward_model = self.reward_model.to(self.device)
        
        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Setup value function (using model's hidden states)
        self.value_head = nn.Linear(config.n_embd, 1).to(self.device)
        
        # Setup optimizers
        self.policy_optimizer = get_optimizer(
            list(self.model.parameters()) + list(self.value_head.parameters()),
            optimizer_name="adamw",
            learning_rate=config.rlhf.learning_rate,
            weight_decay=1e-6,  # Lower weight decay for RL
        )
        
        # PPO hyperparameters
        self.ppo_epochs = config.rlhf.ppo_epochs
        self.clip_range = config.rlhf.clip_range
        self.clip_range_vf = config.rlhf.clip_range_vf or config.rlhf.clip_range
        self.vf_coef = config.rlhf.vf_coef
        self.ent_coef = config.rlhf.ent_coef
        self.kl_coef = config.rlhf.kl_coef
        self.target_kl = config.rlhf.target_kl
        self.gamma = config.rlhf.gamma
        self.gae_lambda = config.rlhf.gae_lambda
        
        # Vedic alignment
        self.vedic_reward_weight = config.rlhf.vedic_reward_weight
        
        # Setup mixed precision
        self.scaler = GradScaler(enabled=config.use_fp16 or config.use_bf16)
        self.use_amp = config.use_fp16 or config.use_bf16
        self.amp_dtype = torch.float16 if config.use_fp16 else torch.bfloat16
        
        # Setup data loader
        self.data_loader = self._create_data_loader()
        
        # Tracking and logging
        self.metrics_tracker = MetricsTracker(window_size=config.logging_steps)
        self.global_step = 0
        self.epoch = 0
        
        # Statistics tracking
        self.stats = {
            'rewards': deque(maxlen=1000),
            'kl_divergences': deque(maxlen=1000),
            'entropies': deque(maxlen=1000),
        }
        
        # Setup logging
        TrainerUtils.setup_logging()
        self.wandb = TrainerUtils.setup_wandb(
            vars(config), config.wandb_project,
            config.wandb_run_name or "indra-rlhf"
        )
        
        logging.info("RLHF Trainer initialized with Vedic alignment")
        
    def _create_data_loader(self) -> DataLoader:
        """Create data loader for RLHF prompts."""
        return DataLoader(
            self.dataset,
            batch_size=self.config.rlhf.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.pin_memory,
        )
    
    def train(self) -> Dict[str, Any]:
        """Main RLHF training loop with PPO."""
        logging.info("Starting RLHF training with PPO")
        
        start_time = time.time()
        
        while self.global_step < self.config.max_steps:
            # Collect rollouts
            rollout_batch = self._collect_rollouts()
            
            # Compute advantages and returns
            rollout_batch = self._compute_advantages(rollout_batch)
            
            # PPO training steps
            ppo_stats = self._ppo_training_step(rollout_batch)
            
            # Update statistics
            self._update_statistics(rollout_batch, ppo_stats)
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                self._log_metrics(ppo_stats)
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint()
            
            self.global_step += 1
            
            # Early stopping based on KL divergence
            if self._should_early_stop():
                logging.info("Early stopping due to high KL divergence")
                break
            
            if self.global_step >= self.config.max_steps:
                break
        
        # Final save
        self._save_checkpoint(final=True)
        
        total_time = time.time() - start_time
        logging.info(f"RLHF training completed in {total_time:.2f} seconds")
        
        return {
            'final_step': self.global_step,
            'total_time': total_time,
            'final_metrics': self.metrics_tracker.get_summary(),
            'final_reward': self._get_average_reward(),
        }
    
    def _collect_rollouts(self) -> PPOBatch:
        """Collect rollouts from the environment (generate responses)."""
        self.model.eval()
        
        query_tensors = []
        response_tensors = []
        logprobs = []
        values = []
        rewards = []
        masks = []
        
        with torch.no_grad():
            for batch in self.data_loader:
                # Process batch of prompts
                batch_queries = batch['input_ids'].to(self.device)
                batch_attention_mask = batch.get('attention_mask', None)
                
                # Generate responses
                batch_responses, batch_logprobs, batch_values = self._generate_responses(
                    batch_queries, batch_attention_mask
                )
                
                # Compute rewards
                batch_rewards = self._compute_rewards(batch_queries, batch_responses)
                
                # Create masks for padding
                batch_masks = self._create_masks(batch_responses)
                
                # Collect data
                query_tensors.append(batch_queries)
                response_tensors.append(batch_responses)
                logprobs.append(batch_logprobs)
                values.append(batch_values)
                rewards.append(batch_rewards)
                masks.append(batch_masks)
                
                # Break after collecting enough samples
                if sum(len(r) for r in response_tensors) >= self.config.rlhf.batch_size:
                    break
        
        # Concatenate all batches
        query_tensors = torch.cat(query_tensors, dim=0)
        response_tensors = torch.cat(response_tensors, dim=0)
        logprobs = torch.cat(logprobs, dim=0)
        values = torch.cat(values, dim=0)
        rewards = torch.cat(rewards, dim=0)
        masks = torch.cat(masks, dim=0)
        
        return PPOBatch(
            query_tensors=query_tensors,
            response_tensors=response_tensors,
            logprobs=logprobs,
            values=values,
            rewards=rewards,
            advantages=torch.zeros_like(rewards),  # Will be computed later
            returns=torch.zeros_like(rewards),     # Will be computed later
            masks=masks
        )
    
    def _generate_responses(
        self, 
        queries: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate responses and compute log probabilities and values."""
        batch_size = queries.size(0)
        max_new_tokens = 100  # Maximum response length
        
        responses = []
        all_logprobs = []
        all_values = []
        
        for i in range(batch_size):
            query = queries[i:i+1]
            
            # Generate response
            generated_ids = []
            logprobs = []
            values = []
            
            current_input = query
            
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.model(
                    input_ids=current_input,
                    return_dict=True
                )
                
                # Get logits and compute value
                logits = outputs.logits[:, -1, :]  # Last token logits
                hidden_states = outputs.hidden_states[-1][:, -1, :] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state[:, -1, :]
                value = self.value_head(hidden_states).squeeze(-1)
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                next_token = dist.sample()
                
                # Compute log probability
                logprob = dist.log_prob(next_token)
                
                # Append results
                generated_ids.append(next_token.item())
                logprobs.append(logprob)
                values.append(value)
                
                # Update input
                current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)
                
                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
            
            # Convert to tensors
            response = torch.tensor(generated_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            response_logprobs = torch.stack(logprobs) if logprobs else torch.zeros(1, device=self.device)
            response_values = torch.stack(values) if values else torch.zeros(1, device=self.device)
            
            responses.append(response)
            all_logprobs.append(response_logprobs)
            all_values.append(response_values)
        
        # Pad sequences to same length
        max_response_length = max(r.size(1) for r in responses)
        
        padded_responses = []
        padded_logprobs = []
        padded_values = []
        
        for i in range(batch_size):
            response = responses[i]
            logprob = all_logprobs[i]
            value = all_values[i]
            
            # Pad response
            pad_length = max_response_length - response.size(1)
            if pad_length > 0:
                response = F.pad(response, (0, pad_length), value=self.tokenizer.pad_token_id)
                logprob = F.pad(logprob, (0, pad_length), value=0.0)
                value = F.pad(value, (0, pad_length), value=0.0)
            
            padded_responses.append(response)
            padded_logprobs.append(logprob)
            padded_values.append(value)
        
        response_tensors = torch.cat(padded_responses, dim=0)
        logprob_tensors = torch.stack(padded_logprobs)
        value_tensors = torch.stack(padded_values)
        
        return response_tensors, logprob_tensors, value_tensors
    
    def _compute_rewards(
        self,
        queries: torch.Tensor,
        responses: torch.Tensor
    ) -> torch.Tensor:
        """Compute rewards using Vedic reward model and KL penalty."""
        batch_size = queries.size(0)
        rewards = []
        
        for i in range(batch_size):
            query = queries[i:i+1]
            response = responses[i:i+1]
            
            # Combine query and response
            full_sequence = torch.cat([query, response], dim=1)
            
            # Get reward from Vedic reward model
            with torch.no_grad():
                # Forward pass through model to get hidden states
                outputs = self.model(input_ids=full_sequence, return_dict=True)
                hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
                
                # Compute Vedic rewards
                vedic_rewards = self.reward_model(hidden_states)
                base_reward = vedic_rewards['total_reward'].mean()
                
                # Compute KL penalty (compare with reference model)
                ref_outputs = self.ref_model(input_ids=full_sequence, return_dict=True)
                
                current_logits = outputs.logits
                ref_logits = ref_outputs.logits
                
                # KL divergence between current policy and reference
                kl_div = self._compute_kl_divergence(current_logits, ref_logits, response)
                
                # Final reward with KL penalty
                final_reward = base_reward - self.kl_coef * kl_div
                
            rewards.append(final_reward)
        
        return torch.stack(rewards)
    
    def _compute_kl_divergence(
        self,
        current_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        response_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between current and reference policies."""
        # Focus on response tokens only
        response_length = response_tokens.size(1)
        current_response_logits = current_logits[:, -response_length:, :]
        ref_response_logits = ref_logits[:, -response_length:, :]
        
        # Compute distributions
        current_probs = F.softmax(current_response_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_response_logits, dim=-1)
        
        # KL divergence
        kl_div = F.kl_div(ref_log_probs, current_probs, reduction='batchmean')
        
        return kl_div
    
    def _create_masks(self, response_tensors: torch.Tensor) -> torch.Tensor:
        """Create masks for padding tokens."""
        return (response_tensors != self.tokenizer.pad_token_id).float()
    
    def _compute_advantages(self, batch: PPOBatch) -> PPOBatch:
        """Compute advantages and returns using GAE."""
        batch_size, seq_len = batch.response_tensors.shape
        
        advantages = torch.zeros_like(batch.rewards)
        returns = torch.zeros_like(batch.rewards)
        
        for i in range(batch_size):
            # Get rewards and values for this sequence
            rewards = batch.rewards[i]
            values = batch.values[i]
            masks = batch.masks[i]
            
            # Compute GAE advantages
            gae = 0
            for t in reversed(range(seq_len)):
                if t == seq_len - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
                advantages[i, t] = gae * masks[t]
            
            # Compute returns
            returns[i] = advantages[i] + batch.values[i]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        batch.advantages = advantages
        batch.returns = returns
        
        return batch
    
    def _ppo_training_step(self, batch: PPOBatch) -> Dict[str, float]:
        """Perform PPO training step."""
        self.model.train()
        
        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'approx_kl': 0.0,
            'clipfrac': 0.0,
        }
        
        # Create mini-batches
        batch_size = batch.query_tensors.size(0)
        mini_batch_size = self.config.rlhf.mini_batch_size
        num_mini_batches = batch_size // mini_batch_size
        
        indices = torch.randperm(batch_size)
        
        for epoch in range(self.ppo_epochs):
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]
                
                # Extract mini-batch
                mb_queries = batch.query_tensors[mb_indices]
                mb_responses = batch.response_tensors[mb_indices]
                mb_old_logprobs = batch.logprobs[mb_indices]
                mb_advantages = batch.advantages[mb_indices]
                mb_returns = batch.returns[mb_indices]
                mb_masks = batch.masks[mb_indices]
                
                # Forward pass
                mb_logprobs, mb_values, mb_entropy = self._forward_pass(
                    mb_queries, mb_responses
                )
                
                # Compute losses
                policy_loss, value_loss, entropy_loss, approx_kl, clipfrac = self._compute_losses(
                    mb_logprobs, mb_old_logprobs, mb_advantages, 
                    mb_values, mb_returns, mb_entropy, mb_masks
                )
                
                total_loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss
                
                # Backward pass
                self.policy_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.value_head.parameters()),
                    1.0
                )
                self.policy_optimizer.step()
                
                # Update stats
                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy_loss'] += entropy_loss.item()
                stats['total_loss'] += total_loss.item()
                stats['approx_kl'] += approx_kl.item()
                stats['clipfrac'] += clipfrac.item()
                
                # Early stopping within epoch if KL too high
                if approx_kl.item() > 1.5 * self.target_kl:
                    logging.info(f"Early stopping PPO epoch {epoch} due to high KL: {approx_kl.item():.4f}")
                    break
        
        # Average stats
        total_updates = num_mini_batches * self.ppo_epochs
        for key in stats:
            stats[key] /= total_updates
        
        return stats
    
    def _forward_pass(
        self,
        queries: torch.Tensor,
        responses: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass to get log probabilities, values, and entropy."""
        # Combine queries and responses
        full_sequences = torch.cat([queries, responses], dim=1)
        
        # Forward pass through model
        outputs = self.model(input_ids=full_sequences, return_dict=True)
        logits = outputs.logits
        
        # Focus on response part
        response_length = responses.size(1)
        response_logits = logits[:, -response_length-1:-1, :]  # Exclude last token
        
        # Compute log probabilities
        log_probs = F.log_softmax(response_logits, dim=-1)
        response_log_probs = log_probs.gather(2, responses.unsqueeze(-1)).squeeze(-1)
        
        # Compute values using value head
        hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
        response_hidden = hidden_states[:, -response_length:, :]
        values = self.value_head(response_hidden).squeeze(-1)
        
        # Compute entropy
        probs = F.softmax(response_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return response_log_probs, values, entropy
    
    def _compute_losses(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute PPO losses."""
        # Policy loss
        ratio = torch.exp(logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = (policy_loss * masks).sum() / masks.sum()
        
        # Value loss
        value_pred_clipped = values + torch.clamp(
            values - values,  # This should be old_values, but we don't track them
            -self.clip_range_vf,
            self.clip_range_vf
        )
        
        value_loss1 = (values - returns) ** 2
        value_loss2 = (value_pred_clipped - returns) ** 2
        value_loss = torch.max(value_loss1, value_loss2)
        value_loss = (value_loss * masks).sum() / masks.sum()
        
        # Entropy loss
        entropy_loss = (entropy * masks).sum() / masks.sum()
        
        # Approximate KL divergence
        with torch.no_grad():
            approx_kl = ((old_logprobs - logprobs) * masks).sum() / masks.sum()
            
            # Clip fraction
            clipped = (ratio > (1 + self.clip_range)) | (ratio < (1 - self.clip_range))
            clipfrac = (clipped.float() * masks).sum() / masks.sum()
        
        return policy_loss, value_loss, entropy_loss, approx_kl, clipfrac
    
    def _update_statistics(self, batch: PPOBatch, ppo_stats: Dict[str, float]):
        """Update training statistics."""
        # Reward statistics
        rewards = batch.rewards[batch.masks.bool()].cpu().numpy()
        self.stats['rewards'].extend(rewards)
        
        # KL and entropy statistics
        self.stats['kl_divergences'].append(ppo_stats['approx_kl'])
        self.stats['entropies'].append(ppo_stats['entropy_loss'])
    
    def _get_average_reward(self) -> float:
        """Get average reward over recent episodes."""
        if len(self.stats['rewards']) == 0:
            return 0.0
        return sum(self.stats['rewards']) / len(self.stats['rewards'])
    
    def _should_early_stop(self) -> bool:
        """Check if training should stop early."""
        if len(self.stats['kl_divergences']) < 10:
            return False
        
        recent_kl = sum(list(self.stats['kl_divergences'])[-10:]) / 10
        return recent_kl > 2 * self.target_kl
    
    def _log_metrics(self, ppo_stats: Dict[str, float]):
        """Log RLHF training metrics."""
        # Calculate additional metrics
        avg_reward = self._get_average_reward()
        avg_kl = sum(list(self.stats['kl_divergences'])[-10:]) / max(len(list(self.stats['kl_divergences'])[-10:]), 1)
        avg_entropy = sum(list(self.stats['entropies'])[-10:]) / max(len(list(self.stats['entropies'])[-10:]), 1)
        
        # Get memory usage
        memory_stats = TrainerUtils.get_memory_usage()
        
        metrics = {
            'rlhf_policy_loss': ppo_stats['policy_loss'],
            'rlhf_value_loss': ppo_stats['value_loss'],
            'rlhf_entropy_loss': ppo_stats['entropy_loss'],
            'rlhf_total_loss': ppo_stats['total_loss'],
            'rlhf_approx_kl': ppo_stats['approx_kl'],
            'rlhf_clipfrac': ppo_stats['clipfrac'],
            'rlhf_avg_reward': avg_reward,
            'rlhf_avg_kl': avg_kl,
            'rlhf_avg_entropy': avg_entropy,
            'global_step': self.global_step,
        }
        metrics.update(memory_stats)
        
        # Update tracker
        self.metrics_tracker.update(metrics, self.global_step)
        
        # Log to console
        logging.info(
            f"RLHF Step {self.global_step}: "
            f"reward={avg_reward:.4f}, policy_loss={ppo_stats['policy_loss']:.4f}, "
            f"value_loss={ppo_stats['value_loss']:.4f}, kl={ppo_stats['approx_kl']:.4f}"
        )
        
        # Log to wandb
        if self.wandb:
            self.wandb.log(metrics, step=self.global_step)
    
    def _save_checkpoint(self, final: bool = False):
        """Save RLHF model checkpoint."""
        suffix = "final" if final else f"step-{self.global_step}"
        
        # Save both policy model and value head
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'value_head_state_dict': self.value_head.state_dict(),
            'optimizer_state_dict': self.policy_optimizer.state_dict(),
            'step': self.global_step,
            'config': vars(self.config),
            'stats': {
                'avg_reward': self._get_average_reward(),
                'recent_kl': sum(list(self.stats['kl_divergences'])[-10:]) / max(len(list(self.stats['kl_divergences'])[-10:]), 1)
            }
        }
        
        checkpoint_path = os.path.join(self.config.output_dir, "rlhf", f"checkpoint-{suffix}.pt")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved RLHF checkpoint to {checkpoint_path}")
        
        # Cleanup old checkpoints
        if not final:
            TrainerUtils.cleanup_checkpoints(
                os.path.join(self.config.output_dir, "rlhf"),
                keep_latest=self.config.save_total_limit
            )
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume RLHF training from checkpoint."""
        logging.info(f"Resuming RLHF from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.value_head.load_state_dict(checkpoint['value_head_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.global_step = checkpoint.get('step', 0)
        
        # Load statistics if available
        if 'stats' in checkpoint:
            stats = checkpoint['stats']
            logging.info(f"Resuming with avg_reward={stats.get('avg_reward', 0):.4f}")
        
        logging.info(f"Resumed RLHF from step {self.global_step}")
        
        return checkpoint
    
    def evaluate_policy(
        self, 
        eval_prompts: List[str],
        num_samples_per_prompt: int = 5
    ) -> Dict[str, float]:
        """Evaluate the current policy on a set of prompts."""
        self.model.eval()
        
        total_reward = 0.0
        total_samples = 0
        reward_variance = []
        
        with torch.no_grad():
            for prompt in eval_prompts:
                prompt_rewards = []
                
                # Tokenize prompt
                prompt_tokens = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512
                )
                query = prompt_tokens['input_ids'].to(self.device)
                
                for _ in range(num_samples_per_prompt):
                    # Generate response
                    response, _, _ = self._generate_responses(query)
                    
                    # Compute reward
                    reward = self._compute_rewards(query, response)
                    
                    prompt_rewards.append(reward.item())
                    total_reward += reward.item()
                    total_samples += 1
                
                # Track variance within prompt
                if len(prompt_rewards) > 1:
                    prompt_var = sum((r - sum(prompt_rewards)/len(prompt_rewards))**2 for r in prompt_rewards) / len(prompt_rewards)
                    reward_variance.append(prompt_var)
        
        self.model.train()
        
        return {
            'avg_reward': total_reward / max(total_samples, 1),
            'reward_variance': sum(reward_variance) / max(len(reward_variance), 1),
            'total_samples': total_samples
        }
