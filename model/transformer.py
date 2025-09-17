"""
Main transformer architecture for INDRA LLM
(c) Divyansh Bharadwaj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from torch import Tensor

from .attention import GroupedQueryAttention
from .moe import SparseMLP
from .vedic_core import VedicIntegrationLayer
from .hierarchical_reasoning import HierarchicalReasoningModule, VedicReasoningCoordinator, add_hierarchical_reasoning_to_model
from .kv_cache import KVCacheManager, get_cache_manager

class RMSNorm(nn.Module):
    """RMS Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class TransformerBlock(nn.Module):
    """Single transformer block with attention, MLP, and optional Vedic integration."""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Normalization
        if config.use_rms_norm:
            self.input_layernorm = RMSNorm(config.n_embd, config.layer_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.n_embd, config.layer_norm_eps)
        else:
            self.input_layernorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        
        # Attention
        self.self_attn = GroupedQueryAttention(config)
        
        # MLP (with optional MoE)
        self.mlp = SparseMLP(config)
        
        # Vedic integration
        self.vedic_layer = VedicIntegrationLayer(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.residual_dropout)
        
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        compute_vedic_rewards: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]], Dict]:
        """
        Forward pass through transformer block.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len, seq_len] or None
            position_ids: [batch_size, seq_len] or None
            past_key_value: Cached key-value pairs from previous forward passes
            use_cache: Whether to return key-value cache for next iteration
            compute_vedic_rewards: Whether to compute Vedic reward signals
            
        Returns:
            - hidden_states: [batch_size, seq_len, hidden_dim]
            - present_key_value: Optional cached key-value pairs
            - aux_outputs: Dictionary with auxiliary outputs (MoE loss, Vedic metrics)
        """
        aux_outputs = {}
        
        # Pre-attention normalization
        normed_hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        attn_output, present_key_value = self.self_attn(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        
        # Residual connection
        hidden_states = hidden_states + self.dropout(attn_output)
        
        # Pre-MLP normalization
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MLP (potentially MoE)
        mlp_output, aux_loss = self.mlp(normed_hidden_states)
        if aux_loss is not None:
            aux_outputs['moe_aux_loss'] = aux_loss
        
        # Residual connection
        hidden_states = hidden_states + self.dropout(mlp_output)
        
        # Vedic integration
        hidden_states, vedic_outputs = self.vedic_layer(
            hidden_states, compute_rewards=compute_vedic_rewards
        )
        aux_outputs.update(vedic_outputs)
        
        return hidden_states, present_key_value, aux_outputs

class INDRATransformer(nn.Module):
    """
    INDRA: Vedic-aligned Decoder-only Transformer LLM
    
    Main transformer model combining:
    - Grouped Query Attention with FlashAttention
    - Mixture of Experts with load balancing
    - Vedic philosophical integration
    - KV caching for efficient inference
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i) for i in range(config.n_layer)
        ])
        
        # Final layer norm
        if config.use_rms_norm:
            self.norm = RMSNorm(config.n_embd, config.layer_norm_eps)
        else:
            self.norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Hierarchical reasoning module (optional)
        if getattr(config, 'use_hierarchical_reasoning', False):
            from .hierarchical_reasoning import VedicReasoningCoordinator
            self.hierarchical_reasoning = VedicReasoningCoordinator(config)
        else:
            self.hierarchical_reasoning = None
        
        # Gradient checkpointing
        self.gradient_checkpointing = config.gradient_checkpointing
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie embeddings and output weights if specified
        if hasattr(config, 'tie_word_embeddings') and config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
    
    def _init_weights(self, module):
        """Initialize model weights following GPT-style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def prepare_inputs_for_generation(
        self, 
        input_ids: Tensor,
        past_key_values: Optional[List] = None,
        attention_mask: Optional[Tensor] = None,
        **kwargs
    ) -> Dict:
        """Prepare inputs for generation step."""
        if past_key_values is not None:
            # Only use the last token if we have past_key_values
            input_ids = input_ids[:, -1:]
        
        # Create position_ids
        if attention_mask is not None and past_key_values is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = None
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": kwargs.get("use_cache", True),
        }
    
    def create_attention_mask(self, input_ids: Tensor, past_key_values_length: int = 0) -> Tensor:
        """Create causal attention mask."""
        batch_size, seq_length = input_ids.shape
        seq_length_with_past = seq_length + past_key_values_length
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_length_with_past, seq_length_with_past, dtype=torch.bool),
            diagonal=1
        )
        causal_mask = causal_mask.to(input_ids.device)
        
        # Convert to attention mask format (0 for attend, -inf for mask)
        attention_mask = torch.zeros(
            batch_size, 1, seq_length, seq_length_with_past, 
            device=input_ids.device
        )
        attention_mask.masked_fill_(causal_mask[-seq_length:], float('-inf'))
        
        return attention_mask
    
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        compute_vedic_rewards: bool = False,
        enable_reasoning: bool = False,
        reasoning_depth: int = 3,
        return_reasoning_chain: bool = False,
    ) -> Dict:
        """
        Forward pass through the transformer.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] 
            position_ids: [batch_size, seq_len]
            past_key_values: List of cached key-value pairs
            inputs_embeds: [batch_size, seq_len, hidden_dim]
            labels: [batch_size, seq_len] for computing loss
            use_cache: Whether to use KV caching
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dictionary
            compute_vedic_rewards: Whether to compute Vedic alignment rewards
            
        Returns:
            Dictionary containing:
            - logits: [batch_size, seq_len, vocab_size]
            - past_key_values: List of key-value caches
            - hidden_states: List of all layer hidden states (if requested)
            - loss: Language modeling loss (if labels provided)
            - aux_losses: Dictionary of auxiliary losses (MoE, Vedic alignment)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_kv_cache
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else True
    
        # Get inputs - handle validation and embedding creation properly
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            
            # Add input validation and debugging
            # print(f"Input IDs shape: {input_ids.shape}")
            # print(f"Input IDs min/max: {input_ids.min().item()}/{input_ids.max().item()}")
            # print(f"Vocab size: {self.embed_tokens.num_embeddings}")
    
            # Check for out-of-bounds token IDs and fix them
            max_token_id = input_ids.max().item()
            if max_token_id >= self.embed_tokens.num_embeddings:
                unk_token_id = self.embed_tokens.num_embeddings - 1
                print(f"WARNING: Clamping token ID {max_token_id} to {unk_token_id}")
                input_ids = torch.clamp(input_ids, max=self.embed_tokens.num_embeddings - 1)
            
            # Now create embeddings
            inputs_embeds = self.embed_tokens(input_ids)
            
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
            input_ids = None  # Make sure this is None when using inputs_embeds
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")
        
        # Create position_ids if not provided
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values else 0
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length,
                dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Create attention mask if not provided
        if attention_mask is None:
            past_key_values_length = past_key_values[0][0].shape[2] if past_key_values else 0
            # Use input_ids if available, otherwise create dummy tensor for mask creation
            ids_for_mask = input_ids if input_ids is not None else torch.zeros(batch_size, seq_length).long().to(inputs_embeds.device)
            attention_mask = self.create_attention_mask(ids_for_mask, past_key_values_length)
        elif attention_mask.dim() == 2:
            # Convert padding mask to causal attention mask
            batch_size, seq_len = attention_mask.shape
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=attention_mask.device),
                diagonal=1
            )
            attention_mask = attention_mask[:, None, None, :] * ~causal_mask[None, None, :, :]
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)
            attention_mask.masked_fill_(attention_mask == 0, float('-inf'))
        
        # Apply input dropout
        hidden_states = self.embed_dropout(inputs_embeds)
        
        # Initialize outputs
        all_hidden_states = [] if output_hidden_states else None
        next_decoder_cache = [] if use_cache else None
        aux_losses = {
            'moe_aux_loss': 0.0,
            'vedic_alignment_loss': 0.0,
            'total_aux_loss': 0.0
        }
        vedic_metrics = {}
        
        # Pass through each transformer layer
        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            # Handle past key values
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            # Forward through layer (with optional gradient checkpointing)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    use_cache,
                    compute_vedic_rewards,
                    use_reentrant=False
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    compute_vedic_rewards=compute_vedic_rewards,
                )
            
            hidden_states, present_key_value, layer_aux_outputs = layer_outputs
            
            # Accumulate auxiliary losses
            if 'moe_aux_loss' in layer_aux_outputs:
                aux_losses['moe_aux_loss'] += layer_aux_outputs['moe_aux_loss']
            
            # Accumulate Vedic metrics
            for key, value in layer_aux_outputs.items():
                if key not in ['moe_aux_loss']:
                    if key not in vedic_metrics:
                        vedic_metrics[key] = []
                    vedic_metrics[key].append(value)
            
            if use_cache:
                next_decoder_cache.append(present_key_value)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Apply hierarchical reasoning if enabled
        reasoning_outputs = None
        if enable_reasoning and self.hierarchical_reasoning is not None:
            reasoning_outputs = self.hierarchical_reasoning(
                hidden_states,
                reasoning_depth=reasoning_depth,
                return_reasoning_chain=return_reasoning_chain
            )
            
            # Optionally modify logits based on reasoning
            if 'coordinated_output' in reasoning_outputs:
                reasoning_logits = self.lm_head(reasoning_outputs['coordinated_output'])
                # Blend original and reasoning-enhanced logits
                reasoning_weight = 0.3  # Can be made configurable
                logits = (1 - reasoning_weight) * logits + reasoning_weight * reasoning_logits
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels and logits for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Add auxiliary losses
            total_aux_loss = aux_losses['moe_aux_loss']
            
            # Add Vedic alignment loss if computing rewards
            if compute_vedic_rewards and 'combined_reward' in vedic_metrics:
                # Convert rewards to alignment loss (negative reward)
                vedic_rewards = torch.stack([r.mean() for r in vedic_metrics['combined_reward']])
                clipped_reward = torch.clamp(vedic_rewards.mean(), min=-5.0, max=5.0)
                vedic_alignment_loss = -clipped_reward * self.config.vedic.dharmic_alignment_weight 
                aux_losses['vedic_alignment_loss'] = vedic_alignment_loss
                total_aux_loss += vedic_alignment_loss
            
            aux_losses['total_aux_loss'] = total_aux_loss
            loss += total_aux_loss
        
        # Prepare output
        if not return_dict:
            output = (logits,)
            if use_cache:
                output += (next_decoder_cache,)
            if output_hidden_states:
                output += (all_hidden_states,)
            if loss is not None:
                output = (loss,) + output
            return output
        
        result = {
            'loss': loss,
            'logits': logits,
            'past_key_values': next_decoder_cache,
            'hidden_states': all_hidden_states,
            'aux_losses': aux_losses,
            'vedic_metrics': vedic_metrics,
        }
        
        # Add reasoning outputs if available
        if reasoning_outputs is not None:
            result['reasoning_output'] = reasoning_outputs.get('coordinated_output')
            result['confidence_scores'] = reasoning_outputs.get('confidence_scores')
            if return_reasoning_chain:
                result['reasoning_chain'] = reasoning_outputs.get('reasoning_chain', [])
        
        return result
    
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: [batch_size, seq_len] Input token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Repetition penalty factor
            do_sample: Whether to use sampling or greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            use_cache: Whether to use KV caching
            
        Returns:
            generated_ids: [batch_size, seq_len + max_new_tokens]
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Initialize
        generated_ids = input_ids.clone()
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self(
                input_ids=generated_ids if past_key_values is None else generated_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=True,
            )
            
            logits = outputs['logits']
            past_key_values = outputs['past_key_values'] if use_cache else None
            
            # Get next token logits
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated_ids[i].tolist()):
                        if next_token_logits[i, token_id] < 0:
                            next_token_logits[i, token_id] *= repetition_penalty
                        else:
                            next_token_logits[i, token_id] /= repetition_penalty
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Sample next token
            if do_sample:
                # Top-k filtering
                if top_k is not None:
                    top_k = min(top_k, next_token_logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits.fill_(float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits.masked_fill_(indices_to_remove, float('-inf'))
                
                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Update generated_ids
            generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
            
            # Check for EOS tokens
            if eos_token_id is not None:
                finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
                if finished.all():
                    break
        
        return generated_ids


