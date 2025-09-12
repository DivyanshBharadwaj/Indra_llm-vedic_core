"""
Mixture of Experts implementation for INDRA LLM
(c) Divyansh Bharadwaj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from torch import Tensor

class Expert(nn.Module):
    """Single expert network."""
    
    def __init__(self, config, expert_id: int = -1):
        super().__init__()
        self.expert_id = expert_id
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        self.activation_fn = nn.GELU() if config.activation_function == "gelu" else nn.SiLU()
        
    def forward(self, x: Tensor) -> Tensor:
        gate = self.activation_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class TopKRouter(nn.Module):
    """Top-K router for MoE with load balancing."""
    
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe.num_experts
        self.top_k = config.moe.top_k
        self.jitter_noise = config.moe.jitter_noise
        self.gate = nn.Linear(config.n_embd, self.num_experts, bias=False)
        
    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            - router_logits: [batch_size, seq_len, num_experts]
            - selected_experts: [batch_size, seq_len, top_k] 
            - router_weights: [batch_size, seq_len, top_k]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Add jitter noise for better load balancing during training
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(hidden_states) * self.jitter_noise
            hidden_states = hidden_states + noise
        
        router_logits = self.gate(hidden_states)  # [batch_size, seq_len, num_experts]
        
        # Select top-k experts
        router_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        router_weights = F.softmax(router_weights, dim=-1)
        
        return router_logits, selected_experts, router_weights

def load_balancing_loss(router_logits: Tensor, selected_experts: Tensor, num_experts: int) -> Tensor:
    """Compute auxiliary load balancing loss."""
    # Compute the number of tokens routed to each expert
    tokens_per_expert = torch.bincount(
        selected_experts.flatten(),
        minlength=num_experts
    ).float()
    
    # Compute the average probability of routing to each expert
    router_probs = F.softmax(router_logits, dim=-1)
    avg_prob_per_expert = router_probs.mean(dim=(0, 1))
    
    # Load balancing loss encourages uniform distribution
    tokens_per_expert = tokens_per_expert / tokens_per_expert.sum()
    loss = (tokens_per_expert * avg_prob_per_expert).sum() * num_experts
    
    return loss

class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with load balancing."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.moe.num_experts
        self.top_k = config.moe.top_k
        self.aux_loss_weight = config.moe.auxiliary_loss_weight
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(config, expert_id=i) for i in range(self.num_experts)
        ])
        
        # Router
        self.router = TopKRouter(config)
        
    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            
        Returns:
            - output: [batch_size, seq_len, hidden_dim]
            - aux_loss: scalar tensor
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Get routing decisions
        router_logits, selected_experts, router_weights = self.router(hidden_states)
        
        # Initialize output
        final_output = torch.zeros_like(hidden_states)
        
        # Process tokens for each expert
        flat_hidden_states = hidden_states.view(-1, hidden_dim)
        flat_selected_experts = selected_experts.view(-1, self.top_k)
        flat_router_weights = router_weights.view(-1, self.top_k)
        
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (flat_selected_experts == expert_idx)
            expert_tokens_idx = torch.where(expert_mask)
            
            if len(expert_tokens_idx[0]) == 0:
                continue
                
            # Get tokens and weights for this expert
            token_idx = expert_tokens_idx[0]
            position_in_topk = expert_tokens_idx[1]
            
            expert_input = flat_hidden_states[token_idx]
            expert_weights = flat_router_weights[token_idx, position_in_topk].unsqueeze(-1)
            
            # Process through expert
            expert_output = self.experts[expert_idx](expert_input)
            weighted_output = expert_output * expert_weights
            
            # Accumulate weighted outputs
            final_output.view(-1, hidden_dim)[token_idx] += weighted_output
        
        # Compute auxiliary load balancing loss
        aux_loss = load_balancing_loss(router_logits, selected_experts, self.num_experts)
        aux_loss = aux_loss * self.aux_loss_weight
        
        return final_output, aux_loss

class SparseMLP(nn.Module):
    """Sparse MLP that can switch between MoE and standard MLP."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if config.use_moe:
            self.mlp = MixtureOfExperts(config)
        else:
            # Standard MLP
            self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False) 
            self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
            self.activation_fn = nn.GELU() if config.activation_function == "gelu" else nn.SiLU()
            
    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        if self.config.use_moe:
            return self.mlp(hidden_states)
        else:
            # Standard MLP forward
            gate = self.activation_fn(self.gate_proj(hidden_states))
            up = self.up_proj(hidden_states)
            output = self.down_proj(gate * up)
            return output, None
