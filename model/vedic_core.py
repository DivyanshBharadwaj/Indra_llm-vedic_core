"""
Vedic philosophical integration for INDRA LLM
(c) Divyansh Bharadwaj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch import Tensor

class VedicExpert(nn.Module):
    """Dedicated expert network trained on Vedic texts."""
    
    def __init__(self, config, frozen: bool = True):
        super().__init__()
        self.config = config
        self.frozen = frozen
        
        # Standard expert architecture
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)
        self.activation_fn = nn.GELU() if config.activation_function == "gelu" else nn.SiLU()
        
        # Vedic-specific components
        self.dharma_projection = nn.Linear(config.n_embd, config.n_embd // 4, bias=False)
        self.karma_projection = nn.Linear(config.n_embd, config.n_embd // 4, bias=False)
        self.ahimsa_gate = nn.Linear(config.n_embd, 1, bias=False)
        
        # Initialize with Vedic-aligned weights
        self._init_vedic_weights()
        
        if self.frozen:
            self._freeze_weights()
    
    def _init_vedic_weights(self):
        """Initialize weights with Vedic-aligned patterns."""
        # Custom initialization that promotes dharmic patterns
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'dharma' in name or 'karma' in name or 'ahimsa' in name:
                    # Initialize with smaller variance to promote stability
                    nn.init.normal_(param, mean=0.0, std=0.01)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.02)
    
    def _freeze_weights(self):
        """Freeze expert weights after Vedic pre-training."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze weights for fine-tuning."""
        self.frozen = False
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass with Vedic principle evaluation.
        
        Returns:
            - output: Standard expert output
            - vedic_metrics: Dict with dharma, karma, ahimsa scores
        """
        # Standard expert computation
        gate = self.activation_fn(self.gate_proj(x))
        up = self.up_proj(x)
        output = self.down_proj(gate * up)
        
        # Vedic principle evaluations
        dharma_score = torch.sigmoid(self.dharma_projection(x)).mean(dim=-1)
        karma_score = torch.sigmoid(self.karma_projection(x)).mean(dim=-1)
        ahimsa_score = torch.sigmoid(self.ahimsa_gate(x)).squeeze(-1)
        
        vedic_metrics = {
            'dharma': dharma_score,
            'karma': karma_score,
            'ahimsa': ahimsa_score
        }
        
        return output, vedic_metrics

class VedicMemoryAdapter(nn.Module):
    """RAG-style adapter for Vedic knowledge integration."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.memory_size = config.vedic.vedic_memory_size
        self.retrieval_top_k = config.vedic.vedic_retrieval_top_k
        self.hidden_dim = config.n_embd
        
        # Memory components
        self.memory_embeddings = nn.Parameter(
            torch.randn(self.memory_size, self.hidden_dim)
        )
        self.memory_values = nn.Parameter(
            torch.randn(self.memory_size, self.hidden_dim)
        )
        
        # Retrieval and integration
        self.query_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.memory_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads=8, batch_first=True
        )
        self.integration_gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        
        # Initialize with Vedic knowledge patterns
        self._init_vedic_memory()
    
    def _init_vedic_memory(self):
        """Initialize memory with Vedic concept embeddings."""
        # This would ideally be loaded from pre-computed Vedic text embeddings
        nn.init.normal_(self.memory_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.memory_values, mean=0.0, std=0.02)
    
    def retrieve_vedic_knowledge(self, query_states: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Retrieve relevant Vedic knowledge based on query states.
        
        Args:
            query_states: [batch_size, seq_len, hidden_dim]
            
        Returns:
            - retrieved_values: [batch_size, seq_len, top_k, hidden_dim]
            - attention_weights: [batch_size, seq_len, top_k]
        """
        batch_size, seq_len, _ = query_states.shape
        
        # Project queries
        queries = self.query_projection(query_states)
        queries = queries.view(-1, self.hidden_dim)  # [batch_size * seq_len, hidden_dim]
        
        # Compute similarity with memory
        similarities = torch.matmul(queries, self.memory_embeddings.T)  # [B*S, memory_size]
        
        # Retrieve top-k most relevant memories
        top_k_scores, top_k_indices = torch.topk(
            similarities, self.retrieval_top_k, dim=-1
        )
        
        # Get retrieved values
        retrieved_values = self.memory_values[top_k_indices]  # [B*S, top_k, hidden_dim]
        attention_weights = F.softmax(top_k_scores, dim=-1)  # [B*S, top_k]
        
        # Reshape back
        retrieved_values = retrieved_values.view(batch_size, seq_len, self.retrieval_top_k, -1)
        attention_weights = attention_weights.view(batch_size, seq_len, self.retrieval_top_k)
        
        return retrieved_values, attention_weights
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Integrate Vedic knowledge into hidden states.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            
        Returns:
            integrated_states: [batch_size, seq_len, hidden_dim]
        """
        # Retrieve relevant Vedic knowledge
        retrieved_values, attention_weights = self.retrieve_vedic_knowledge(hidden_states)
        
        # Weighted combination of retrieved values
        weighted_memory = torch.sum(
            retrieved_values * attention_weights.unsqueeze(-1), dim=2
        )  # [batch_size, seq_len, hidden_dim]
        
        # Integrate with original hidden states
        combined = torch.cat([hidden_states, weighted_memory], dim=-1)
        integrated_states = torch.sigmoid(self.integration_gate(combined)) * hidden_states + \
                           (1 - torch.sigmoid(self.integration_gate(combined))) * weighted_memory
        
        return integrated_states

class VedicRewardModel(nn.Module):
    """Reward model trained to align with Vedic principles."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.n_embd
        
        # Principle-specific reward heads
        self.dharma_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        self.ahimsa_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        self.satya_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        self.karma_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Combined reward head
        self.combined_head = nn.Linear(4, 1)
        
    def forward(self, hidden_states: Tensor) -> Dict[str, Tensor]:
        """
        Compute Vedic-aligned reward scores.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            
        Returns:
            rewards: Dict with individual and combined rewards
        """
        # Use the last token's hidden state for sequence-level reward
        last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_dim]
        
        # Compute individual principle rewards
        dharma_reward = self.dharma_head(last_hidden)  # [batch_size, 1]
        ahimsa_reward = self.ahimsa_head(last_hidden)  # [batch_size, 1]
        satya_reward = self.satya_head(last_hidden)   # [batch_size, 1]
        karma_reward = self.karma_head(last_hidden)   # [batch_size, 1]
        
        # Combine rewards
        principle_rewards = torch.cat([
            dharma_reward, ahimsa_reward, satya_reward, karma_reward
        ], dim=-1)  # [batch_size, 4]
        
        combined_reward = self.combined_head(principle_rewards)  # [batch_size, 1]
        
        return {
            'dharma_reward': dharma_reward,
            'ahimsa_reward': ahimsa_reward,
            'satya_reward': satya_reward,
            'karma_reward': karma_reward,
            'combined_reward': combined_reward,
            'total_reward': combined_reward.squeeze(-1)  # [batch_size]
        }

class VedicIntegrationLayer(nn.Module):
    """Complete Vedic integration layer combining all components."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_vedic_core = config.vedic.use_vedic_core
        
        if self.use_vedic_core:
            self.vedic_expert = VedicExpert(config, frozen=config.vedic.vedic_expert_frozen)
            self.memory_adapter = VedicMemoryAdapter(config)
            self.reward_model = VedicRewardModel(config)
            
            # Integration weights
            self.vedic_weight = nn.Parameter(torch.tensor(config.vedic.vedic_routing_weight))
            self.memory_weight = nn.Parameter(torch.tensor(0.5))
            
    def forward(
        self, 
        hidden_states: Tensor,
        compute_rewards: bool = False
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Apply Vedic integration to hidden states.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            compute_rewards: Whether to compute reward signals
            
        Returns:
            - integrated_states: [batch_size, seq_len, hidden_dim]
            - vedic_outputs: Dict with metrics and optional rewards
        """
        vedic_outputs = {}
        
        if not self.use_vedic_core:
            return hidden_states, vedic_outputs
        
        # Apply Vedic expert
        expert_output, vedic_metrics = self.vedic_expert(hidden_states)
        vedic_outputs.update(vedic_metrics)
        
        # Apply memory adapter
        memory_integrated = self.memory_adapter(hidden_states)
        
        # Combine outputs with learnable weights
        integrated_states = (
            hidden_states + 
            self.vedic_weight * expert_output + 
            self.memory_weight * memory_integrated
        ) / (1 + self.vedic_weight + self.memory_weight)
        
        # Compute rewards if requested
        if compute_rewards:
            rewards = self.reward_model(integrated_states)
            vedic_outputs.update(rewards)
        
        return integrated_states, vedic_outputs
