"""
Hierarchical reasoning model for INDRA LLM with Vedic philosophical integration
(c) Divyansh Bharadwaj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass

class ReasoningLevel(Enum):
    """Levels of hierarchical reasoning."""
    SURFACE = 0      # Direct factual responses
    LOGICAL = 1      # Logical inference and deduction
    CONCEPTUAL = 2   # Abstract concept relationships
    PHILOSOPHICAL = 3 # Deep philosophical reasoning
    TRANSCENDENT = 4  # Vedic transcendent wisdom

@dataclass
class ReasoningStep:
    """Single step in hierarchical reasoning chain."""
    level: ReasoningLevel
    content: str
    confidence: float
    vedic_principles: List[str]
    supporting_evidence: Optional[str] = None

class HierarchicalReasoningModule(nn.Module):
    """
    Hierarchical reasoning model that processes information at multiple abstraction levels,
    integrating Vedic philosophical principles at each level.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.n_embd
        self.num_levels = len(ReasoningLevel)
        
        # Level-specific processing layers
        self.level_processors = nn.ModuleList([
            LevelProcessor(config, level) for level in ReasoningLevel
        ])
        
        # Inter-level communication
        self.level_communication = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_levels - 1)
        ])
        
        # Vedic integration at each level
        self.vedic_integrators = nn.ModuleList([
            VedicLevelIntegrator(config, level) for level in ReasoningLevel
        ])
        
        # Reasoning chain synthesis
        self.chain_synthesizer = ReasoningChainSynthesizer(config)
        
        # Output projection
        self.reasoning_output = nn.Linear(self.hidden_dim * self.num_levels, self.hidden_dim)
        
        # Confidence estimation
        self.confidence_estimator = ConfidenceEstimator(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        reasoning_query: Optional[torch.Tensor] = None,
        vedic_context: Optional[Dict[str, torch.Tensor]] = None,
        return_reasoning_chain: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical reasoning.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_dim]
            reasoning_query: Optional specific query for reasoning
            vedic_context: Optional Vedic contextual information
            return_reasoning_chain: Whether to return detailed reasoning steps
            
        Returns:
            Dictionary containing reasoning outputs and optional chain
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Initialize level states
        level_states = []
        reasoning_chain = []
        
        current_state = hidden_states
        
        # Process through each reasoning level
        for level_idx, (processor, vedic_integrator) in enumerate(
            zip(self.level_processors, self.vedic_integrators)
        ):
            level = ReasoningLevel(level_idx)
            
            # Process at current level
            level_output = processor(
                current_state,
                reasoning_query=reasoning_query,
                context_from_below=current_state if level_idx > 0 else None
            )
            
            # Integrate Vedic principles
            vedic_enhanced_output = vedic_integrator(
                level_output,
                vedic_context=vedic_context,
                level=level
            )
            
            level_states.append(vedic_enhanced_output['enhanced_states'])
            
            if return_reasoning_chain:
                reasoning_chain.append({
                    'level': level,
                    'reasoning_content': vedic_enhanced_output.get('reasoning_content'),
                    'vedic_principles': vedic_enhanced_output.get('vedic_principles', []),
                    'confidence': vedic_enhanced_output.get('confidence', 0.0)
                })
            
            # Prepare state for next level
            if level_idx < self.num_levels - 1:
                # Inter-level communication
                next_state = self.level_communication[level_idx](vedic_enhanced_output['enhanced_states'])
                current_state = current_state + next_state  # Residual connection
        
        # Synthesize reasoning chain
        if len(level_states) > 1:
            synthesized_reasoning = self.chain_synthesizer(
                level_states, reasoning_chain if return_reasoning_chain else None
            )
        else:
            synthesized_reasoning = level_states[0]
        
        # Final output projection
        concatenated_states = torch.cat(level_states, dim=-1)
        reasoning_output = self.reasoning_output(concatenated_states)
        
        # Estimate confidence
        confidence_scores = self.confidence_estimator(reasoning_output, level_states)
        
        result = {
            'reasoning_output': reasoning_output,
            'synthesized_reasoning': synthesized_reasoning,
            'confidence_scores': confidence_scores,
            'level_states': level_states
        }
        
        if return_reasoning_chain:
            result['reasoning_chain'] = reasoning_chain
        
        return result

class LevelProcessor(nn.Module):
    """Processor for a specific reasoning level."""
    
    def __init__(self, config, level: ReasoningLevel):
        super().__init__()
        self.config = config
        self.level = level
        self.hidden_dim = config.n_embd
        
        # Level-specific parameters
        if level == ReasoningLevel.SURFACE:
            self.processing_layers = 2
            self.attention_heads = 4
        elif level == ReasoningLevel.LOGICAL:
            self.processing_layers = 3
            self.attention_heads = 6
        elif level == ReasoningLevel.CONCEPTUAL:
            self.processing_layers = 4
            self.attention_heads = 8
        elif level == ReasoningLevel.PHILOSOPHICAL:
            self.processing_layers = 5
            self.attention_heads = 10
        else:  # TRANSCENDENT
            self.processing_layers = 6
            self.attention_heads = 12
        
        # Multi-layer processing
        self.processing_stack = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.attention_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(self.processing_layers)
        ])
        
        # Level-specific projections
        self.query_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.context_integration = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        
    def forward(
        self,
        input_states: torch.Tensor,
        reasoning_query: Optional[torch.Tensor] = None,
        context_from_below: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process input through this reasoning level."""
        
        current_states = input_states
        
        # Integrate context from lower level if available
        if context_from_below is not None:
            combined_context = torch.cat([current_states, context_from_below], dim=-1)
            current_states = self.context_integration(combined_context)
        
        # Apply query-specific processing if query provided
        if reasoning_query is not None:
            query_enhanced = self.query_projection(reasoning_query)
            current_states = current_states + query_enhanced
        
        # Process through transformer layers
        for layer in self.processing_stack:
            current_states = layer(current_states)
        
        # Output normalization
        output_states = self.output_norm(current_states)
        
        return output_states

class VedicLevelIntegrator(nn.Module):
    """Integrates Vedic principles at each reasoning level."""
    
    def __init__(self, config, level: ReasoningLevel):
        super().__init__()
        self.config = config
        self.level = level
        self.hidden_dim = config.n_embd
        
        # Vedic principle embeddings for this level
        self.principle_embeddings = nn.ModuleDict({
            'dharma': nn.Linear(self.hidden_dim, self.hidden_dim),
            'karma': nn.Linear(self.hidden_dim, self.hidden_dim),
            'ahimsa': nn.Linear(self.hidden_dim, self.hidden_dim),
            'satya': nn.Linear(self.hidden_dim, self.hidden_dim),
            'moksha': nn.Linear(self.hidden_dim, self.hidden_dim)
        })
        
        # Level-specific Vedic integration weights
        self.vedic_weights = self._get_level_vedic_weights(level)
        
        # Integration mechanism
        self.vedic_attention = nn.MultiheadAttention(
            self.hidden_dim, num_heads=8, batch_first=True
        )
        
        # Output synthesis
        self.integration_synthesis = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        
        # Reasoning content generator
        self.content_generator = nn.Linear(self.hidden_dim, self.hidden_dim)
        
    def _get_level_vedic_weights(self, level: ReasoningLevel) -> Dict[str, float]:
        """Get Vedic principle weights for specific reasoning level."""
        if level == ReasoningLevel.SURFACE:
            return {'dharma': 0.3, 'karma': 0.2, 'ahimsa': 0.2, 'satya': 0.2, 'moksha': 0.1}
        elif level == ReasoningLevel.LOGICAL:
            return {'dharma': 0.4, 'karma': 0.3, 'ahimsa': 0.1, 'satya': 0.2, 'moksha': 0.0}
        elif level == ReasoningLevel.CONCEPTUAL:
            return {'dharma': 0.3, 'karma': 0.2, 'ahimsa': 0.2, 'satya': 0.1, 'moksha': 0.2}
        elif level == ReasoningLevel.PHILOSOPHICAL:
            return {'dharma': 0.2, 'karma': 0.2, 'ahimsa': 0.2, 'satya': 0.2, 'moksha': 0.2}
        else:  # TRANSCENDENT
            return {'dharma': 0.1, 'karma': 0.1, 'ahimsa': 0.1, 'satya': 0.1, 'moksha': 0.6}
    
    def forward(
        self,
        level_states: torch.Tensor,
        vedic_context: Optional[Dict[str, torch.Tensor]] = None,
        level: Optional[ReasoningLevel] = None
    ) -> Dict[str, torch.Tensor]:
        """Integrate Vedic principles into level processing."""
        
        # Generate Vedic principle representations
        vedic_representations = []
        active_principles = []
        
        for principle, weight in self.vedic_weights.items():
            if weight > 0.1:  # Only include significant principles
                principle_repr = self.principle_embeddings[principle](level_states)
                vedic_representations.append(principle_repr * weight)
                active_principles.append(principle)
        
        # Combine Vedic representations
        if vedic_representations:
            combined_vedic = torch.stack(vedic_representations).sum(dim=0)
            
            # Apply Vedic attention
            vedic_enhanced, attention_weights = self.vedic_attention(
                level_states, combined_vedic, combined_vedic
            )
            
            # Synthesize with original states
            integrated_states = self.integration_synthesis(
                torch.cat([level_states, vedic_enhanced], dim=-1)
            )
        else:
            integrated_states = level_states
            attention_weights = None
        
        # Generate reasoning content representation
        reasoning_content = self.content_generator(integrated_states)
        
        # Estimate confidence based on Vedic alignment
        vedic_alignment_score = self._calculate_vedic_alignment(integrated_states, active_principles)
        
        return {
            'enhanced_states': integrated_states,
            'reasoning_content': reasoning_content,
            'vedic_principles': active_principles,
            'confidence': vedic_alignment_score,
            'attention_weights': attention_weights
        }
    
    def _calculate_vedic_alignment(
        self, 
        states: torch.Tensor, 
        principles: List[str]
    ) -> float:
        """Calculate alignment with Vedic principles."""
        # Simple heuristic based on principle activation
        if not principles:
            return 0.0
        
        # Calculate normalized activation strength
        activation_strength = torch.mean(torch.abs(states)).item()
        principle_coverage = len(principles) / 5.0  # 5 total principles
        
        alignment_score = min(1.0, activation_strength * principle_coverage)
        return alignment_score

class ReasoningChainSynthesizer(nn.Module):
    """Synthesizes reasoning across all levels into coherent output."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.n_embd
        self.num_levels = len(ReasoningLevel)
        
        # Cross-level attention
        self.cross_level_attention = nn.ModuleList([
            nn.MultiheadAttention(self.hidden_dim, num_heads=8, batch_first=True)
            for _ in range(self.num_levels - 1)
        ])
        
        # Synthesis layers
        self.synthesis_layers = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_levels, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Chain coherence validator
        self.coherence_validator = nn.Linear(self.hidden_dim, 1)
        
    def forward(
        self, 
        level_states: List[torch.Tensor],
        reasoning_chain: Optional[List[Dict]] = None
    ) -> torch.Tensor:
        """Synthesize reasoning chain across levels."""
        
        # Apply cross-level attention
        enhanced_states = []
        
        for i, state in enumerate(level_states):
            if i == 0:
                enhanced_states.append(state)
            else:
                # Attend to all previous levels
                attended_state = state
                for j in range(i):
                    attended, _ = self.cross_level_attention[j](
                        state, level_states[j], level_states[j]
                    )
                    attended_state = attended_state + attended
                
                enhanced_states.append(attended_state / (i + 1))  # Normalize
        
        # Concatenate all enhanced level states
        concatenated = torch.cat(enhanced_states, dim=-1)
        
        # Synthesize into final reasoning output
        synthesized = self.synthesis_layers(concatenated)
        
        # Validate chain coherence
        coherence_score = torch.sigmoid(self.coherence_validator(synthesized))
        
        # Apply coherence weighting
        final_output = synthesized * coherence_score
        
        return final_output

class ConfidenceEstimator(nn.Module):
    """Estimates confidence in reasoning outputs."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.n_embd
        
        # Confidence estimation network
        self.confidence_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Multi-level confidence aggregation
        self.level_confidence_weights = nn.Parameter(
            torch.ones(len(ReasoningLevel)) / len(ReasoningLevel)
        )
        
    def forward(
        self,
        reasoning_output: torch.Tensor,
        level_states: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Estimate confidence scores."""
        
        # Overall confidence from final output
        overall_confidence = self.confidence_net(reasoning_output)
        
        # Level-specific confidences
        level_confidences = []
        for state in level_states:
            level_conf = self.confidence_net(state)
            level_confidences.append(level_conf)
        
        # Weighted aggregate confidence
        level_conf_stack = torch.stack(level_confidences, dim=0)  # [num_levels, batch, seq, 1]
        weights = F.softmax(self.level_confidence_weights, dim=0).view(-1, 1, 1, 1)
        weighted_confidence = (level_conf_stack * weights).sum(dim=0)
        
        return {
            'overall_confidence': overall_confidence,
            'level_confidences': level_confidences,
            'weighted_confidence': weighted_confidence
        }

class VedicReasoningCoordinator(nn.Module):
    """
    Coordinates hierarchical reasoning with Vedic core memory and principles.
    This connects the hierarchical reasoning to the existing Vedic core.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.n_embd
        
        # Hierarchical reasoning module
        self.hierarchical_reasoner = HierarchicalReasoningModule(config)
        
        # Integration with Vedic core
        self.vedic_memory_interface = VedicMemoryInterface(config)
        
        # Reasoning query generator
        self.query_generator = ReasoningQueryGenerator(config)
        
        # Output coordinator
        self.output_coordinator = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        reasoning_prompt: Optional[torch.Tensor] = None,
        vedic_memory_context: Optional[Dict[str, torch.Tensor]] = None,
        reasoning_depth: int = 3,
        return_reasoning_chain: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Coordinate hierarchical reasoning with Vedic principles.
        
        Args:
            hidden_states: Input hidden states
            reasoning_prompt: Optional specific reasoning prompt
            vedic_memory_context: Context from Vedic memory
            reasoning_depth: Depth of reasoning (1-5)
            return_reasoning_chain: Whether to return detailed chain
            
        Returns:
            Coordinated reasoning output with Vedic integration
        """
        
        # Generate reasoning query if not provided
        if reasoning_prompt is None:
            reasoning_query = self.query_generator(hidden_states)
        else:
            reasoning_query = reasoning_prompt
        
        # Retrieve relevant Vedic context
        vedic_context = self.vedic_memory_interface.retrieve_context(
            hidden_states, reasoning_query, vedic_memory_context
        )
        
        # Apply hierarchical reasoning with Vedic integration
        reasoning_output = self.hierarchical_reasoner(
            hidden_states,
            reasoning_query=reasoning_query,
            vedic_context=vedic_context,
            return_reasoning_chain=return_reasoning_chain
        )
        
        # Coordinate with original hidden states
        coordinated_output = self.output_coordinator(
            torch.cat([hidden_states, reasoning_output['reasoning_output']], dim=-1)
        )
        
        result = {
            'coordinated_output': coordinated_output,
            'reasoning_output': reasoning_output['reasoning_output'],
            'confidence_scores': reasoning_output['confidence_scores'],
            'vedic_context_used': vedic_context
        }
        
        if return_reasoning_chain:
            result['reasoning_chain'] = reasoning_output.get('reasoning_chain', [])
            result['level_states'] = reasoning_output.get('level_states', [])
        
        return result

class VedicMemoryInterface(nn.Module):
    """Interface to Vedic memory for reasoning context."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.n_embd
        
        # Memory query projection
        self.memory_query_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        
        # Context synthesis
        self.context_synthesis = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        
    def retrieve_context(
        self,
        hidden_states: torch.Tensor,
        reasoning_query: torch.Tensor,
        vedic_memory_context: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Retrieve relevant Vedic context for reasoning."""
        
        # Combine hidden states and query for memory retrieval
        combined_query = torch.cat([hidden_states.mean(dim=1), reasoning_query.mean(dim=1)], dim=-1)
        memory_query = self.memory_query_proj(combined_query)
        
        # If Vedic memory context provided, synthesize with query
        if vedic_memory_context:
            context_keys = ['dharma_context', 'karma_context', 'wisdom_context']
            available_contexts = []
            
            for key in context_keys:
                if key in vedic_memory_context:
                    available_contexts.append(vedic_memory_context[key])
            
            if available_contexts:
                # Simple averaging of available contexts
                avg_context = torch.stack(available_contexts).mean(dim=0)
                
                # Synthesize query, states, and context
                synthesized_context = self.context_synthesis(
                    torch.cat([
                        memory_query,
                        hidden_states.mean(dim=1),
                        avg_context
                    ], dim=-1)
                )
            else:
                synthesized_context = memory_query
        else:
            synthesized_context = memory_query
        
        return {
            'synthesized_context': synthesized_context,
            'memory_query': memory_query,
            'available_contexts': vedic_memory_context or {}
        }

class ReasoningQueryGenerator(nn.Module):
    """Generates reasoning queries from hidden states."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.n_embd
        
        # Query generation network
        self.query_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Query type classifier
        self.query_type_classifier = nn.Linear(self.hidden_dim, 5)  # 5 reasoning levels
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Generate reasoning query from hidden states."""
        
        # Use mean pooling to get sequence representation
        sequence_repr = hidden_states.mean(dim=1, keepdim=True)
        sequence_repr = sequence_repr.expand(-1, hidden_states.size(1), -1)
        
        # Generate query
        reasoning_query = self.query_generator(sequence_repr)
        
        # Classify query type (for potential future use)
        query_type_logits = self.query_type_classifier(reasoning_query.mean(dim=1))
        
        return reasoning_query

# Integration function to add hierarchical reasoning to existing model
def add_hierarchical_reasoning_to_model(model: 'INDRATransformer') -> 'INDRATransformer':
    """
    Add hierarchical reasoning capabilities to existing INDRA model.
    
    Args:
        model: Existing INDRA transformer model
        
    Returns:
        Enhanced model with hierarchical reasoning
    """
    # Add hierarchical reasoning coordinator
    model.hierarchical_reasoning = VedicReasoningCoordinator(model.config)
    
    # Add reasoning integration to forward pass
    original_forward = model.forward
    
    def enhanced_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=None,
        enable_reasoning=False,
        reasoning_depth=3,
        return_reasoning_chain=False,
        **kwargs
    ):
        # Call original forward
        outputs = original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True,  # Force hidden states for reasoning
            return_dict=True,
            **kwargs
        )
        
        # Apply hierarchical reasoning if enabled
        if enable_reasoning and hasattr(self, 'hierarchical_reasoning'):
            hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
            
            reasoning_outputs = self.hierarchical_reasoning(
                hidden_states,
                reasoning_depth=reasoning_depth,
                return_reasoning_chain=return_reasoning_chain
            )
            
            # Update outputs with reasoning
            outputs.reasoning_output = reasoning_outputs['coordinated_output']
            outputs.confidence_scores = reasoning_outputs['confidence_scores']
            
            if return_reasoning_chain:
                outputs.reasoning_chain = reasoning_outputs.get('reasoning_chain', [])
        
        return outputs
    
    # Bind enhanced forward method
    model.forward = enhanced_forward.__get__(model, model.__class__)
    
    return model
