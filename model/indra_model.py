"""
INDRA Model for Causal Language Modeling - HuggingFace Integration
(c) Divyansh Bharadwaj

A Vedic-aligned Decoder-only Transformer with:
- Grouped Query Attention with FlashAttention
- Mixture of Experts with load balancing
- Vedic philosophical integration
- Hierarchical reasoning capabilities
- KV caching for efficient inference
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Dict, Any
from dataclasses import dataclass

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput
from transformers.generation import GenerationMixin

# Import INDRA components
from transformer import INDRATransformer
from hierarchical_reasoning import add_hierarchical_reasoning_to_model


@dataclass
class IndraConfig(PretrainedConfig):
    model_type = "vedic-indra"

    def __init__(
        self,
        # Core Architecture
        vocab_size: int = 262179,
        n_positions: int = 2048,
        n_embd: int = 480,
        n_layer: int = 8,
        n_head: int = 6,
        n_kv_head: int = 2,
        intermediate_size: int = 1536,
        activation_function: str = "gelu",
        layer_norm_eps: float = 1e-5,
        use_rms_norm: bool = True,
        initializer_range: float = 0.02,
        
        # Attention & Performance
        use_flash_attention: bool = True,
        use_kv_cache: bool = True,
        gradient_checkpointing: bool = True,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        tie_word_embeddings: bool = False,
        use_cache: bool = True,

        # MoE Parameters (Now flattened)
        use_moe: bool = True,
        num_experts: int = 8,
        expert_capacity: int = 2,
        top_k: int = 2,
        jitter_noise: float = 0.1,
        auxiliary_loss_weight: float = 0.01,
        expert_hidden_size: int = 1536,

        # Vedic Parameters (Now flattened)
        use_vedic_core: bool = True,
        vedic_memory_size: int = 10000,
        vedic_retrieval_top_k: int = 5,
        vedic_expert_frozen: bool = True,
        vedic_expert_id: int = 0,
        vedic_routing_weight: float = 1.5,
        dharmic_alignment_weight: float = 0.1,

        # Hierarchical Reasoning Parameters (Now flattened)
        use_hierarchical_reasoning: bool = False,
        reasoning_levels: int = 5,
        reasoning_weight: float = 0.3,
        enable_in_training: bool = False,
        confidence_threshold: float = 0.7,
        
        **kwargs
    ):
        super().__init__(**kwargs)

        # Assign all parameters directly
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.intermediate_size = intermediate_size
        self.activation_function = activation_function
        self.layer_norm_eps = layer_norm_eps
        self.use_rms_norm = use_rms_norm
        self.initializer_range = initializer_range
        self.use_flash_attention = use_flash_attention
        self.use_kv_cache = use_kv_cache
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity 
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.expert_hidden_size = expert_hidden_size
        self.use_vedic_core = use_vedic_core
        self.vedic_memory_size = vedic_memory_size
        self.vedic_retrieval_top_k = vedic_retrieval_top_k
        self.vedic_expert_frozen = vedic_expert_frozen
        self.vedic_expert_id = vedic_expert_id
        self.vedic_routing_weight = vedic_routing_weight
        self.dharmic_alignment_weight = dharmic_alignment_weight
        self.gradient_checkpointing = gradient_checkpointing
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache
        
        # 1. MoE Sub-Config
        @dataclass
        class MoE:
            use_moe: bool = self.use_moe
            num_experts: int = self.num_experts
            expert_capacity: int = self.expert_capacity
            top_k: int = self.top_k
            jitter_noise: float = self.jitter_noise
            auxiliary_loss_weight: float = self.auxiliary_loss_weight
            expert_hidden_size: int = self.expert_hidden_size
        
        self.moe = MoE()
        
        # 2. Vedic Sub-Config (If the model also accesses vedic parameters via config.vedic)
        @dataclass
        class Vedic:
            use_vedic_core: bool = self.use_vedic_core
            vedic_memory_size: int = self.vedic_memory_size
            vedic_retrieval_top_k: int = self.vedic_retrieval_top_k
            vedic_expert_frozen: bool = self.vedic_expert_frozen
            vedic_expert_id: int = self.vedic_expert_id
            vedic_routing_weight: float = self.vedic_routing_weight
            dharmic_alignment_weight: float = self.dharmic_alignment_weight
        
        self.vedic = Vedic()

        # Hierarchical Reasoning
        self.use_hierarchical_reasoning = use_hierarchical_reasoning
        self.reasoning_levels = reasoning_levels
        self.reasoning_weight = reasoning_weight
        self.enable_in_training = enable_in_training
        self.confidence_threshold = confidence_threshold



class IndraPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading
    and loading pretrained models.
    """
    
    config_class = IndraConfig
    base_model_prefix = "indra"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]
    _skip_keys_device_placement = "past_key_values"
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, INDRATransformer):
            module.gradient_checkpointing = value


@dataclass
class IndraModelOutput(ModelOutput):
    """
    Output type for INDRA model with additional Vedic and reasoning metrics.
    
    Args:
        loss: Optional language modeling loss
        logits: Prediction scores of the language modeling head
        past_key_values: Pre-computed hidden-states for fast decoding
        hidden_states: Hidden-states of the model at each layer
        attentions: Attention weights (not used in current implementation)
        aux_losses: Auxiliary losses (MoE load balancing, Vedic alignment)
        vedic_metrics: Vedic principle scores and metrics
        reasoning_output: Optional hierarchical reasoning output
        confidence_scores: Optional confidence scores for reasoning
        reasoning_chain: Optional detailed reasoning chain
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    aux_losses: Optional[Dict[str, torch.FloatTensor]] = None
    vedic_metrics: Optional[Dict[str, Any]] = None
    reasoning_output: Optional[torch.FloatTensor] = None
    confidence_scores: Optional[Dict[str, torch.FloatTensor]] = None
    reasoning_chain: Optional[List[Dict]] = None


class IndraModel(IndraPreTrainedModel):
    """
    The bare INDRA Model transformer outputting raw hidden-states without any specific head.
    """
    
    def __init__(self, config: IndraConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize the core transformer
        self.transformer = INDRATransformer(config)
        
        # Add hierarchical reasoning if enabled
        if config.use_hierarchical_reasoning:
            self.transformer = add_hierarchical_reasoning_to_model(self.transformer)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.transformer.embed_tokens
    
    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        compute_vedic_rewards: bool = False,
        enable_reasoning: bool = False,
        reasoning_depth: int = 3,
        return_reasoning_chain: bool = False,
    ) -> Union[Tuple, IndraModelOutput]:
        """
        Forward pass through the INDRA model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            attention_mask: Mask to avoid attention on padding tokens
            position_ids: Position indices
            past_key_values: Pre-computed key-value pairs for fast decoding
            inputs_embeds: Embedded inputs (alternative to input_ids)
            use_cache: Whether to use key-value caching
            output_attentions: Whether to return attention weights (not implemented)
            output_hidden_states: Whether to return hidden states at all layers
            return_dict: Whether to return a ModelOutput object
            compute_vedic_rewards: Whether to compute Vedic alignment rewards
            enable_reasoning: Whether to enable hierarchical reasoning
            reasoning_depth: Depth of reasoning (1-5)
            return_reasoning_chain: Whether to return detailed reasoning chain
            
        Returns:
            IndraModelOutput or tuple with model outputs
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None 
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            compute_vedic_rewards=compute_vedic_rewards,
            enable_reasoning=enable_reasoning,
            reasoning_depth=reasoning_depth,
            return_reasoning_chain=return_reasoning_chain,
        )
        
        if not return_dict:
            return (outputs['logits'], outputs['past_key_values'], outputs['hidden_states'])
        
        return IndraModelOutput(
            logits=outputs['logits'],
            past_key_values=outputs['past_key_values'],
            hidden_states=outputs['hidden_states'],
            aux_losses=outputs.get('aux_losses'),
            vedic_metrics=outputs.get('vedic_metrics'),
            reasoning_output=outputs.get('reasoning_output'),
            confidence_scores=outputs.get('confidence_scores'),
            reasoning_chain=outputs.get('reasoning_chain'),
        )


class IndraForCausalLM(IndraPreTrainedModel, GenerationMixin):
    """
    INDRA Model with a language modeling head for causal language modeling.
    
    This is the main class for using INDRA with HuggingFace's generation utilities,
    training loops, and inference pipelines.
    """
    
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: IndraConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize the core transformer
        self.transformer = INDRATransformer(config)
        
        # The LM head is already part of INDRATransformer,
        # but we create a reference for HuggingFace compatibility
        self.lm_head = self.transformer.lm_head
        
        # Add hierarchical reasoning if enabled
        if config.use_hierarchical_reasoning:
            self.transformer = add_hierarchical_reasoning_to_model(self.transformer)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.transformer.embed_tokens
    
    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        self.transformer.lm_head = new_embeddings
    
    def set_decoder(self, decoder):
        self.transformer = decoder
    
    def get_decoder(self):
        return self.transformer
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        compute_vedic_rewards: bool = False,
        enable_reasoning: bool = False,
        reasoning_depth: int = 3,
        return_reasoning_chain: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for causal language modeling.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            attention_mask: Mask to avoid attention on padding tokens
            position_ids: Position indices
            past_key_values: Pre-computed key-value pairs for fast decoding
            inputs_embeds: Embedded inputs (alternative to input_ids)
            labels: Labels for computing language modeling loss
            use_cache: Whether to use key-value caching
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states at all layers
            return_dict: Whether to return a ModelOutput object
            compute_vedic_rewards: Whether to compute Vedic alignment rewards
            enable_reasoning: Whether to enable hierarchical reasoning
            reasoning_depth: Depth of reasoning (1-5)
            return_reasoning_chain: Whether to return detailed reasoning chain
            
        Returns:
            CausalLMOutputWithPast or tuple with model outputs including loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            compute_vedic_rewards=compute_vedic_rewards,
            enable_reasoning=enable_reasoning,
            reasoning_depth=reasoning_depth,
            return_reasoning_chain=return_reasoning_chain,
        )
        
        loss = outputs.get('loss')
        logits = outputs['logits']
        
        if not return_dict:
            output = (logits,) + (outputs['past_key_values'],)
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.get('past_key_values'),
            hidden_states=outputs.get('hidden_states'),
            attentions=None,  # Not implemented in current version
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare inputs for generation. This is called by HuggingFace's generate() method.
        """
        # Only use last token if we have past_key_values
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        # Create position_ids
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        
        # If inputs_embeds are passed, only use them in the first generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        
        return model_inputs
    
    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], 
        beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        Reorder past_key_values cache for beam search.
        This is required by HuggingFace's beam search implementation.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


# Register the model for auto classes
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register("vedic-indra", IndraConfig)
AutoModel.register(IndraConfig, IndraModel)
AutoModelForCausalLM.register(IndraConfig, IndraForCausalLM)
