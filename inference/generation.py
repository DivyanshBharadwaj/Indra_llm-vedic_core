"""
Text generation engine for INDRA LLM with advanced sampling and caching
(c) Divyansh Bharadwaj
"""

import time
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from model import INDRATransformer
from model.kv_cache import KVCacheManager, get_cache_manager

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = False
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    min_length: int = 0
    
class InferenceEngine:
    """High-performance inference engine for INDRA LLM."""
    
    def __init__(
        self,
        model: INDRATransformer,
        tokenizer,
        device: Optional[torch.device] = None,
        use_cache: bool = True,
        cache_size: int = 4096,
    ):
        """
        Initialize inference engine.
        
        Args:
            model: INDRA transformer model
            tokenizer: Tokenizer instance
            device: Device to run inference on
            use_cache: Whether to use KV caching
            cache_size: Maximum cache size
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cache = use_cache
        self.cache_size = cache_size
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup KV cache manager
        if self.use_cache:
            self.cache_manager = get_cache_manager(model.config)
        else:
            self.cache_manager = None
        
        # Generation statistics
        self.stats = {
            'total_tokens_generated': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
        
        logging.info(f"Inference engine initialized on {self.device}")
        if self.use_cache:
            logging.info(f"KV caching enabled with cache size: {cache_size}")
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        generation_config: Optional[GenerationConfig] = None,
        return_dict: bool = False,
        **kwargs
    ) -> Union[str, List[str], Dict[str, Any]]:
        """
        Generate text from prompt(s).
        
        Args:
            prompt: Input prompt(s)
            generation_config: Generation configuration
            return_dict: Whether to return detailed results
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text(s) or detailed results dictionary
        """
        # Handle default config
        if generation_config is None:
            generation_config = GenerationConfig(**kwargs)
        else:
            # Update with kwargs
            for key, value in kwargs.items():
                if hasattr(generation_config, key):
                    setattr(generation_config, key, value)
        
        # Handle single prompt vs batch
        is_single = isinstance(prompt, str)
        if is_single:
            prompts = [prompt]
        else:
            prompts = prompt
        
        # Generate for each prompt
        results = []
        generation_stats = []
        
        for prompt_text in prompts:
            result, stats = self._generate_single(prompt_text, generation_config)
            results.append(result)
            generation_stats.append(stats)
        
        # Return format
        if return_dict:
            return {
                'generated_texts': results if not is_single else results[0],
                'generation_stats': generation_stats if not is_single else generation_stats[0],
                'engine_stats': self.get_stats()
            }
        else:
            return results if not is_single else results[0]
    
    def _generate_single(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text for a single prompt."""
        start_time = time.time()
        
        # Tokenize input
        input_tokens = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=self.model.config.n_positions - config.max_new_tokens
        )
        input_ids = input_tokens['input_ids'].to(self.device)
        attention_mask = input_tokens.get('attention_mask')
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Setup generation parameters
        pad_token_id = config.pad_token_id or self.tokenizer.pad_token_id or 0
        eos_token_id = config.eos_token_id or self.tokenizer.eos_token_id or 1
        
        # Generate using appropriate strategy
        if config.num_beams > 1:
            generated_ids, generation_stats = self._beam_search_generate(
                input_ids, attention_mask, config
            )
        else:
            generated_ids, generation_stats = self._sampling_generate(
                input_ids, attention_mask, config
            )
        
        # Decode generated tokens
        generated_tokens = generated_ids[0, input_ids.size(1):]  # Remove input tokens
        generated_text = self.tokenizer.decode(
            generated_tokens, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Update statistics
        generation_time = time.time() - start_time
        self.stats['total_tokens_generated'] += len(generated_tokens)
        self.stats['total_time'] += generation_time
        
        generation_stats.update({
            'prompt_length': input_ids.size(1),
            'generated_length': len(generated_tokens),
            'generation_time': generation_time,
            'tokens_per_second': len(generated_tokens) / generation_time if generation_time > 0 else 0,
        })
        
        return generated_text, generation_stats
    
    def _sampling_generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
        config: GenerationConfig
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Generate using sampling strategies (greedy, top-k, top-p)."""
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Initialize generation
        generated_ids = input_ids.clone()
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Track generation statistics
        stats = {
            'total_tokens': 0,
            'cache_hits': 0,
            'repetitions_blocked': 0,
        }
        
        for step in range(config.max_new_tokens):
            # Prepare inputs for current step
            if past_key_values is not None and config.use_cache:
                # Use only the last token when we have cache
                current_input_ids = generated_ids[:, -1:]
                current_attention_mask = None  # Will be handled by cache
                stats['cache_hits'] += 1
            else:
                current_input_ids = generated_ids
                current_attention_mask = attention_mask
                if config.use_cache:
                    stats['cache_hits'] = stats.get('cache_hits', 0)  # No increment for cache miss
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=config.use_cache,
                )
            
            logits = outputs.logits[:, -1, :]  # Last token logits
            past_key_values = outputs.past_key_values if config.use_cache else None
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, generated_ids, config.repetition_penalty
                )
            
            # Apply no-repeat n-gram constraint
            if config.no_repeat_ngram_size > 0:
                logits = self._apply_no_repeat_ngrams(
                    logits, generated_ids, config.no_repeat_ngram_size
                )
            
            # Apply temperature
            if config.temperature != 1.0:
                logits = logits / config.temperature
            
            # Apply sampling constraints
            if config.do_sample:
                # Top-k filtering
                if config.top_k is not None and config.top_k > 0:
                    logits = self._apply_top_k(logits, config.top_k)
                
                # Top-p (nucleus) filtering
                if config.top_p is not None and config.top_p < 1.0:
                    logits = self._apply_top_p(logits, config.top_p)
                
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Update generated sequence
            generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
            stats['total_tokens'] += 1
            
            # Check for EOS tokens
            if config.eos_token_id is not None:
                finished = finished | (next_tokens.squeeze(-1) == config.eos_token_id)
                if finished.all():
                    break
            
            # Check minimum length
            if generated_ids.size(1) - input_ids.size(1) >= config.min_length:
                if config.early_stopping and finished.any():
                    break
        
        return generated_ids, stats
    
    def _beam_search_generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
        config: GenerationConfig
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Generate using beam search."""
        batch_size = input_ids.size(0)
        num_beams = config.num_beams
        device = input_ids.device
        
        # Expand inputs for beam search
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1).reshape(batch_size * num_beams, -1)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_beams, -1).reshape(batch_size * num_beams, -1)
        
        # Initialize beam search
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9  # Make sure only first beam is considered initially
        beam_scores = beam_scores.view(-1)
        
        generated_ids = input_ids.clone()
        past_key_values = None
        
        # Track statistics
        stats = {'total_tokens': 0, 'beam_expansions': 0}
        
        for step in range(config.max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated_ids if past_key_values is None else generated_ids[:, -1:],
                    attention_mask=attention_mask if past_key_values is None else None,
                    past_key_values=past_key_values,
                    use_cache=config.use_cache,
                )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values if config.use_cache else None
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, generated_ids, config.repetition_penalty
                )
            
            # Convert to log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Add beam scores
            next_scores = beam_scores.unsqueeze(-1) + log_probs
            
            # Reshape for beam search
            next_scores = next_scores.view(batch_size, num_beams * self.tokenizer.vocab_size)
            
            # Get top 2*num_beams candidates
            next_scores, next_tokens = torch.topk(
                next_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            
            # Process beams
            next_indices = torch.div(next_tokens, self.tokenizer.vocab_size, rounding_mode='floor')
            next_tokens = next_tokens % self.tokenizer.vocab_size
            
            # Prepare next generation
            beam_outputs = []
            beam_scores_new = []
            
            for batch_idx in range(batch_size):
                batch_beam_idx = 0
                for beam_token_rank, (next_score, next_token, next_index) in enumerate(
                    zip(next_scores[batch_idx], next_tokens[batch_idx], next_indices[batch_idx])
                ):
                    if batch_beam_idx >= num_beams:
                        break
                    
                    # Get the beam and token
                    beam_idx = batch_idx * num_beams + next_index
                    
                    # Add token to beam
                    beam_output = torch.cat([generated_ids[beam_idx], next_token.unsqueeze(0)], dim=-1)
                    beam_outputs.append(beam_output)
                    beam_scores_new.append(next_score)
                    
                    batch_beam_idx += 1
            
            # Update for next iteration
            generated_ids = torch.stack(beam_outputs)
            beam_scores = torch.stack(beam_scores_new)
            
            stats['total_tokens'] += num_beams
            stats['beam_expansions'] += 1
            
            # Early stopping check
            if config.early_stopping and config.eos_token_id is not None:
                # Check if any beam has EOS token
                last_tokens = generated_ids[:, -1]
                if (last_tokens == config.eos_token_id).any():
                    break
        
        # Select best beam (first beam has highest score)
        best_sequence = generated_ids[0:batch_size:num_beams]  # Take first beam of each batch
        
        return best_sequence, stats
    
    def _apply_repetition_penalty(
        self, 
        logits: Tensor, 
        generated_ids: Tensor, 
        penalty: float
    ) -> Tensor:
        """Apply repetition penalty to logits."""
        if penalty == 1.0:
            return logits
        
        batch_size = logits.size(0)
        
        for i in range(batch_size):
            for token_id in set(generated_ids[i].tolist()):
                # Apply penalty
                if logits[i, token_id] < 0:
                    logits[i, token_id] *= penalty
                else:
                    logits[i, token_id] /= penalty
        
        return logits
    
    def _apply_no_repeat_ngrams(
        self, 
        logits: Tensor, 
        generated_ids: Tensor, 
        no_repeat_ngram_size: int
    ) -> Tensor:
        """Prevent repetition of n-grams."""
        if no_repeat_ngram_size <= 0:
            return logits
        
        batch_size = logits.size(0)
        
        for i in range(batch_size):
            sequence = generated_ids[i].tolist()
            seq_len = len(sequence)
            
            if seq_len >= no_repeat_ngram_size:
                # Get current n-gram context
                current_ngram = sequence[-(no_repeat_ngram_size - 1):]
                
                # Find all previous occurrences of this n-gram
                for j in range(seq_len - no_repeat_ngram_size + 1):
                    if sequence[j:j + no_repeat_ngram_size - 1] == current_ngram:
                        # Block the next token that would complete this n-gram
                        blocked_token = sequence[j + no_repeat_ngram_size - 1]
                        logits[i, blocked_token] = float('-inf')
        
        return logits
    
    def _apply_top_k(self, logits: Tensor, top_k: int) -> Tensor:
        """Apply top-k filtering."""
        top_k = min(top_k, logits.size(-1))
        top_k_logits, _ = torch.topk(logits, top_k)
        min_values = top_k_logits[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
    
    def _apply_top_p(self, logits: Tensor, top_p: float) -> Tensor:
        """Apply nucleus (top-p) filtering."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits
    
    def batch_generate(
        self,
        prompts: List[str],
        generation_config: Optional[GenerationConfig] = None,
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts in batches."""
        if generation_config is None:
            generation_config = GenerationConfig(**kwargs)
        
        results = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = self.generate(
                batch_prompts, 
                generation_config=generation_config,
                return_dict=False
            )
            
            # Handle single result vs list
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
        
        return results
    
    def stream_generate(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ):
        """Generate text with streaming output."""
        if generation_config is None:
            generation_config = GenerationConfig(**kwargs)
        
        # Tokenize input
        input_tokens = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=self.model.config.n_positions - generation_config.max_new_tokens
        )
        input_ids = input_tokens['input_ids'].to(self.device)
        attention_mask = input_tokens.get('attention_mask')
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Initialize generation
        generated_ids = input_ids.clone()
        past_key_values = None
        
        # Stream generation
        for step in range(generation_config.max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated_ids if past_key_values is None else generated_ids[:, -1:],
                    attention_mask=attention_mask if past_key_values is None else None,
                    past_key_values=past_key_values,
                    use_cache=generation_config.use_cache,
                )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values if generation_config.use_cache else None
            
            # Apply generation constraints
            if generation_config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, generated_ids, generation_config.repetition_penalty
                )
            
            if generation_config.temperature != 1.0:
                logits = logits / generation_config.temperature
            
            # Sample next token
            if generation_config.do_sample:
                if generation_config.top_k is not None:
                    logits = self._apply_top_k(logits, generation_config.top_k)
                if generation_config.top_p is not None:
                    logits = self._apply_top_p(logits, generation_config.top_p)
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Decode and yield new token
            new_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield new_text
            
            # Update sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Check for EOS
            if (generation_config.eos_token_id is not None and 
                next_token.item() == generation_config.eos_token_id):
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference engine statistics."""
        total_time = max(self.stats['total_time'], 1e-6)  # Avoid division by zero
        
        return {
            'total_tokens_generated': self.stats['total_tokens_generated'],
            'total_time_seconds': self.stats['total_time'],
            'average_tokens_per_second': self.stats['total_tokens_generated'] / total_time,
            'cache_hit_rate': (
                self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1)
                if self.use_cache else 0.0
            ),
            'memory_usage': self._get_memory_usage(),
        }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        if torch.cuda.is_available():
            return {
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated(self.device) / 1024**2,
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved(self.device) / 1024**2,
                'gpu_memory_cached_mb': torch.cuda.memory_cached(self.device) / 1024**2,
            }
        else:
            return {'cpu_memory_mb': 0.0}  # Could implement CPU memory tracking
    
    def clear_cache(self):
        """Clear KV cache and reset statistics."""
        if self.cache_manager:
            self.cache_manager.cleanup_unused_caches()
        
        # Reset statistics
        self.stats = {
            'total_tokens_generated': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def set_model_dtype(self, dtype: torch.dtype):
        """Change model precision for inference."""
        self.model = self.model.to(dtype=dtype)
        logging.info(f"Model dtype changed to {dtype}")
    
    def enable_torch_compile(self):
        """Enable torch.compile for faster inference (PyTorch 2.0+)."""
        try:
            self.model = torch.compile(self.model)
            logging.info("Torch compile enabled")
        except Exception as e:
            logging.warning(f"Failed to enable torch.compile: {e}")
    
    def benchmark_generation(
        self,
        test_prompts: List[str],
        generation_config: Optional[GenerationConfig] = None,
        num_runs: int = 5
    ) -> Dict[str, float]:
        """Benchmark generation performance."""
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # Warmup
        self.generate(test_prompts[0], generation_config)
        
        # Benchmark
        total_time = 0.0
        total_tokens = 0
        
        for run in range(num_runs):
            start_time = time.time()
            
            for prompt in test_prompts:
                result = self.generate(prompt, generation_config, return_dict=True)
                total_tokens += result['generation_stats']['generated_length']
            
            run_time = time.time() - start_time
            total_time += run_time
        
        avg_time_per_run = total_time / num_runs
        avg_tokens_per_second = total_tokens / total_time
        
        return {
            'avg_time_per_run_seconds': avg_time_per_run,
            'avg_tokens_per_second': avg_tokens_per_second,
            'total_tokens_generated': total_tokens,
            'num_test_prompts': len(test_prompts),
            'num_runs': num_runs,
        }
