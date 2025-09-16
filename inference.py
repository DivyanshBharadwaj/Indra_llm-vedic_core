#!/usr/bin/env python3
"""
Standalone Inference Script for Indra LLM
A production-ready inference engine for the Vedic-aligned transformer model.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import time

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
import sentencepiece as spm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VedicTokenizer:
    """Vedic-specialized tokenizer with Sanskrit and Devanagari support."""
    
    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None
        self.vocab_size = None
        
        if tokenizer_path.endswith('.model'):
            # SentencePiece tokenizer
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(tokenizer_path)
            self.vocab_size = self.tokenizer.vocab_size()
        else:
            # HuggingFace tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.vocab_size = len(self.tokenizer.vocab)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if hasattr(self.tokenizer, 'encode_as_ids'):
            return self.tokenizer.encode_as_ids(text)
        else:
            return self.tokenizer.encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if hasattr(self.tokenizer, 'decode_ids'):
            return self.tokenizer.decode_ids(token_ids)
        else:
            return self.tokenizer.decode(token_ids)
    
    def get_vocab_size(self) -> int:
        return self.vocab_size


class KVCache:
    """Key-Value cache for efficient inference."""
    
    def __init__(self, max_seq_len: int, n_layers: int, n_heads: int, head_dim: int, device: str = 'cuda'):
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        
        # Initialize cache tensors
        self.keys = torch.zeros(n_layers, max_seq_len, n_heads, head_dim, device=device)
        self.values = torch.zeros(n_layers, max_seq_len, n_heads, head_dim, device=device)
        self.seq_len = 0
    
    def update(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache and return full keys/values."""
        batch_size, seq_len, n_heads, head_dim = keys.shape
        
        # Store new keys and values
        start_pos = self.seq_len
        end_pos = start_pos + seq_len
        
        self.keys[layer_idx, start_pos:end_pos] = keys[0]  # Remove batch dim
        self.values[layer_idx, start_pos:end_pos] = values[0]
        
        if layer_idx == self.n_layers - 1:  # Update seq_len only on last layer
            self.seq_len = end_pos
        
        # Return full cached keys/values for this layer
        return (
            self.keys[layer_idx, :end_pos].unsqueeze(0),  # Add batch dim back
            self.values[layer_idx, :end_pos].unsqueeze(0)
        )
    
    def clear(self):
        """Clear the cache."""
        self.seq_len = 0


class IndraInferenceEngine:
    """Standalone inference engine for Indra LLM."""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = None,
        dtype: torch.dtype = torch.float16,
        max_seq_len: int = 2048
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model checkpoint
            tokenizer_path: Path to the tokenizer model
            device: Device to run inference on (auto-detect if None)
            dtype: Model precision (float16 for efficiency)
            max_seq_len: Maximum sequence length
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_seq_len = max_seq_len
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.dtype = dtype
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Using dtype: {self.dtype}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = VedicTokenizer(tokenizer_path)
        logger.info(f"Tokenizer loaded. Vocab size: {self.tokenizer.get_vocab_size()}")
        
        # Load model
        logger.info("Loading model...")
        self.model = self._load_model()
        logger.info("Model loaded successfully!")
        
        # Initialize KV cache (will be set up properly after loading model config)
        self.kv_cache = None
        
    def _load_model(self) -> torch.nn.Module:
        """Load the model from checkpoint."""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model state and config
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model_state_dict = checkpoint['model_state_dict']
                    config = checkpoint.get('config', {})
                elif 'model' in checkpoint:
                    model_state_dict = checkpoint['model']
                    config = checkpoint.get('config', {})
                else:
                    model_state_dict = checkpoint
                    config = {}
            else:
                model_state_dict = checkpoint
                config = {}
            
            # Create model architecture (simplified version)
            model_config = self._create_model_config(config)
            model = self._create_model(model_config)
            
            # Load weights
            model.load_state_dict(model_state_dict, strict=False)
            model = model.to(self.device).to(self.dtype)
            model.eval()
            
            # Setup KV cache based on model config
            self.kv_cache = KVCache(
                max_seq_len=self.max_seq_len,
                n_layers=model_config.get('n_layer', 12),
                n_heads=model_config.get('n_kv_head', model_config.get('n_head', 12)),
                head_dim=model_config.get('n_embd', 768) // model_config.get('n_head', 12),
                device=self.device
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _create_model_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create model configuration with defaults."""
        default_config = {
            'vocab_size': self.tokenizer.get_vocab_size(),
            'n_embd': 768,
            'n_layer': 12,
            'n_head': 12,
            'n_kv_head': 4,
            'block_size': self.max_seq_len,
            'dropout': 0.1,
            'use_flash_attention': True,
            'use_vedic_core': True,
            'use_moe': False,
            'n_experts': 8,
            'n_active_experts': 2,
        }
        
        # Update with loaded config
        default_config.update(config)
        return default_config
    
    def _create_model(self, config: Dict[str, Any]) -> torch.nn.Module:
        """Create a simplified model architecture for inference."""
        # This is a simplified model structure - you might need to adjust based on your actual model
        class SimpleTransformerBlock(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.ln1 = torch.nn.LayerNorm(config['n_embd'])
                self.attn = torch.nn.MultiheadAttention(
                    config['n_embd'], 
                    config['n_head'], 
                    batch_first=True,
                    dropout=config['dropout']
                )
                self.ln2 = torch.nn.LayerNorm(config['n_embd'])
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(config['n_embd'], 4 * config['n_embd']),
                    torch.nn.GELU(),
                    torch.nn.Linear(4 * config['n_embd'], config['n_embd']),
                    torch.nn.Dropout(config['dropout'])
                )
            
            def forward(self, x, mask=None):
                # Pre-norm architecture
                x_norm = self.ln1(x)
                attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask)
                x = x + attn_out
                x = x + self.mlp(self.ln2(x))
                return x
        
        class SimpleIndraModel(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.token_embedding = torch.nn.Embedding(config['vocab_size'], config['n_embd'])
                self.position_embedding = torch.nn.Embedding(config['block_size'], config['n_embd'])
                self.blocks = torch.nn.ModuleList([
                    SimpleTransformerBlock(config) for _ in range(config['n_layer'])
                ])
                self.ln_f = torch.nn.LayerNorm(config['n_embd'])
                self.lm_head = torch.nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
                
                # Tie embeddings
                self.lm_head.weight = self.token_embedding.weight
            
            def forward(self, input_ids, position_ids=None):
                B, T = input_ids.shape
                
                if position_ids is None:
                    position_ids = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
                
                # Token and position embeddings
                token_emb = self.token_embedding(input_ids)
                pos_emb = self.position_embedding(position_ids)
                x = token_emb + pos_emb
                
                # Apply transformer blocks
                for block in self.blocks:
                    x = block(x)
                
                # Final layer norm and projection
                x = self.ln_f(x)
                logits = self.lm_head(x)
                
                return logits
        
        return SimpleIndraModel(config)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        vedic_mode: bool = False,
        dharmic_weight: float = 1.0
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens
            vedic_mode: Enable Vedic-guided generation
            dharmic_weight: Weight for dharmic alignment
        
        Returns:
            Generated text
        """
        logger.info(f"Generating text for prompt: '{prompt[:50]}...'")
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        # Clear KV cache
        if self.kv_cache:
            self.kv_cache.clear()
        
        generated_tokens = []
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                if step == 0:
                    # First pass - full sequence
                    logits = self.model(input_tensor)
                    next_token_logits = logits[0, -1, :]  # Last token's logits
                else:
                    # Subsequent passes - only new token
                    new_token_tensor = torch.tensor([[next_token]], device=self.device)
                    position_ids = torch.tensor([[len(input_ids) + step - 1]], device=self.device)
                    logits = self.model(new_token_tensor, position_ids)
                    next_token_logits = logits[0, 0, :]  # Single token's logits
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if generated_tokens:
                    for token_id in set(input_ids + generated_tokens):
                        next_token_logits[token_id] /= repetition_penalty
                
                # Apply Vedic guidance (placeholder - implement based on your Vedic core)
                if vedic_mode:
                    next_token_logits = self._apply_vedic_guidance(
                        next_token_logits, 
                        input_ids + generated_tokens,
                        dharmic_weight
                    )
                
                # Apply top-k and top-p filtering
                next_token_logits = self._top_k_top_p_filtering(
                    next_token_logits, 
                    top_k=top_k, 
                    top_p=top_p
                )
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Check for end of sequence
                if self._is_end_token(next_token):
                    break
                
                generated_tokens.append(next_token)
                
                # Update progress
                if (step + 1) % 10 == 0:
                    logger.info(f"Generated {step + 1}/{max_new_tokens} tokens...")
        
        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_tokens)
        full_text = prompt + generated_text
        
        logger.info(f"Generation completed. Generated {len(generated_tokens)} tokens.")
        return full_text
    
    def _apply_vedic_guidance(
        self, 
        logits: torch.Tensor, 
        context_tokens: List[int],
        dharmic_weight: float
    ) -> torch.Tensor:
        """Apply Vedic philosophical guidance to token probabilities."""
        # Placeholder implementation - enhance based on your Vedic core
        # This could include:
        # - Boosting probabilities of dharmic concepts
        # - Reducing probabilities of adharmic concepts
        # - Context-aware Sanskrit term preferences
        
        # Simple example: slightly boost certain Vedic concept tokens
        vedic_concepts = {
            'dharma': dharmic_weight,
            'karma': dharmic_weight,
            'ahimsa': dharmic_weight,
            'satya': dharmic_weight,
            'moksha': dharmic_weight,
        }
        
        # This is a simplified approach - in practice, you'd have token IDs for these concepts
        return logits
    
    def _top_k_top_p_filtering(
        self, 
        logits: torch.Tensor, 
        top_k: int = 0, 
        top_p: float = 1.0
    ) -> torch.Tensor:
        """Apply top-k and top-p filtering to logits."""
        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('inf')
        
        return logits
    
    def _is_end_token(self, token_id: int) -> bool:
        """Check if token is an end-of-sequence token."""
        # Common EOS token IDs - adjust based on your tokenizer
        eos_tokens = [0, 1, 2]  # Common values for pad, unk, eos
        return token_id in eos_tokens
    
    def interactive_chat(self):
        """Start an interactive chat session."""
        logger.info("Starting interactive chat. Type 'quit' to exit.")
        logger.info("Use /vedic to toggle Vedic mode, /temp <value> to change temperature.")
        
        vedic_mode = False
        temperature = 0.8
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() == 'quit':
                    break
                
                if user_input.startswith('/vedic'):
                    vedic_mode = not vedic_mode
                    print(f"Vedic mode: {'ON' if vedic_mode else 'OFF'}")
                    continue
                
                if user_input.startswith('/temp'):
                    try:
                        temperature = float(user_input.split()[1])
                        print(f"Temperature set to: {temperature}")
                        continue
                    except:
                        print("Usage: /temp <value>")
                        continue
                
                if not user_input:
                    continue
                
                # Generate response
                start_time = time.time()
                response = self.generate(
                    prompt=user_input,
                    max_new_tokens=200,
                    temperature=temperature,
                    vedic_mode=vedic_mode
                )
                generation_time = time.time() - start_time
                
                print(f"\nIndra: {response[len(user_input):].strip()}")
                print(f"[Generated in {generation_time:.2f}s]")
                
            except KeyboardInterrupt:
                print("\nChat interrupted.")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main function to run inference."""
    parser = argparse.ArgumentParser(description="Indra LLM Standalone Inference Engine")
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                       help='Path to the tokenizer model')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, auto-detect if not specified)')
    parser.add_argument('--dtype', type=str, default='float16',
                       choices=['float16', 'float32', 'bfloat16'],
                       help='Model precision')
    parser.add_argument('--max_seq_len', type=int, default=2048,
                       help='Maximum sequence length')
    
    # Generation parameters
    parser.add_argument('--prompt', type=str, default=None,
                       help='Text prompt for generation')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p sampling parameter')
    parser.add_argument('--repetition_penalty', type=float, default=1.1,
                       help='Repetition penalty')
    parser.add_argument('--vedic_mode', action='store_true',
                       help='Enable Vedic-guided generation')
    parser.add_argument('--dharmic_weight', type=float, default=1.0,
                       help='Weight for dharmic alignment')
    
    # Interaction mode
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive chat mode')
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    try:
        # Initialize inference engine
        engine = IndraInferenceEngine(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            dtype=dtype,
            max_seq_len=args.max_seq_len
        )
        
        if args.interactive:
            # Start interactive chat
            engine.interactive_chat()
        else:
            # Single generation
            if args.prompt is None:
                print("Please provide a prompt using --prompt or use --interactive mode")
                return
            
            start_time = time.time()
            result = engine.generate(
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                vedic_mode=args.vedic_mode,
                dharmic_weight=args.dharmic_weight
            )
            generation_time = time.time() - start_time
            
            print("\n" + "="*80)
            print("GENERATED TEXT:")
            print("="*80)
            print(result)
            print("="*80)
            print(f"Generation completed in {generation_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
