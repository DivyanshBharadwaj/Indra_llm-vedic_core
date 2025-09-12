"""
SentencePiece tokenizer for INDRA LLM with multi-language support
(c) Divyansh Bharadwaj
"""

import os
import json
import logging
from typing import List, Dict, Optional, Union, Tuple
import sentencepiece as spm

class SentencePieceTokenizer:
    """SentencePiece Unigram tokenizer with multi-language support."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_size: int = 50000,
        special_tokens: Optional[Dict[str, str]] = None,
        add_bos_token: bool = True,
        add_eos_token: bool = True,
    ):
        """
        Initialize SentencePiece tokenizer.
        
        Args:
            model_path: Path to trained SentencePiece model
            vocab_size: Vocabulary size for training
            special_tokens: Dictionary of special tokens
            add_bos_token: Whether to add BOS token
            add_eos_token: Whether to add EOS token
        """
        self.vocab_size = vocab_size
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        
        # Default special tokens
        default_special_tokens = {
            "sep_token": "<sep>",
            "cls_token": "<cls>",
            "mask_token": "<mask>",
            # Vedic special tokens
            "vedic_start": "<vedic>",
            "vedic_end": "</vedic>",
            "sanskrit_start": "<sanskrit>",
            "sanskrit_end": "</sanskrit>",
            "dharma_token": "<dharma>",
            "karma_token": "<karma>",
            "ahimsa_token": "<ahimsa>",
            "satya_token": "<satya>",
        }
        
        self.special_tokens = special_tokens or default_special_tokens
        self.sp_model = None
        self.token_to_id = {}
        self.id_to_token = {}
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def train_tokenizer(
        self,
        input_files: List[str],
        model_prefix: str,
        character_coverage: float = 0.9995,
        model_type: str = "unigram",
        max_sentence_length: int = 10000000, # <= Orignal 16384
        shuffle_input_sentence: bool = True,
    ) -> None:
        """
        Train SentencePiece tokenizer on input files.
        
        Args:
            input_files: List of input text files
            model_prefix: Prefix for output model files
            character_coverage: Character coverage for vocabulary
            model_type: Model type (unigram, bpe, word, char)
            max_sentence_length: Maximum sentence length
            shuffle_input_sentence: Whether to shuffle input sentences
        """
        # Prepare special tokens string
        user_defined_symbols = list(self.special_tokens.values())
        
        # Training arguments
        train_args = (
            f"--input={','.join(input_files)} "
            f"--model_prefix={model_prefix} "
            f"--vocab_size={self.vocab_size} "
            f"--character_coverage={character_coverage} "
            f"--model_type={model_type} "
            f"--max_sentence_length={max_sentence_length} "
            f"--shuffle_input_sentence={shuffle_input_sentence} "
            f"--user_defined_symbols={','.join(user_defined_symbols)} "
            f"--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
            f"--normalization_rule_name=identity "
            f"--remove_extra_whitespaces=false "
            f"--input_sentence_size=10000000 "
            f"--seed_sentencepiece_size=1000000"
        )
        
        logging.info(f"Training SentencePiece model with args: {train_args}")
        spm.SentencePieceTrainer.train(train_args)
        
        # Load the trained model
        model_path = f"{model_prefix}.model"
        self.load_model(model_path)
        
        # Save tokenizer config
        self.save_config(f"{model_prefix}_config.json")
    
    def load_model(self, model_path: str) -> None:
        """Load trained SentencePiece model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
        
        # Build token mappings
        self._build_token_mappings()
        
        logging.info(f"Loaded SentencePiece model from {model_path}")
        logging.info(f"Vocabulary size: {self.sp_model.vocab_size()}")
    
    def _build_token_mappings(self) -> None:
        """Build token to ID and ID to token mappings."""
        self.token_to_id = {}
        self.id_to_token = {}
        
        for i in range(self.sp_model.vocab_size()):
            token = self.sp_model.id_to_piece(i)
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        # Verify special tokens
        for token_name, token in self.special_tokens.items():
            if token not in self.token_to_id:
                logging.warning(f"Special token {token_name}='{token}' not found in vocabulary")
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_attention_mask: bool = False,
    ) -> Union[List[int], Dict[str, List[int]]]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length
            return_attention_mask: Whether to return attention mask
            
        Returns:
            Token IDs or dict with input_ids and attention_mask
        """
        if self.sp_model is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        
        # Encode text
        token_ids = self.sp_model.encode(text, out_type=int)
        
        # Add special tokens
        if add_special_tokens:
            if self.add_bos_token:
                bos_id = self.token_to_id.get(self.special_tokens["bos_token"], 2)
                token_ids = [bos_id] + token_ids
            if self.add_eos_token:
                eos_id = self.token_to_id.get(self.special_tokens["eos_token"], 3)
                token_ids = token_ids + [eos_id]
        
        # Handle max_length
        original_length = len(token_ids)
        
        if max_length is not None:
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            elif padding and len(token_ids) < max_length:
                pad_id = self.token_to_id.get(self.special_tokens["pad_token"], 0)
                token_ids = token_ids + [pad_id] * (max_length - len(token_ids))
        
        if return_attention_mask:
            attention_mask = [1] * min(original_length, len(token_ids))
            if len(attention_mask) < len(token_ids):
                attention_mask.extend([0] * (len(token_ids) - len(attention_mask)))
            
            return {
                "input_ids": token_ids,
                "attention_mask": attention_mask
            }
        
        return token_ids
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            Decoded text
        """
        if self.sp_model is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        
        # Filter out special tokens if requested
        if skip_special_tokens:
            special_token_ids = [
                self.token_to_id.get(token, -1) for token in self.special_tokens.values()
            ]
            token_ids = [tid for tid in token_ids if tid not in special_token_ids]
        
        # Decode
        text = self.sp_model.decode(token_ids)
        
        if clean_up_tokenization_spaces:
            text = text.strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into subword tokens."""
        if self.sp_model is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        
        return self.sp_model.encode(text, out_type=str)
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs."""
        return [self.token_to_id.get(token, self.token_to_id.get(self.special_tokens["unk_token"], 1)) 
                for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens."""
        return [self.id_to_token.get(id, self.special_tokens["unk_token"]) for id in ids]
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary."""
        return self.token_to_id.copy()
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.token_to_id)
    
    def save_config(self, config_path: str) -> None:
        """Save tokenizer configuration."""
        config = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "add_bos_token": self.add_bos_token,
            "add_eos_token": self.add_eos_token,
            "tokenizer_type": "SentencePiece",
        }
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved tokenizer config to {config_path}")
    
    def load_config(self, config_path: str) -> None:
        """Load tokenizer configuration."""
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        self.vocab_size = config.get("vocab_size", self.vocab_size)
        self.special_tokens = config.get("special_tokens", self.special_tokens)
        self.add_bos_token = config.get("add_bos_token", self.add_bos_token)
        self.add_eos_token = config.get("add_eos_token", self.add_eos_token)
        
        logging.info(f"Loaded tokenizer config from {config_path}")
    
    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_attention_mask: bool = True,
    ) -> Dict[str, List[List[int]]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input texts
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            return_attention_mask: Whether to return attention masks
            
        Returns:
            Dictionary with input_ids and optionally attention_mask
        """
        batch_input_ids = []
        batch_attention_mask = []
        
        # Determine max_length if not provided and padding is True
        if padding and max_length is None:
            all_lengths = []
            for text in texts:
                token_ids = self.encode(text, add_special_tokens=add_special_tokens)
                all_lengths.append(len(token_ids))
            max_length = max(all_lengths)
        
        # Encode each text
        for text in texts:
            encoded = self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_attention_mask=return_attention_mask,
            )
            
            if return_attention_mask:
                batch_input_ids.append(encoded["input_ids"])
                batch_attention_mask.append(encoded["attention_mask"])
            else:
                batch_input_ids.append(encoded)
        
        result = {"input_ids": batch_input_ids}
        if return_attention_mask:
            result["attention_mask"] = batch_attention_mask
        
        return result
    
    def decode_batch(
        self,
        batch_token_ids: List[List[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> List[str]:
        """
        Decode a batch of token ID sequences.
        
        Args:
            batch_token_ids: List of token ID sequences
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            List of decoded texts
        """
        return [
            self.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for token_ids in batch_token_ids
        ]
    
    @property
    def pad_token(self) -> str:
        return self.special_tokens["pad_token"]
    
    @property
    def pad_token_id(self) -> int:
        return self.token_to_id.get(self.pad_token, 0)
    
    @property
    def unk_token(self) -> str:
        return self.special_tokens["unk_token"]
    
    @property
    def unk_token_id(self) -> int:
        return self.token_to_id.get(self.unk_token, 1)
    
    @property
    def bos_token(self) -> str:
        return self.special_tokens["bos_token"]
    
    @property
    def bos_token_id(self) -> int:
        return self.token_to_id.get(self.bos_token, 2)
    
    @property
    def eos_token(self) -> str:
        return self.special_tokens["eos_token"]
    
    @property
    def eos_token_id(self) -> int:
        return self.token_to_id.get(self.eos_token, 3)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.get_vocab_size()
    
    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_attention_mask: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], Dict[str, List[int]], Dict[str, List[List[int]]]]:
        """
        Main tokenization method supporting both single and batch inputs.
        
        Args:
            text: Single text or list of texts
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            return_attention_mask: Whether to return attention mask
            return_tensors: Format of return tensors (None, 'pt')
            
        Returns:
            Encoded output in requested format
        """
        if isinstance(text, str):
            # Single text
            result = self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_attention_mask=return_attention_mask,
            )
        else:
            # Batch of texts
            result = self.encode_batch(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_attention_mask=return_attention_mask,
            )
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch
            if isinstance(result, dict):
                result = {k: torch.tensor(v) for k, v in result.items()}
            else:
                result = torch.tensor(result)
        
        return result
