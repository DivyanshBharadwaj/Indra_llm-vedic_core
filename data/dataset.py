"""
Streaming Dataset classes for INDRA LLM with support for multiple formats and languages
Memory-efficient streaming implementation with configurable batch sizes
(c) Divyansh Bharadwaj - Modified for streaming support
"""

import os
import glob
import json
import logging
import random
import pickle
import hashlib
from typing import List, Dict, Optional, Union, Iterator, Tuple, Generator
from pathlib import Path
import gc

import torch
from torch.utils.data import Dataset, IterableDataset
import pandas as pd
from datasets import load_dataset, Dataset as HFDataset

class StreamingINDRADataset(IterableDataset):
    """
    Memory-efficient streaming dataset class supporting multiple data formats:
    - .txt files (raw text)
    - .json files (structured data)
    - .csv files (tabular data)
    - HuggingFace datasets
    
    Processes data in configurable batch chunks (1-10MB) to minimize memory usage.
    """
    
    def __len__(self):
        """Return estimated dataset length based on files and batch size."""
        # Rough estimate: assume average file size and examples per file
        estimated_examples_per_file = max(1, self.examples_per_batch // 10)
        return self.total_files * estimated_examples_per_file
    
    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer,
        max_length: int = 2048,
        data_type: str = "auto",
        text_column: str = "text",
        instruction_column: Optional[str] = None,
        response_column: Optional[str] = None,
        split_documents: bool = True,
        chunk_size: int = 1024,
        overlap_size: int = 128,
        min_length: int = 10,
        filter_languages: Optional[List[str]] = None,
        batch_size_mb: float = 5.0,  # Memory batch size in MB
        cache_dir: Optional[str] = None,
        shuffle_buffer_size: int = 1000,
        prefetch_factor: int = 2,
        random_seed: int = 42,  # Add random seed for reproducibility
    ):
        """
        Initialize streaming INDRA dataset.
        
        Args:
            data_path: Path to data file(s) or HuggingFace dataset name
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            data_type: Type of data ('txt', 'json', 'csv', 'hf', 'auto')
            text_column: Column name for text data
            instruction_column: Column name for instructions (SFT)
            response_column: Column name for responses (SFT)
            split_documents: Whether to split long documents
            chunk_size: Size of text chunks when splitting
            overlap_size: Overlap between chunks
            min_length: Minimum text length to include
            filter_languages: List of languages to include
            batch_size_mb: Target batch size in MB for memory management
            cache_dir: Directory for caching processed chunks
            shuffle_buffer_size: Size of shuffle buffer
            prefetch_factor: Number of batches to prefetch
        """
        self.data_paths = [data_path] if isinstance(data_path, str) else data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_type = data_type
        self.text_column = text_column
        self.instruction_column = instruction_column
        self.response_column = response_column
        self.split_documents = split_documents
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_length = min_length
        self.filter_languages = filter_languages
        self.batch_size_mb = batch_size_mb
        self.cache_dir = cache_dir
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_factor = prefetch_factor
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
        # Calculate approximate batch size in number of examples
        # Rough estimate: 1 token ≈ 4 bytes, average text ≈ max_length/2 tokens
        avg_tokens_per_example = max_length // 2
        bytes_per_example = avg_tokens_per_example * 4 * 2  # 2x for safety margin
        self.examples_per_batch = max(1, int((batch_size_mb * 1024 * 1024) // bytes_per_example))
        
        # Setup cache directory
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # File tracking
        self.file_list = self._build_file_list()
        self.total_files = len(self.file_list)
        
        # Shuffle buffer
        self._shuffle_buffer = []
        
        logging.info(f"Initialized streaming dataset with {self.total_files} files")
        logging.info(f"Target batch size: {self.examples_per_batch} examples (~{batch_size_mb}MB)")
    
    def _build_file_list(self) -> List[Tuple[str, str]]:
        """Build list of all files to process with their detected types."""
        all_files = []
        
        for path in self.data_paths:
            if os.path.isdir(path):
                # Find all supported files recursively
                for ext in ['*.txt', '*.json', '*.jsonl', '*.csv']:
                    all_files.extend([
                        (f, self._detect_data_type(f))
                        for f in glob.glob(os.path.join(path, '**', ext), recursive=True)
                    ])
            elif os.path.isfile(path):
                all_files.append((path, self._detect_data_type(path)))
            else:
                # Assume HuggingFace dataset
                all_files.append((path, 'hf'))
        
        return all_files
    
    def _detect_data_type(self, path: str) -> str:
        """Auto-detect data type from file extension or content."""
        if not os.path.isfile(path):
            return 'hf'
            
        ext = Path(path).suffix.lower()
        if ext in ['.txt']:
            return 'txt'
        elif ext in ['.json', '.jsonl']:
            return 'json'
        elif ext in ['.csv']:
            return 'csv'
        else:
            # Try to detect from content
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('{'):
                        return 'json'
                    elif ',' in first_line and '"' in first_line:
                        return 'csv'
                    else:
                        return 'txt'
            except Exception:
                return 'txt'
    
    def _get_cache_path(self, file_path: str) -> Optional[str]:
        """Get cache file path for a given input file."""
        if not self.cache_dir:
            return None
        
        # Create hash of file path + modification time for cache key
        stat = os.stat(file_path) if os.path.exists(file_path) else None
        cache_key = hashlib.md5(
            f"{file_path}_{stat.st_mtime if stat else 0}_{self.max_length}_{self.chunk_size}".encode()
        ).hexdigest()
        
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _load_from_cache(self, cache_path: str) -> Optional[List[Dict]]:
        """Load processed examples from cache."""
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.debug(f"Cache miss for {cache_path}: {e}")
            return None
    
    def _save_to_cache(self, cache_path: str, examples: List[Dict]):
        """Save processed examples to cache."""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(examples, f)
        except Exception as e:
            logging.warning(f"Failed to save cache {cache_path}: {e}")
    
    def _stream_file_batches(self, file_path: str, file_type: str) -> Generator[List[Dict], None, None]:
        """Stream batches of examples from a single file."""
        cache_path = self._get_cache_path(file_path)
        
        # Determine if this is Vedic content based on file path
        is_vedic_file = self._is_vedic_file(file_path)
        
        # Try loading from cache first
        if cache_path and os.path.exists(cache_path):
            cached_examples = self._load_from_cache(cache_path)
            if cached_examples:
                # Add Vedic metadata to cached examples if needed
                if is_vedic_file:
                    for example in cached_examples:
                        if 'is_vedic' not in example:
                            example['is_vedic'] = True
                            example['vedic_source'] = file_path
                
                # Yield cached examples in batches
                for i in range(0, len(cached_examples), self.examples_per_batch):
                    batch = cached_examples[i:i + self.examples_per_batch]
                    yield batch
                return
        
        # Process file and collect examples for caching
        all_examples = []
        current_batch = []
        
        try:
            if file_type == 'txt':
                for example in self._stream_txt_file(file_path):
                    # Add Vedic metadata based on file path
                    if is_vedic_file:
                        example['is_vedic'] = True
                        example['vedic_source'] = file_path
                    else:
                        example['is_vedic'] = False
                    
                    current_batch.append(example)
                    all_examples.append(example)
                    
                    if len(current_batch) >= self.examples_per_batch:
                        yield current_batch
                        current_batch = []
                        
            elif file_type == 'json':
                for example in self._stream_json_file(file_path):
                    # Add Vedic metadata based on file path
                    if is_vedic_file:
                        example['is_vedic'] = True
                        example['vedic_source'] = file_path
                    else:
                        example['is_vedic'] = False
                    
                    current_batch.append(example)
                    all_examples.append(example)
                    
                    if len(current_batch) >= self.examples_per_batch:
                        yield current_batch
                        current_batch = []
                        
            elif file_type == 'csv':
                for example in self._stream_csv_file(file_path):
                    # Add Vedic metadata based on file path
                    if is_vedic_file:
                        example['is_vedic'] = True
                        example['vedic_source'] = file_path
                    else:
                        example['is_vedic'] = False
                    
                    current_batch.append(example)
                    all_examples.append(example)
                    
                    if len(current_batch) >= self.examples_per_batch:
                        yield current_batch
                        current_batch = []
                        
            elif file_type == 'hf':
                for example in self._stream_hf_dataset(file_path):
                    # Add Vedic metadata based on file path
                    if is_vedic_file:
                        example['is_vedic'] = True
                        example['vedic_source'] = file_path
                    else:
                        example['is_vedic'] = False
                    
                    current_batch.append(example)
                    all_examples.append(example)
                    
                    if len(current_batch) >= self.examples_per_batch:
                        yield current_batch
                        current_batch = []
                        
            # Yield remaining examples
            if current_batch:
                yield current_batch
                
            # Cache all examples
            if cache_path and all_examples:
                self._save_to_cache(cache_path, all_examples)
                
        except Exception as e:
            logging.error(f"Error streaming file {file_path}: {e}")
        
        # Clean up memory
        del all_examples
        gc.collect()
    
    def _is_vedic_file(self, file_path: str) -> bool:
        """Determine if a file should be considered Vedic based on its path."""
        # Convert to lowercase for case-insensitive matching
        normalized_path = file_path.lower().replace('\\', '/')
        
        # Check if the file is in vedic_texts directory or similar
        vedic_indicators = [
            'vedic_texts',
            'vedic',
            'sanskrit',
            'vedas',
            'upanishads',
            'puranas',
            'mahabharata',
            'ramayana',
            'bhagavad_gita',
            'gita'
        ]
        
        return any(indicator in normalized_path for indicator in vedic_indicators)
    
    def _stream_txt_file(self, path: str) -> Generator[Dict, None, None]:
        """Stream examples from a text file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if self.split_documents and len(content) > self.chunk_size:
                chunks = self._split_text(content)
                for chunk in chunks:
                    if len(chunk.strip()) >= self.min_length:
                        yield {'text': chunk.strip()}
            else:
                if len(content.strip()) >= self.min_length:
                    yield {'text': content.strip()}
                    
        except Exception as e:
            logging.error(f"Error reading text file {path}: {e}")
    
    def _stream_json_file(self, path: str) -> Generator[Dict, None, None]:
        """Stream examples from a JSON/JSONL file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                # Try single JSON first
                try:
                    f.seek(0)
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            processed = self._process_json_item(item)
                            if processed:
                                yield processed
                    elif isinstance(data, dict):
                        processed = self._process_json_item(data)
                        if processed:
                            yield processed
                            
                except json.JSONDecodeError:
                    # Try JSONL
                    f.seek(0)
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if line:
                            try:
                                item = json.loads(line)
                                processed = self._process_json_item(item)
                                if processed:
                                    yield processed
                            except json.JSONDecodeError:
                                logging.warning(f"Skipping invalid JSON line {line_num} in {path}")
                                
        except Exception as e:
            logging.error(f"Error reading JSON file {path}: {e}")
    
    def _stream_csv_file(self, path: str) -> Generator[Dict, None, None]:
        """Stream examples from a CSV file in chunks."""
        try:
            # Read CSV in chunks to manage memory
            chunk_size = max(100, self.examples_per_batch)
            for chunk_df in pd.read_csv(path, chunksize=chunk_size):
                for _, row in chunk_df.iterrows():
                    processed = self._process_json_item(row.to_dict())
                    if processed:
                        yield processed
                        
        except Exception as e:
            logging.error(f"Error reading CSV file {path}: {e}")
    
    def _stream_hf_dataset(self, dataset_name: str) -> Generator[Dict, None, None]:
        """Stream examples from a HuggingFace dataset."""
        try:
            # Try loading with different splits
            dataset = None
            for split in ['train', 'validation', 'test']:
                try:
                    dataset = load_dataset(dataset_name, split=split, streaming=True)
                    break
                except Exception:
                    continue
            
            if dataset is None:
                # Try loading without split
                dataset = load_dataset(dataset_name, streaming=True)
                
            # Stream examples
            if hasattr(dataset, '__iter__'):
                for item in dataset:
                    processed = self._process_json_item(item)
                    if processed:
                        yield processed
                        
        except Exception as e:
            logging.error(f"Error streaming HuggingFace dataset {dataset_name}: {e}")
    
    def _process_json_item(self, item: Dict) -> Optional[Dict]:
        """Process a single JSON item and return processed example or None."""
        if not isinstance(item, dict):
            return None
        
        # Handle pre-tokenized format: {"filename": "", "original_text": "", "cleaned_text": "", "tokens": [], "token_count": }
        if 'tokens' in item and 'cleaned_text' in item:
            tokens = item.get('tokens', [])
            cleaned_text = item.get('cleaned_text', '').strip()
            token_count = item.get('token_count', len(tokens))
            
            if tokens and len(tokens) >= self.min_length and cleaned_text:
                processed_item = {
                    'text': cleaned_text,
                    'tokens': tokens,
                    'token_count': token_count,
                    'filename': item.get('filename', ''),
                    'original_text': item.get('original_text', ''),
                    'is_pretokenized': True
                }
                return processed_item
            
        # Handle instruction-response format
        elif self.instruction_column and self.response_column:
            if self.instruction_column in item and self.response_column in item:
                instruction = str(item[self.instruction_column]).strip()
                response = str(item[self.response_column]).strip()
                if instruction and response and len(instruction + response) >= self.min_length:
                    return {
                        'instruction': instruction,
                        'response': response,
                        'text': f"{instruction}\n{response}",
                        'is_pretokenized': False
                    }
                    
        # Handle simple text format
        elif self.text_column in item:
            text = str(item[self.text_column]).strip()
            if len(text) >= self.min_length:
                processed_item = {'text': text, 'is_pretokenized': False}
                # Copy other fields, but skip problematic ones
                for key, value in item.items():
                    if key != self.text_column and key not in ['veda', 'vedic']:  # Skip problematic keys
                        try:
                            # Only copy JSON-serializable values
                            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                processed_item[key] = value
                        except:
                            # Skip any problematic values
                            continue
                return processed_item
                
        # Try common text fields
        else:
            for field in ['text', 'content', 'body', 'message']:
                if field in item:
                    text = str(item[field]).strip()
                    if len(text) >= self.min_length:
                        return {'text': text, 'is_pretokenized': False}
                        
        return None
    
    def _split_text(self, text: str) -> List[str]:
        """Split long text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Find good breaking point
            chunk = text[start:end]
            for delimiter in ['\n\n', '\n', '. ', '! ', '? ']:
                last_pos = chunk.rfind(delimiter)
                if last_pos > self.chunk_size // 2:
                    chunk = text[start:start + last_pos + len(delimiter)]
                    break
            
            chunks.append(chunk)
            start = start + len(chunk) - self.overlap_size
        
        return chunks
    
    def _shuffle_buffer_add(self, examples: List[Dict]):
        """Add examples to shuffle buffer."""
        self._shuffle_buffer.extend(examples)
        
        # If buffer is full, shuffle and yield some examples
        if len(self._shuffle_buffer) >= self.shuffle_buffer_size:
            random.shuffle(self._shuffle_buffer)
            # Yield half the buffer
            yield_count = len(self._shuffle_buffer) // 2
            for i in range(yield_count):
                yield self._shuffle_buffer.pop(0)
    
    def _flush_shuffle_buffer(self):
        """Flush remaining examples from shuffle buffer."""
        random.shuffle(self._shuffle_buffer)
        while self._shuffle_buffer:
            yield self._shuffle_buffer.pop(0)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over dataset examples."""
        # Shuffle files for each epoch
        files_to_process = self.file_list.copy()
        random.shuffle(files_to_process)
        
        try:
            for file_path, file_type in files_to_process:
                # Stream batches from this file
                for batch in self._stream_file_batches(file_path, file_type):
                    # Add to shuffle buffer and yield when ready
                    for example in self._shuffle_buffer_add(batch):
                        # Tokenize and yield
                        tokenized = self._tokenize_example(example)
                        if tokenized:
                            yield tokenized
                    
                    # Clean up memory after each batch
                    gc.collect()
            
            # Flush remaining examples
            for example in self._flush_shuffle_buffer():
                tokenized = self._tokenize_example(example)
                if tokenized:
                    yield tokenized
                    
        except Exception as e:
            logging.error(f"Error in dataset iteration: {e}")
    
    def _tokenize_example(self, example: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """Tokenize example, using pre-tokenized data when available."""
        try:
            # Handle pre-tokenized data
            if example.get('is_pretokenized', False) and 'tokens' in example:
                tokens = example['tokens']
                if not isinstance(tokens, list):
                    return None
                
                # Convert tokens to tensor and handle length
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                elif len(tokens) < self.max_length:
                    # Pad with tokenizer's pad_token_id
                    pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
                    tokens = tokens + [pad_token_id] * (self.max_length - len(tokens))
                
                input_ids = torch.tensor(tokens, dtype=torch.long)
                attention_mask = (input_ids != getattr(self.tokenizer, 'pad_token_id', 0)).long()
                
                result = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': input_ids.clone(),
                }
                
                # Add metadata (but skip problematic keys)
                for key, value in example.items():
                    if key not in ['text', 'tokens', 'is_pretokenized'] and not key.startswith('_'):
                        try:
                            if isinstance(value, (str, int, float, bool)):
                                result[key] = value
                            elif isinstance(value, list) and all(isinstance(x, (str, int, float, bool)) for x in value):
                                result[key] = value
                        except:
                            continue
                
                return result
            
            # Handle regular text that needs tokenization
            text = example.get('text', '')
            if not text:
                return None
            
            # Standard tokenization
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            result = {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': encoding['input_ids'].squeeze(0).clone(),
            }
            
            # Add metadata (but skip problematic keys)
            for key, value in example.items():
                if key not in ['text', 'is_pretokenized'] and not key.startswith('_'):
                    try:
                        if isinstance(value, (str, int, float, bool)):
                            result[key] = value
                        elif isinstance(value, list) and all(isinstance(x, (str, int, float, bool)) for x in value):
                            result[key] = value
                    except:
                        continue
            
            return result
            
        except Exception as e:
            logging.error(f"Error tokenizing example: {e}")
            return None


class StreamingVedicDataset(StreamingINDRADataset):
    """Streaming version of VedicDataset for memory-efficient Vedic text processing."""
    
    def __len__(self):
        """Return estimated dataset length based on files and batch size."""
        # Vedic texts might have different density, but use same estimation
        estimated_examples_per_file = max(1, self.examples_per_batch // 8)  # Slightly higher for Vedic
        return self.total_files * estimated_examples_per_file
    
    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer,
        vedic_weight: float = 2.0,
        preserve_structure: bool = True,
        add_vedic_markers: bool = True,
        **kwargs
    ):
        self.vedic_weight = vedic_weight
        self.preserve_structure = preserve_structure
        self.add_vedic_markers = add_vedic_markers
        
        super().__init__(data_path, tokenizer, **kwargs)
    
    def _tokenize_example(self, example: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """Tokenize with Vedic-specific processing, supporting pre-tokenized data."""
        try:
            # Check if this is Vedic content
            is_vedic = example.get('is_vedic', False)
            
            # Handle pre-tokenized data
            if example.get('is_pretokenized', False) and 'tokens' in example:
                tokens = example['tokens']
                if not isinstance(tokens, list):
                    return None
                
                # Convert tokens to tensor and handle length
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                elif len(tokens) < self.max_length:
                    # Pad with tokenizer's pad_token_id
                    pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
                    tokens = tokens + [pad_token_id] * (self.max_length - len(tokens))
                
                input_ids = torch.tensor(tokens, dtype=torch.long)
                attention_mask = (input_ids != getattr(self.tokenizer, 'pad_token_id', 0)).long()
                
                # Apply Vedic weighting
                if is_vedic:
                    vedic_weights = torch.ones_like(attention_mask, dtype=torch.float) * self.vedic_weight
                else:
                    vedic_weights = torch.ones_like(attention_mask, dtype=torch.float)
                
                result = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': input_ids.clone(),
                    'vedic_weights': vedic_weights,
                    'is_vedic': torch.tensor(1 if is_vedic else 0, dtype=torch.long),
                }
                
                # Add metadata (but skip problematic keys)
                for key, value in example.items():
                    if key not in ['text', 'tokens', 'is_vedic', 'is_pretokenized'] and not key.startswith('_'):
                        try:
                            if isinstance(value, (str, int, float, bool)):
                                result[key] = value
                            elif isinstance(value, list) and all(isinstance(x, (str, int, float, bool)) for x in value):
                                result[key] = value
                        except:
                            continue
                
                return result
            
            # Handle regular text that needs tokenization
            text = example.get('text', '')
            if not text:
                return None
            
            # Use Vedic tokenizer if available and content is Vedic
            if hasattr(self.tokenizer, 'encode_vedic_text') and is_vedic:
                token_ids = self.tokenizer.encode_vedic_text(
                    text,
                    add_vedic_markers=self.add_vedic_markers,
                    preserve_structure=self.preserve_structure,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                )
                
                input_ids = torch.tensor(token_ids, dtype=torch.long)
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
                
                if hasattr(self.tokenizer, 'create_vedic_attention_mask'):
                    vedic_weights = torch.tensor(
                        self.tokenizer.create_vedic_attention_mask(token_ids),
                        dtype=torch.float
                    )
                else:
                    vedic_weights = torch.ones_like(attention_mask, dtype=torch.float) * self.vedic_weight
                    
            else:
                # Fall back to standard tokenization
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].squeeze(0)
                attention_mask = encoding['attention_mask'].squeeze(0)
                
                # Apply Vedic weighting if this is Vedic content
                if is_vedic:
                    vedic_weights = torch.ones_like(attention_mask, dtype=torch.float) * self.vedic_weight
                else:
                    vedic_weights = torch.ones_like(attention_mask, dtype=torch.float)
            
            result = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': input_ids.clone(),
                'vedic_weights': vedic_weights,
                'is_vedic': torch.tensor(1 if is_vedic else 0, dtype=torch.long),
            }
            
            # Add metadata (but skip problematic keys)
            for key, value in example.items():
                if key not in ['text', 'tokens', 'is_vedic', 'is_pretokenized'] and not key.startswith('_'):
                    try:
                        if isinstance(value, (str, int, float, bool)):
                            result[key] = value
                        elif isinstance(value, list) and all(isinstance(x, (str, int, float, bool)) for x in value):
                            result[key] = value
                    except:
                        # Skip any problematic values
                        continue
            
            return result
            
        except Exception as e:
            logging.error(f"Error tokenizing Vedic example: {e}")
            return None


class StreamingInstructionDataset(StreamingINDRADataset):
    """Streaming version of InstructionDataset for memory-efficient instruction tuning."""
    
    def __len__(self):
        """Return estimated dataset length based on files and batch size."""
        # Instruction datasets typically have fewer examples per file
        estimated_examples_per_file = max(1, self.examples_per_batch // 20)
        return self.total_files * estimated_examples_per_file
    
    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer,
        instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}",
        response_loss_only: bool = True,
        **kwargs
    ):
        self.instruction_template = instruction_template
        self.response_loss_only = response_loss_only
        
        super().__init__(
            data_path,
            tokenizer,
            instruction_column="instruction",
            response_column="response",
            **kwargs
        )
    
    def _tokenize_example(self, example: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """Tokenize instruction-response example."""
        try:
            if 'instruction' in example and 'response' in example:
                # Format with template
                formatted_text = self.instruction_template.format(
                    instruction=example['instruction'],
                    response=example['response']
                )
                
                # Tokenize full text
                full_encoding = self.tokenizer(
                    formatted_text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = full_encoding['input_ids'].squeeze(0)
                attention_mask = full_encoding['attention_mask'].squeeze(0)
                
                if self.response_loss_only:
                    # Create labels that ignore instruction tokens
                    instruction_text = self.instruction_template.format(
                        instruction=example['instruction'],
                        response=""
                    ).rstrip()
                    
                    instruction_encoding = self.tokenizer(
                        instruction_text,
                        add_special_tokens=False,
                        return_tensors='pt'
                    )
                    
                    instruction_length = instruction_encoding['input_ids'].size(1)
                    
                    # Create labels (ignore instruction, predict response)
                    labels = input_ids.clone()
                    labels[:instruction_length] = -100
                else:
                    labels = input_ids.clone()
                
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                }
            else:
                # Fall back to standard processing
                return super()._tokenize_example(example)
                
        except Exception as e:
            logging.error(f"Error tokenizing instruction example: {e}")
            return None


# Utility function to create streaming dataset
def create_streaming_dataset(
    data_path: Union[str, List[str]],
    tokenizer,
    dataset_type: str = "base",
    batch_size_mb: float = 5.0,
    cache_dir: Optional[str] = None,
    **kwargs
) -> StreamingINDRADataset:
    """
    Create a streaming dataset of the specified type.
    
    Args:
        data_path: Path to data
        tokenizer: Tokenizer instance
        dataset_type: Type of dataset ('base', 'vedic', 'instruction')
        batch_size_mb: Memory batch size in MB
        cache_dir: Cache directory for processed data
        **kwargs: Additional dataset arguments
        
    Returns:
        Streaming dataset instance
    """
    common_args = {
        'batch_size_mb': batch_size_mb,
        'cache_dir': cache_dir,
        **kwargs
    }
    
    if dataset_type == "vedic":
        return StreamingVedicDataset(data_path, tokenizer, **common_args)
    elif dataset_type == "instruction":
        return StreamingInstructionDataset(data_path, tokenizer, **common_args)
    else:
        return StreamingINDRADataset(data_path, tokenizer, **common_args)

def safe_collate_fn(batch):
    """
    Custom collate function that handles inconsistent dict keys safely.
    """
    if not batch:
        return {}
    
    # Get all possible keys from all examples
    all_keys = set()
    for example in batch:
        if isinstance(example, dict):
            all_keys.update(example.keys())
    
    # Create the collated batch
    collated = {}
    
    for key in all_keys:
        values = []
        for example in batch:
            if isinstance(example, dict) and key in example:
                values.append(example[key])
            else:
                # Skip this example for this key if it doesn't have it
                continue
        
        if values:
            # Only collate if we have values
            try:
                if isinstance(values[0], torch.Tensor):
                    collated[key] = torch.stack(values)
                elif isinstance(values[0], (int, float)):
                    collated[key] = torch.tensor(values)
                elif isinstance(values[0], str):
                    collated[key] = values  # Keep as list for strings
                else:
                    collated[key] = values  # Keep as list for other types
            except Exception as e:
                # If collation fails, keep as list
                collated[key] = values
    
    return collated
