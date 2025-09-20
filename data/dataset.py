"""
Dataset classes for INDRA LLM with support for multiple formats and languages
(c) Divyansh Bharadwaj
"""

import os
import glob
import json
import logging
import random
from typing import List, Dict, Optional, Union, Iterator, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset
import pandas as pd
from datasets import load_dataset, Dataset as HFDataset

class INDRADataset(Dataset):
    """
    Base dataset class supporting multiple data formats:
    - .txt files (raw text)
    - .json files (structured data)
    - .csv files (tabular data)
    - HuggingFace datasets
    """
    
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
    ):
        """
        Initialize INDRA dataset.
        
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
        
        # Load and process data
        self.examples = []
        self.num_files_processed = 0
        self._load_data()

        # Wrap examples in a HuggingFace dataset for map/filter support
        if len(self.examples) > 0:
            self.hf_dataset = HFDataset.from_list(self.examples)
        else:
            self.hf_dataset = HFDataset.from_list([])

        logging.info(f"Loaded {len(self.examples)} examples from {self.num_files_processed} files")

    
    def _detect_data_type(self, path: str) -> str:
        """Auto-detect data type from file extension or content."""
        if os.path.isfile(path):
            ext = Path(path).suffix.lower()
            if ext == '.txt':
                return 'txt'
            elif ext == '.json' or ext == '.jsonl':
                return 'json'
            elif ext == '.csv':
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
        else:
            # Assume HuggingFace dataset
            return 'hf'
    
    # REPLACE the old _load_data method in dataset.py with this one
    # def _load_data(self):
    #     """Load data from all specified paths, correctly handling directories."""
    #     all_files_to_process = []
        
    #     # First, expand all directory paths into a list of files
    #     for path in self.data_paths:
    #         if os.path.isdir(path):
    #             # Use glob to find all .txt, .json, .jsonl, and .csv files recursively
    #             for ext in ['*.txt', '*.json', '*.jsonl', '*.csv']:
    #                 all_files_to_process.extend(
    #                     glob.glob(os.path.join(path, '**', ext), recursive=True)
    #                 )
    #         elif os.path.isfile(path):
    #             all_files_to_process.append(path)
        
    #     if not all_files_to_process:
    #         # Handle HuggingFace dataset names if no local files are found
    #         if self.data_type == 'hf' or (self.data_type == 'auto' and not any(os.path.exists(p) for p in self.data_paths)):
    #              for path in self.data_paths:
    #                 try:
    #                     self._load_hf_dataset(path)
    #                 except Exception as e:
    #                     logging.error(f"Error loading HuggingFace dataset {path}: {e}")
    #         return # Exit if no files were found

    #     # Now, process each file individually
    #     for file_path in all_files_to_process:
    #         if self.data_type == "auto":
    #             # Detect type based on the specific file, not the original path
    #             detected_type = self._detect_data_type(file_path)
    #         else:
    #             detected_type = self.data_type
            
    #         try:
    #             if detected_type == 'txt':
    #                 self._load_txt_file(file_path)
    #             elif detected_type == 'json':
    #                 self._load_json_file(file_path)
    #             elif detected_type == 'csv':
    #                 self._load_csv_file(file_path)
    #             else:
    #                 logging.warning(f"Skipping unsupported file type for {file_path}")
    #         except Exception as e:
    #             logging.error(f"Error loading file {file_path}: {e}")

    # --- Replace the _load_data method with this ---
    def _load_data(self):
        """Load data from all specified paths, correctly handling directories."""
        all_files_to_process = []
        
        for path in self.data_paths:
            if os.path.isdir(path):
                for ext in ['*.txt', '*.json', '*.jsonl', '*.csv']:
                    all_files_to_process.extend(
                        glob.glob(os.path.join(path, '**', ext), recursive=True)
                    )
            elif os.path.isfile(path):
                all_files_to_process.append(path)
        
        self.num_files_processed = len(all_files_to_process) # <-- ADD THIS LINE
    
        if not all_files_to_process:
            if self.data_type == 'hf' or (self.data_type == 'auto' and not any(os.path.exists(p) for p in self.data_paths)):
                 for path in self.data_paths:
                    try:
                        self._load_hf_dataset(path)
                    except Exception as e:
                        logging.error(f"Error loading HuggingFace dataset {path}: {e}")
            return
    
        for file_path in all_files_to_process:
            if self.data_type == "auto":
                detected_type = self._detect_data_type(file_path)
            else:
                detected_type = self.data_type

            print(f"DEBUG: Processing '{file_path}' | Detected type: '{detected_type}'")
            
            try:
                if detected_type == 'txt':
                    self._load_txt_file(file_path)
                elif detected_type == 'json':
                    self._load_json_file(file_path)
                elif detected_type == 'csv':
                    self._load_csv_file(file_path)
                else:
                    logging.warning(f"Skipping unsupported file type for {file_path}")
            except Exception as e:
                logging.error(f"Error loading file {file_path}: {e}")
    
    def _load_txt_file(self, path: str):
        """Load plain text file."""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if self.split_documents and len(content) > self.chunk_size:
            # Split into overlapping chunks
            chunks = self._split_text(content)
            for chunk in chunks:
                if len(chunk.strip()) >= self.min_length:
                    self.examples.append({'text': chunk.strip()})
        else:
            if len(content.strip()) >= self.min_length:
                self.examples.append({'text': content.strip()})
    
    def _load_json_file(self, path: str):
        """Load JSON or JSONL file."""
        with open(path, 'r', encoding='utf-8') as f:
            # Try to load as single JSON object first
            try:
                f.seek(0)
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        self._process_json_item(item)
                elif isinstance(data, dict):
                    self._process_json_item(data)
            except json.JSONDecodeError:
                # Try as JSONL
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            self._process_json_item(item)
                        except json.JSONDecodeError as e:
                            logging.warning(f"Skipping invalid JSON line: {line[:100]}...")
    
    def _process_json_item(self, item: Dict):
        """Process a single JSON item."""
        if isinstance(item, dict):
            # Extract text based on expected columns
            if self.instruction_column and self.response_column:
                # Instruction-response format
                if self.instruction_column in item and self.response_column in item:
                    instruction = item[self.instruction_column].strip()
                    response = item[self.response_column].strip()
                    if instruction and response:
                        self.examples.append({
                            'instruction': instruction,
                            'response': response,
                            'text': f"{instruction}\n{response}"
                        })
            elif self.text_column in item:
                # Simple text format
                text = item[self.text_column].strip()
                if len(text) >= self.min_length:
                    processed_item = {'text': text}
                    # Copy other fields
                    for key, value in item.items():
                        if key != self.text_column:
                            processed_item[key] = value
                    self.examples.append(processed_item)
            else:
                # Try common text fields
                for field in ['text', 'content', 'body', 'message']:
                    if field in item:
                        text = str(item[field]).strip()
                        if len(text) >= self.min_length:
                            self.examples.append({'text': text})
                        break
    
    def _load_csv_file(self, path: str):
        """Load CSV file."""
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                self._process_json_item(row.to_dict())
        except Exception as e:
            logging.error(f"Error loading CSV {path}: {e}")
    
    def _load_hf_dataset(self, dataset_name: str):
        """Load HuggingFace dataset."""
        try:
            # Try loading with different splits
            for split in ['train', 'validation', 'test']:
                try:
                    dataset = load_dataset(dataset_name, split=split)
                    for item in dataset:
                        self._process_json_item(item)
                    break
                except Exception:
                    continue
            else:
                # Try loading without split
                dataset = load_dataset(dataset_name)
                if isinstance(dataset, dict):
                    # Multiple splits
                    for split_name, split_data in dataset.items():
                        for item in split_data:
                            self._process_json_item(item)
                else:
                    # Single dataset
                    for item in dataset:
                        self._process_json_item(item)
        except Exception as e:
            logging.error(f"Error loading HuggingFace dataset {dataset_name}: {e}")
    
    def _split_text(self, text: str) -> List[str]:
        """Split long text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Try to find a good breaking point (sentence end, paragraph break)
            chunk = text[start:end]
            
            # Look for sentence boundaries in the last part of the chunk
            for delimiter in ['\n\n', '\n', '. ', '! ', '? ']:
                last_pos = chunk.rfind(delimiter)
                if last_pos > self.chunk_size // 2:  # Don't break too early
                    chunk = text[start:start + last_pos + len(delimiter)]
                    break
            
            chunks.append(chunk)
            start = start + len(chunk) - self.overlap_size
        
        return chunks
    
    def __len__(self) -> int:
        return len(self.hf_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        example = self.hf_dataset[idx]
        
        # Tokenize text
        text = example['text']
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
        
        # Add any additional fields
        for key, value in example.items():
            if key not in ['text'] and not key.startswith('_'):
                result[key] = value
        
        return result

    # Allow HuggingFace-style operations (map, filter, shuffle, etc.)
    def map(self, *args, **kwargs):
        self.hf_dataset = self.hf_dataset.map(*args, **kwargs)
        return self

    def filter(self, *args, **kwargs):
        self.hf_dataset = self.hf_dataset.filter(*args, **kwargs)
        return self

    def shuffle(self, *args, **kwargs):
        self.hf_dataset = self.hf_dataset.shuffle(*args, **kwargs)
        return self


class VedicDataset(INDRADataset):
    """Specialized dataset for Vedic texts with enhanced processing."""
    
    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer,
        vedic_weight: float = 2.0,
        preserve_structure: bool = True,
        add_vedic_markers: bool = True,
        **kwargs
    ):
        """
        Initialize Vedic dataset.
        
        Args:
            data_path: Path to Vedic text files
            tokenizer: VedicTokenizer instance
            vedic_weight: Sampling weight for Vedic content
            preserve_structure: Whether to preserve verse structure
            add_vedic_markers: Whether to add Vedic content markers
            **kwargs: Additional arguments for base dataset
        """
        self.vedic_weight = vedic_weight
        self.preserve_structure = preserve_structure
        self.add_vedic_markers = add_vedic_markers
        
        super().__init__(data_path, tokenizer, **kwargs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single Vedic example with enhanced processing."""
        example = self.examples[idx]
        text = example['text']
        
        # Use Vedic tokenizer's specialized encoding if available
        if hasattr(self.tokenizer, 'encode_vedic_text'):
            token_ids = self.tokenizer.encode_vedic_text(
                text,
                add_vedic_markers=self.add_vedic_markers,
                preserve_structure=self.preserve_structure,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
            )
            
            # Convert to tensors
            input_ids = torch.tensor(token_ids, dtype=torch.long)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            
            # Create Vedic attention weights if supported
            if hasattr(self.tokenizer, 'create_vedic_attention_mask'):
                vedic_weights = torch.tensor(
                    self.tokenizer.create_vedic_attention_mask(token_ids),
                    dtype=torch.float
                )
            else:
                vedic_weights = torch.ones_like(attention_mask, dtype=torch.float)
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
            vedic_weights = torch.ones_like(attention_mask, dtype=torch.float) * self.vedic_weight
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone(),
            'vedic_weights': vedic_weights,
            'is_vedic': torch.tensor(1, dtype=torch.long),
        }
        
        # Add metadata
        for key, value in example.items():
            if key not in ['text'] and not key.startswith('_'):
                result[key] = value
        
        return result

class MultiLanguageDataset(INDRADataset):
    """Dataset for handling multiple languages with appropriate sampling."""
    
    def __init__(
        self,
        data_configs: List[Dict],
        tokenizer,
        sampling_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        Initialize multi-language dataset.
        
        Args:
            data_configs: List of data configuration dicts with 'path' and 'language'
            tokenizer: Tokenizer instance
            sampling_weights: Sampling weights for different languages
            **kwargs: Additional arguments for base dataset
        """
        self.data_configs = data_configs
        self.sampling_weights = sampling_weights or {}
        
        # Initialize with empty examples
        self.tokenizer = tokenizer
        self.max_length = kwargs.get('max_length', 2048)
        self.examples = []
        self.language_indices = {}  # Track which examples belong to which language
        
        # Load data from all configurations
        for config in data_configs:
            lang = config.get('language', 'unknown')
            start_idx = len(self.examples)
            
            # Create temporary dataset for this language
            temp_dataset = INDRADataset(
                data_path=config['path'],
                tokenizer=tokenizer,
                **{k: v for k, v in kwargs.items() if k not in ['data_path']}
            )
            
            # Add language info to examples
            for example in temp_dataset.examples:
                example['language'] = lang
                self.examples.append(example)
            
            end_idx = len(self.examples)
            self.language_indices[lang] = list(range(start_idx, end_idx))
        
        logging.info(f"Loaded multi-language dataset with {len(self.examples)} examples")
        for lang, indices in self.language_indices.items():
            logging.info(f"  {lang}: {len(indices)} examples")
    
    def get_language_statistics(self) -> Dict[str, int]:
        """Get statistics about language distribution."""
        return {lang: len(indices) for lang, indices in self.language_indices.items()}
    
    def sample_by_language(self, batch_size: int) -> List[int]:
        """Sample indices with language-aware weighting."""
        if not self.sampling_weights:
            # Uniform sampling
            return random.choices(range(len(self.examples)), k=batch_size)
        
        # Weighted sampling
        indices = []
        for _ in range(batch_size):
            # Choose language based on weights
            languages = list(self.language_indices.keys())
            weights = [self.sampling_weights.get(lang, 1.0) for lang in languages]
            chosen_lang = random.choices(languages, weights=weights, k=1)[0]
            
            # Choose random example from chosen language
            lang_indices = self.language_indices[chosen_lang]
            indices.append(random.choice(lang_indices))
        
        return indices

class InstructionDataset(INDRADataset):
    """Dataset for instruction-following fine-tuning."""
    
    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer,
        instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}",
        response_loss_only: bool = True,
        **kwargs
    ):
        """
        Initialize instruction dataset.
        
        Args:
            data_path: Path to instruction data
            tokenizer: Tokenizer instance
            instruction_template: Template for formatting instruction-response pairs
            response_loss_only: Whether to compute loss only on response tokens
            **kwargs: Additional arguments for base dataset
        """
        self.instruction_template = instruction_template
        self.response_loss_only = response_loss_only
        
        super().__init__(
            data_path,
            tokenizer,
            instruction_column="instruction",
            response_column="response",
            **kwargs
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single instruction-response example."""
        example = self.examples[idx]
        
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
                labels[:instruction_length] = -100  # Ignore instruction tokens
            else:
                labels = input_ids.clone()
            
            result = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }
            
        else:
            # Fall back to standard processing
            result = super().__getitem__(idx)
        
        return result


