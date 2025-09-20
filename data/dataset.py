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
    Memory-efficient base dataset class that "lazy loads" files one at a time.
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

        # Build a lightweight index of all examples without loading them into memory.
        logging.info("Building dataset index from file paths...")
        self.index = self._build_index()
        self.num_files_processed = len(set(item['file_path'] for item in self.index))
        logging.info(f"Dataset initialized. Found {len(self.index)} examples across {self.num_files_processed} files.")

    def _build_index(self) -> List[Dict]:
        """Scans all files and creates an index of examples (file path + chunk/line info)."""
        index = []
        all_files_to_process = []
        
        for path in self.data_paths:
            if os.path.isdir(path):
                for ext in ['*.txt', '*.json', '*.jsonl', '*.csv']:
                    all_files_to_process.extend(
                        glob.glob(os.path.join(path, '**', ext), recursive=True)
                    )
            elif os.path.isfile(path):
                all_files_to_process.append(path)

        all_files_to_process = [p for p in all_files_to_process if os.path.isfile(p)]

        for file_path in all_files_to_process:
            ext = Path(file_path).suffix.lower()
            if ext == '.txt' and self.split_documents:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    chunks = self._split_text(content)
                    for i, chunk in enumerate(chunks):
                         if len(chunk.strip()) >= self.min_length:
                            index.append({'file_path': file_path, 'chunk_id': i})
                except Exception as e:
                    logging.warning(f"Could not read or chunk {file_path}: {e}")
            elif ext == '.jsonl':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f):
                            if len(line.strip()) > 1:
                                index.append({'file_path': file_path, 'line_num': line_num})
                except Exception as e:
                    logging.warning(f"Could not process JSONL file {file_path}: {e}")
            else:
                 index.append({'file_path': file_path})
        
        return index

    def _split_text(self, text: str) -> List[str]:
        """Helper function to split long text into overlapping chunks."""
        if not self.split_documents or len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start += self.chunk_size - self.overlap_size
        return chunks

    def _get_raw_example(self, idx: int) -> Dict:
        """Loads a single raw example from disk based on the index."""
        item_info = self.index[idx]
        file_path = item_info['file_path']
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if 'chunk_id' in item_info:
                    content = f.read()
                    chunks = self._split_text(content)
                    return {'text': chunks[item_info['chunk_id']]}
                elif 'line_num' in item_info:
                    for i, line in enumerate(f):
                        if i == item_info['line_num']:
                            return json.loads(line)
                    return {}
                else:
                    ext = Path(file_path).suffix.lower()
                    if ext == '.json':
                        return json.load(f)
                    elif ext == '.csv':
                        return pd.read_csv(file_path).to_dict('records')[0]
                    else: # .txt
                        return {'text': f.read()}
        except Exception as e:
            logging.error(f"Error lazy-loading item {idx} from file {file_path}: {e}")
            return {'text': ''}
            
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Gets a raw example and performs standard tokenization."""
        raw_example = self._get_raw_example(idx)
        
        text_to_tokenize = raw_example.get(self.text_column, "")

        encoding = self.tokenizer(
            text_to_tokenize,
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
        
        # Add any additional metadata from the raw example
        for key, value in raw_example.items():
            if key not in ['text'] and not isinstance(value, (dict, list)):
                 result[key] = value

        return result

class VedicDataset(INDRADataset):
    """Specialized dataset for Vedic texts with enhanced processing."""
    
    def __init__(self, *args, **kwargs):
        self.vedic_weight = kwargs.pop('vedic_weight', 2.0)
        self.preserve_structure = kwargs.pop('preserve_structure', True)
        self.add_vedic_markers = kwargs.pop('add_vedic_markers', True)
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Gets a raw Vedic example and applies special processing."""
        example = self._get_raw_example(idx) # <-- Uses the lazy-loading method
        text = example.get('text', '')
        
        # This part remains the same, but now it operates on lazily-loaded data
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
        
        for key, value in example.items():
            if key not in ['text'] and not isinstance(value, (dict, list)):
                result[key] = value
        
        return result

class InstructionDataset(INDRADataset):
    """Dataset for instruction-following fine-tuning."""
    
    def __init__(self, *args, **kwargs):
        self.instruction_template = kwargs.pop('instruction_template', "### Instruction:\n{instruction}\n\n### Response:\n{response}")
        self.response_loss_only = kwargs.pop('response_loss_only', True)
        super().__init__(*args, instruction_column="instruction", response_column="response", **kwargs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Gets a raw instruction example and applies SFT formatting."""
        example = self._get_raw_example(idx) # <-- Uses the lazy-loading method
        
        instruction = example.get('instruction', '')
        response = example.get('response', '')

        if instruction and response:
            formatted_text = self.instruction_template.format(instruction=instruction, response=response)
            
            full_encoding = self.tokenizer(
                formatted_text, max_length=self.max_length, padding='max_length',
                truncation=True, return_tensors='pt'
            )
            input_ids = full_encoding['input_ids'].squeeze(0)
            
            labels = input_ids.clone()
            if self.response_loss_only:
                instruction_text = self.instruction_template.format(instruction=instruction, response="").rstrip()
                instruction_encoding = self.tokenizer(instruction_text, add_special_tokens=False, return_tensors='pt')
                instruction_length = instruction_encoding['input_ids'].size(1)
                labels[:instruction_length] = -100
            
            return {
                'input_ids': input_ids,
                'attention_mask': full_encoding['attention_mask'].squeeze(0),
                'labels': labels,
            }
        else:
            # Fallback for non-instruction data
            return super().__getitem__(idx)
