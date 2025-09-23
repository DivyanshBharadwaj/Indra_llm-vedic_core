# -*- coding: utf-8 -*-
"""
Comprehensive script to clean, preprocess, and tokenize a large, multi-format dataset.

This script is designed to process the 'Divyansh008/dataset' by:
1. Loading a custom Hugging Face Gemma3-compatible tokenizer.
2. Using the advanced text cleaning and normalization logic from the user's
   'vedic_tokenizer.py' to handle multi-lingual text, especially Sanskrit.
3. Recursively scanning a source directory with multiple data formats
   (.txt, .json, .jsonl, .csv).
4. Extracting text content from each file.
5. Cleaning and tokenizing the text.
6. Saving the tokenized 'input_ids' into a parallel directory structure,
   preserving the original organization while converting files to a consistent
   .jsonl format.

(c) Divyansh Bharadwaj - Adapted and integrated from original project files.
"""
import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Generator, Any

import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

# --- CONFIGURATION ---
# Set the main paths for your project.
# It's assumed you have downloaded the Hugging Face dataset locally.
INPUT_DATASET_DIR = Path("Divyansh008/dataset")
OUTPUT_DIR = Path("tokenized_dataset")
TOKENIZER_PATH = Path("./tokenizers/gemma3_indra_tokenizer") # Path to your custom tokenizer
LOG_FILE = "preprocessing.log"

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)

# --- TEXT CLEANING MODULE ---
# This class incorporates the brilliant text cleaning logic from your
# `vedic_tokenizer.py`. It's used here as a dedicated text preprocessor.

class TextCleaner:
    """
    A dedicated class for cleaning and normalizing text, especially Vedic and
    Sanskrit content, adapted from the original VedicTokenizer.
    """
    def __init__(self, vedic_concepts_path: Optional[str] = None, sanskrit_rules_path: Optional[str] = None):
        """Initializes the text cleaner with concepts and rules."""
        self.vedic_concepts = self._load_json_data(vedic_concepts_path, self._default_vedic_concepts())
        self.sanskrit_rules = self._load_json_data(sanskrit_rules_path, self._default_sanskrit_rules())
        self.concept_patterns = self._build_concept_patterns()
        self.iast_to_dev_map = self._get_iast_to_devanagari_map()

    def _load_json_data(self, path: Optional[str], default_data: Dict) -> Dict:
        """Loads a JSON file, falling back to default data if not found."""
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    # Merge loaded data with defaults
                    for key, value in loaded_data.items():
                        if key in default_data and isinstance(default_data[key], list):
                            default_data[key].extend(value)
                        else:
                            default_data[key] = value
                    return default_data
            except Exception as e:
                logging.warning(f"Could not load data from {path}: {e}")
        return default_data

    def _build_concept_patterns(self) -> Dict[str, re.Pattern]:
        """Builds regex patterns for identifying Vedic concepts."""
        patterns = {}
        for category, concepts in self.vedic_concepts.items():
            escaped = [re.escape(c) for c in concepts]
            pattern_str = r'\b(?:' + '|'.join(escaped) + r')\b'
            patterns[category] = re.compile(pattern_str, re.IGNORECASE | re.UNICODE)
        return patterns

    def preprocess_vedic_text(self, text: str) -> str:
        """
        Main preprocessing function to clean and normalize a given text string.
        """
        if not isinstance(text, str):
            return ""
        text = self._normalize_sanskrit(text)
        # The original code marked concepts with special tokens, which might not be
        # in the Gemma tokenizer. For pre-tokenization, we will just normalize
        # and clean without adding new tokens. If you need concept marking,
        # ensure those tokens (<dharma>, <karma>) exist in your final tokenizer.
        # text = self._mark_vedic_concepts(text) # Disabled by default
        text = self._handle_sanskrit_compounds(text)
        return text.strip()

    def _normalize_sanskrit(self, text: str) -> str:
        """Normalizes Sanskrit text, including IAST to Devanagari conversion."""
        text = re.sub(self.sanskrit_rules.get("normalize_anusvara", r'[ंॅ]'), 'ं', text)
        text = re.sub(self.sanskrit_rules.get("normalize_visarga", r'[ःḥ]'), 'ḥ', text)

        for iast, dev in self.iast_to_dev_map.items():
            text = text.replace(iast, dev)
        return text

    def _handle_sanskrit_compounds(self, text: str) -> str:
        """
        A simplified placeholder for the complex sandhi rule handling.
        The full sandhi logic from your file is extensive. For a general-purpose
        cleaner, focusing on normalization and whitespace is often sufficient.
        You can paste the full `_handle_sanskrit_compounds` method here if needed.
        """
        # Cleans up multiple spaces that might result from other processing.
        text = re.sub(r'\s+', ' ', text)
        return text

    # --- Default Data and Mappings ---
    # These methods provide the default concepts, rules, and mappings.
    def _default_vedic_concepts(self) -> Dict:
        return {
            "dharma_related": ["dharma", "धर्म", "righteousness", "duty"],
            "karma_related": ["karma", "कर्म", "action", "deed"],
            "moksha_related": ["moksha", "मोक्ष", "liberation", "release"],
        }

    def _default_sanskrit_rules(self) -> Dict:
        return {
            "normalize_anusvara": r'[ंॅ]',
            "normalize_visarga": r'[ःḥ]',
        }

    def _get_iast_to_devanagari_map(self) -> Dict:
        return {
            'a': 'अ', 'ā': 'आ', 'i': 'इ', 'ī': 'ई', 'u': 'उ', 'ū': 'ऊ',
            'ṛ': 'ऋ', 'ṝ': 'ॠ', 'ḷ': 'ऌ', 'ḹ': 'ॡ', 'e': 'ए', 'ai': 'ऐ',
            'o': 'ओ', 'au': 'औ', 'k': 'क', 'kh': 'ख', 'g': 'ग', 'gh': 'घ',
            'ṅ': 'ङ', 'c': 'च', 'ch': 'छ', 'j': 'ज', 'jh': 'झ', 'ñ': 'ञ',
            'ṭ': 'ट', 'ṭh': 'ठ', 'ḍ': 'ड', 'ḍh': 'ढ', 'ṇ': 'ण', 't': 'त',
            'th': 'थ', 'd': 'द', 'dh': 'ध', 'n': 'न', 'p': 'प', 'ph': 'फ',
            'b': 'ब', 'bh': 'भ', 'm': 'म', 'y': 'य', 'r': 'र', 'l': 'ल',
            'v': 'व', 'ś': 'श', 'ṣ': 'ष', 's': 'स', 'h': 'ह', 'ṃ': 'ं',
            'ḥ': 'ः', 'oṃ': 'ॐ', '.': '।', '..': '॥'
        }


# --- DATA FILE PROCESSING ---
# This section contains functions to extract text from different file types,
# inspired by the logic in your `dataset.py`.

def extract_text_from_item(item: Any, text_keys: List[str]) -> Optional[str]:
    """Extracts text from a dictionary item by checking common keys."""
    if not isinstance(item, dict):
        return None
    for key in text_keys:
        if key in item and isinstance(item[key], str) and item[key].strip():
            return item[key].strip()
    return None

def read_file_content(file_path: Path) -> Generator[str, None, None]:
    """
    Reads a file and yields text content, handling different formats.
    """
    text_keys = ["text", "content", "body", "message", "response", "instruction"]
    file_ext = file_path.suffix.lower()

    try:
        if file_ext == ".txt":
            with file_path.open('r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    yield content
        elif file_ext in [".json", ".jsonl"]:
            with file_path.open('r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if isinstance(data, list):
                            for item in data:
                                text = extract_text_from_item(item, text_keys)
                                if text: yield text
                        else:
                            text = extract_text_from_item(data, text_keys)
                            if text: yield text
                    except json.JSONDecodeError:
                        logging.warning(f"Skipping invalid JSON line in {file_path}")
                        continue
        elif file_ext == ".csv":
            for chunk in pd.read_csv(file_path, chunksize=1000, on_bad_lines='warn'):
                for _, row in chunk.iterrows():
                    text = extract_text_from_item(row.to_dict(), text_keys)
                    if text: yield text
    except Exception as e:
        logging.error(f"Failed to read or process {file_path}: {e}")

# --- MAIN ORCHESTRATION ---

def main():
    """
    Main function to orchestrate the dataset cleaning and tokenization process.
    """
    logging.info("Starting dataset preprocessing script.")
    logging.info(f"Input Directory: {INPUT_DATASET_DIR}")
    logging.info(f"Output Directory: {OUTPUT_DIR}")
    logging.info(f"Tokenizer Path: {TOKENIZER_PATH}")

    # 1. Validate paths
    if not INPUT_DATASET_DIR.is_dir():
        logging.error(f"Input directory not found: {INPUT_DATASET_DIR}")
        return
    if not TOKENIZER_PATH.is_dir():
        logging.error(f"Tokenizer directory not found: {TOKENIZER_PATH}")
        return

    # 2. Load tokenizer and cleaner
    try:
        logging.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
        logging.info("Tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        return

    cleaner = TextCleaner()
    logging.info("Text cleaner initialized.")

    # 3. Find all processable files
    supported_extensions = ['.txt', '.json', '.jsonl', '.csv']
    files_to_process = [
        p for p in INPUT_DATASET_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in supported_extensions
    ]

    if not files_to_process:
        logging.warning("No supported files found in the input directory.")
        return

    logging.info(f"Found {len(files_to_process)} files to process.")

    # 4. Process each file
    for input_file in tqdm(files_to_process, desc="Processing files"):
        try:
            # Create the corresponding output path
            relative_path = input_file.relative_to(INPUT_DATASET_DIR)
            output_file = OUTPUT_DIR / relative_path.with_suffix(".tokenized.jsonl")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            processed_count = 0
            with output_file.open('w', encoding='utf-8') as f_out:
                # Extract text records from the source file
                for text_record in read_file_content(input_file):
                    # Clean the text using the Vedic-aware cleaner
                    cleaned_text = cleaner.preprocess_vedic_text(text_record)

                    if not cleaned_text:
                        continue

                    # Tokenize the cleaned text
                    # We add special tokens (like BOS/EOS) as it's common for pre-training.
                    token_ids = tokenizer.encode(cleaned_text, add_special_tokens=True)

                    # Write the tokenized output as a JSON line
                    f_out.write(json.dumps({"input_ids": token_ids}) + "\n")
                    processed_count += 1
            
            if processed_count > 0:
                logging.info(f"Successfully processed {input_file.name} -> {output_file.name} ({processed_count} records)")
            else:
                logging.warning(f"No text records were extracted from {input_file.name}")

        except Exception as e:
            logging.error(f"An unexpected error occurred while processing {input_file}: {e}", exc_info=True)

    logging.info("="*50)
    logging.info("Dataset preprocessing and tokenization complete.")
    logging.info(f"Tokenized files are saved in: {OUTPUT_DIR}")
    logging.info(f"Check the log file '{LOG_FILE}' for details and errors.")
    logging.info("="*50)


if __name__ == "__main__":
    main()
