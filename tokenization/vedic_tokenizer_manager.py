import os
import logging
from pathlib import Path
from transformers import AutoTokenizer

from .sentencepiece_tokenizer import SentencePieceTokenizer
from .vedic_tokenizer import VedicTokenizer


class VedicTokenizerManager:
    def __init__(self, base_dir="./tokenizers", gemma_dir="gemma3_indra_tokenizer"):
        self.base_dir = Path(base_dir)
        self.gemma_dir = self.base_dir / gemma_dir
        self.tokenizer = None
        self.vedic_preprocessor = VedicTokenizer()  # reuse preprocessing logic

    def load_or_create(self, vocab_size=50000):
        """
        Load Gemma3 Indra tokenizer if available,
        otherwise create the Indra SentencePiece tokenizer.
        """
        # 1. Prefer gemma3_indra_tokenizer
        if self.gemma_dir.exists():
            logging.info(f"Found Gemma3 Indra tokenizer at {self.gemma_dir}, loading...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.gemma_dir))
        
        # 2. Else scan ./tokenizers for any other tokenizer
        elif self.base_dir.exists():
            for sub in self.base_dir.iterdir():
                if sub.is_dir() and (sub / "tokenizer.json").exists():
                    logging.info(f"Found tokenizer in {sub}, loading...")
                    self.tokenizer = AutoTokenizer.from_pretrained(str(sub))
                    break

        # 3. Else create fallback Indra tokenizer
        if self.tokenizer is None:
            logging.warning("No existing tokenizer found. Creating Indra SentencePieceTokenizer...")
            self.tokenizer = self._create_indra_tokenizer(vocab_size=vocab_size)

        return self.tokenizer

    def _create_indra_tokenizer(self, vocab_size=50000):
        """Build the fallback Indra tokenizer (SentencePiece)."""
        sp_tokenizer = SentencePieceTokenizer(vocab_size=vocab_size)
        
        # Expect a "corpus" folder with training texts
        corpus_dir = self.base_dir.parent / "corpus"
        input_files = [str(p) for p in corpus_dir.glob("*.txt")]
        
        model_prefix = str(self.base_dir / "indra_tokenizer")
        sp_tokenizer.train_tokenizer(input_files=input_files, model_prefix=model_prefix)
        
        logging.info(f"Indra tokenizer created and saved to {model_prefix}.model")
        return sp_tokenizer

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for pretraining using full VedicTokenizer logic.
        Ensures <vedic>, <verse>, <dharma>, etc. are injected.
        """
        return self.vedic_preprocessor.preprocess_vedic_text(text)
