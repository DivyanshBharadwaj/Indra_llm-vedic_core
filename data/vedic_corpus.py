"""
Vedic corpus processing for INDRA LLM with Sanskrit text handling
(c) Divyansh Bharadwaj
"""

import os
import json
import logging
import re
from typing import List, Dict, Optional, Tuple, Iterator
from pathlib import Path
from dataclasses import dataclass

@dataclass
class VedicText:
    """Structure for Vedic text with metadata."""
    text: str
    collection: str  # rigveda, yajurveda, etc.
    book: Optional[int] = None
    hymn: Optional[int] = None
    verse: Optional[int] = None
    language: str = "sanskrit"
    translation: Optional[str] = None
    commentary: Optional[str] = None
    source: Optional[str] = None
    
class VedicCorpusProcessor:
    """Processor for Vedic texts with Sanskrit support and structure preservation."""
    
    def __init__(
        self,
        vedic_texts_dir: str,
        output_dir: str,
        include_translations: bool = True,
        include_commentaries: bool = True,
        normalize_sanskrit: bool = True,
        split_by_verses: bool = True,
    ):
        """
        Initialize Vedic corpus processor.
        
        Args:
            vedic_texts_dir: Directory containing Vedic text files
            output_dir: Output directory for processed corpus
            include_translations: Whether to include translations
            include_commentaries: Whether to include commentaries
            normalize_sanskrit: Whether to normalize Sanskrit text
            split_by_verses: Whether to split by individual verses
        """
        self.vedic_texts_dir = Path(vedic_texts_dir)
        self.output_dir = Path(output_dir)
        self.include_translations = include_translations
        self.include_commentaries = include_commentaries
        self.normalize_sanskrit = normalize_sanskrit
        self.split_by_verses = split_by_verses
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanskrit normalization patterns
        self.sanskrit_patterns = {
            # IAST to Devanagari basic mappings
            'iast_to_devanagari': {
                'ā': 'आ', 'ī': 'ई', 'ū': 'ऊ', 'ṛ': 'ऋ', 'ḷ': 'ऌ',
                'ē': 'ए', 'ō': 'ओ', 'ṃ': 'ं', 'ḥ': 'ः', 'r̥': 'ऋ',
                'ṅ': 'ङ', 'ñ': 'ञ', 'ṭ': 'ट', 'ḍ': 'ड', 'ṇ': 'ण',
                'ś': 'श', 'ṣ': 'ष', 'kh': 'ख', 'gh': 'घ', 'ch': 'छ',
                'jh': 'झ', 'th': 'थ', 'dh': 'ध', 'ph': 'फ', 'bh': 'भ'
            },
            
            # Verse structure patterns
            'verse_markers': [
                r'^\d+\.',  # Numbered verses
                r'^[IVX]+\.',  # Roman numeral verses
                r'॥.*?॥',  # Sanskrit verse markers
                r'\|\|.*?\|\|',  # Alternative verse markers
            ],
            
            # Sanskrit word patterns
            'sanskrit_words': r'[\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]+',
            'devanagari_range': r'[\u0900-\u097F]',
            'vedic_accents': r'[\u1CD0-\u1CFF]',
        }
        
        # Vedic collection patterns
        self.collection_patterns = {
            'rigveda': ['rigveda', 'rig veda', 'rv', 'ṛgveda'],
            'yajurveda': ['yajurveda', 'yajur veda', 'yv'],
            'samaveda': ['samaveda', 'sama veda', 'sv', 'sāmaveda'],
            'atharvaveda': ['atharvaveda', 'atharva veda', 'av'],
            'upanishad': ['upanishad', 'upaniṣad', 'upanishads'],
            'brahmana': ['brahmana', 'brāhmaṇa', 'brahmanas'],
            'aranyaka': ['aranyaka', 'āraṇyaka', 'aranyakas'],
            'sutra': ['sutra', 'sūtra', 'sutras'],
        }
        
        # Processed texts storage
        self.processed_texts: List[VedicText] = []
    
    def process_corpus(self) -> None:
        """Process entire Vedic corpus."""
        logging.info(f"Processing Vedic corpus from {self.vedic_texts_dir}")
        
        # Process different file formats
        for file_path in self.vedic_texts_dir.rglob("*"):
            if file_path.is_file():
                try:
                    if file_path.suffix.lower() == '.json':
                        self._process_json_file(file_path)
                    elif file_path.suffix.lower() == '.txt':
                        self._process_txt_file(file_path)
                    elif file_path.suffix.lower() == '.xml':
                        self._process_xml_file(file_path)
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}")
        
        logging.info(f"Processed {len(self.processed_texts)} Vedic texts")
        
        # Save processed corpus
        self._save_processed_corpus()
        
        # Generate statistics
        self._generate_statistics()
    
    def _process_json_file(self, file_path: Path) -> None:
        """Process JSON file containing Vedic texts."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                self._process_json_item(item, file_path.stem)
        elif isinstance(data, dict):
            if 'texts' in data:
                for item in data['texts']:
                    self._process_json_item(item, file_path.stem)
            else:
                self._process_json_item(data, file_path.stem)
    
    def _process_json_item(self, item: Dict, source: str) -> None:
        """Process individual JSON item."""
        # Extract text content
        text = item.get('text', item.get('sanskrit', item.get('content', '')))
        if not text:
            return
        
        # Determine collection
        collection = self._identify_collection(item, source)
        
        # Extract metadata
        vedic_text = VedicText(
            text=text,
            collection=collection,
            book=item.get('book', item.get('mandala')),
            hymn=item.get('hymn', item.get('sukta')),
            verse=item.get('verse', item.get('rik')),
            language=item.get('language', 'sanskrit'),
            translation=item.get('translation', item.get('english')) if self.include_translations else None,
            commentary=item.get('commentary', item.get('comment')) if self.include_commentaries else None,
            source=source,
        )
        
        # Process and normalize text
        self._process_vedic_text(vedic_text)
    
    def _process_txt_file(self, file_path: Path) -> None:
        """Process plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        collection = self._identify_collection({'source': file_path.name}, file_path.stem)
        
        if self.split_by_verses:
            verses = self._split_into_verses(content)
            for i, verse in enumerate(verses):
                if verse.strip():
                    vedic_text = VedicText(
                        text=verse.strip(),
                        collection=collection,
                        verse=i + 1,
                        language='sanskrit',
                        source=file_path.stem,
                    )
                    self._process_vedic_text(vedic_text)
        else:
            vedic_text = VedicText(
                text=content.strip(),
                collection=collection,
                language='sanskrit',
                source=file_path.stem,
            )
            self._process_vedic_text(vedic_text)
    
    def _process_xml_file(self, file_path: Path) -> None:
        """Process XML file (basic implementation)."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            collection = self._identify_collection({'source': file_path.name}, file_path.stem)
            
            # Extract text elements (adapt based on XML structure)
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    # Check if this is likely Vedic content
                    if self._contains_sanskrit(elem.text):
                        vedic_text = VedicText(
                            text=elem.text.strip(),
                            collection=collection,
                            language='sanskrit',
                            source=file_path.stem,
                        )
                        self._process_vedic_text(vedic_text)
        except ImportError:
            logging.warning(f"xml.etree.ElementTree not available, skipping {file_path}")
        except Exception as e:
            logging.error(f"Error processing XML file {file_path}: {e}")
    
    def _identify_collection(self, item: Dict, source: str) -> str:
        """Identify which Vedic collection this text belongs to."""
        # Check explicit collection field
        if 'collection' in item:
            return item['collection'].lower()
        
        # Check source filename and content
        source_lower = source.lower()
        text_lower = str(item).lower()
        
        for collection, patterns in self.collection_patterns.items():
            for pattern in patterns:
                if pattern in source_lower or pattern in text_lower:
                    return collection
        
        return 'unknown'
    
    def _contains_sanskrit(self, text: str) -> bool:
        """Check if text contains Sanskrit/Devanagari characters."""
        return bool(re.search(self.sanskrit_patterns['devanagari_range'], text))
    
    def _split_into_verses(self, text: str) -> List[str]:
        """Split text into individual verses."""
        verses = []
        
        # Try different verse splitting strategies
        for pattern in self.sanskrit_patterns['verse_markers']:
            matches = list(re.finditer(pattern, text))
            if matches:
                # Split by verse markers
                last_end = 0
                for match in matches:
                    if last_end < match.start():
                        verse_text = text[last_end:match.start()].strip()
                        if verse_text:
                            verses.append(verse_text)
                    last_end = match.end()
                
                # Add remaining text
                if last_end < len(text):
                    remaining = text[last_end:].strip()
                    if remaining:
                        verses.append(remaining)
                
                if verses:
                    return verses
        
        # Fallback: split by double newlines
        verses = [v.strip() for v in text.split('\n\n') if v.strip()]
        if len(verses) > 1:
            return verses
        
        # Fallback: split by single newlines
        verses = [v.strip() for v in text.split('\n') if v.strip()]
        return verses
    
    def _process_vedic_text(self, vedic_text: VedicText) -> None:
        """Process and normalize individual Vedic text."""
        # Normalize Sanskrit if requested
        if self.normalize_sanskrit:
            vedic_text.text = self._normalize_sanskrit_text(vedic_text.text)
            if vedic_text.translation:
                vedic_text.translation = self._normalize_english_text(vedic_text.translation)
        
        # Add to processed texts
        self.processed_texts.append(vedic_text)
    
    def _normalize_sanskrit_text(self, text: str) -> str:
        """Normalize Sanskrit text for consistency."""
        # Convert IAST to Devanagari where possible
        for iast, devanagari in self.sanskrit_patterns['iast_to_devanagari'].items():
            text = text.replace(iast, devanagari)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove or normalize verse markers
        text = re.sub(r'॥\s*॥', '॥', text)  # Normalize double dandas
        text = re.sub(r'\|\|\s*\|\|', '||', text)  # Normalize ASCII markers
        
        return text.strip()
    
    def _normalize_english_text(self, text: str) -> str:
        """Normalize English translation text."""
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra punctuation
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r',+', ',', text)
        
        return text.strip()
    
    def _save_processed_corpus(self) -> None:
        """Save processed corpus in multiple formats."""
        # Save as JSONL for training
        jsonl_path = self.output_dir / "vedic_corpus.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for text in self.processed_texts:
                # Create training format
                training_example = {
                    'text': text.text,
                    'collection': text.collection,
                    'language': text.language,
                    'is_vedic': True,
                }
                
                # Add metadata
                if text.book is not None:
                    training_example['book'] = text.book
                if text.hymn is not None:
                    training_example['hymn'] = text.hymn
                if text.verse is not None:
                    training_example['verse'] = text.verse
                if text.translation:
                    training_example['translation'] = text.translation
                if text.commentary:
                    training_example['commentary'] = text.commentary
                if text.source:
                    training_example['source'] = text.source
                
                f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
        
        # Save as structured JSON
        json_path = self.output_dir / "vedic_corpus.json"
        corpus_data = {
            'metadata': {
                'total_texts': len(self.processed_texts),
                'collections': list(set(t.collection for t in self.processed_texts)),
                'languages': list(set(t.language for t in self.processed_texts)),
                'processing_config': {
                    'include_translations': self.include_translations,
                    'include_commentaries': self.include_commentaries,
                    'normalize_sanskrit': self.normalize_sanskrit,
                    'split_by_verses': self.split_by_verses,
                }
            },
            'texts': [
                {
                    'text': t.text,
                    'collection': t.collection,
                    'book': t.book,
                    'hymn': t.hymn,
                    'verse': t.verse,
                    'language': t.language,
                    'translation': t.translation,
                    'commentary': t.commentary,
                    'source': t.source,
                }
                for t in self.processed_texts
            ]
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)
        
        # Save plain text version for tokenizer training
        txt_path = self.output_dir / "vedic_corpus.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for text in self.processed_texts:
                f.write(text.text + '\n\n')
                if text.translation:
                    f.write(text.translation + '\n\n')
        
        logging.info(f"Saved processed corpus to {self.output_dir}")
        logging.info(f"  JSONL: {jsonl_path}")
        logging.info(f"  JSON: {json_path}")
        logging.info(f"  TXT: {txt_path}")
    
    def _generate_statistics(self) -> None:
        """Generate corpus statistics."""
        stats = {
            'total_texts': len(self.processed_texts),
            'collections': {},
            'languages': {},
            'avg_text_length': 0,
            'total_characters': 0,
        }
        
        total_chars = 0
        for text in self.processed_texts:
            # Collection stats
            if text.collection not in stats['collections']:
                stats['collections'][text.collection] = 0
            stats['collections'][text.collection] += 1
            
            # Language stats  
            if text.language not in stats['languages']:
                stats['languages'][text.language] = 0
            stats['languages'][text.language] += 1
            
            # Length stats
            text_length = len(text.text)
            total_chars += text_length
        
        stats['total_characters'] = total_chars
        stats['avg_text_length'] = total_chars / len(self.processed_texts) if self.processed_texts else 0
        
        # Save statistics
        stats_path = self.output_dir / "corpus_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logging.info("Corpus Statistics:")
        logging.info(f"  Total texts: {stats['total_texts']}")
        logging.info(f"  Total characters: {stats['total_characters']:,}")
        logging.info(f"  Average text length: {stats['avg_text_length']:.1f}")
        logging.info(f"  Collections: {stats['collections']}")
        logging.info(f"  Languages: {stats['languages']}")
    
    def get_collection_texts(self, collection: str) -> List[VedicText]:
        """Get all texts from a specific collection."""
        return [t for t in self.processed_texts if t.collection == collection]
    
    def get_texts_by_language(self, language: str) -> List[VedicText]:
        """Get all texts in a specific language."""
        return [t for t in self.processed_texts if t.language == language]
    
    def create_training_splits(
        self, 
        train_ratio: float = 0.8, 
        val_ratio: float = 0.1, 
        test_ratio: float = 0.1
    ) -> Dict[str, List[VedicText]]:
        """Create train/validation/test splits."""
        import random
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Shuffle texts
        texts = self.processed_texts.copy()
        random.shuffle(texts)
        
        n_train = int(len(texts) * train_ratio)
        n_val = int(len(texts) * val_ratio)
        
        splits = {
            'train': texts[:n_train],
            'validation': texts[n_train:n_train + n_val],
            'test': texts[n_train + n_val:],
        }
        
        # Save splits
        for split_name, split_texts in splits.items():
            split_path = self.output_dir / f"vedic_corpus_{split_name}.jsonl"
            with open(split_path, 'w', encoding='utf-8') as f:
                for text in split_texts:
                    training_example = {
                        'text': text.text,
                        'collection': text.collection,
                        'language': text.language,
                        'is_vedic': True,
                    }
                    f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
        
        logging.info(f"Created training splits:")
        logging.info(f"  Train: {len(splits['train'])} texts")
        logging.info(f"  Validation: {len(splits['validation'])} texts")
        logging.info(f"  Test: {len(splits['test'])} texts")
        
        return splits
