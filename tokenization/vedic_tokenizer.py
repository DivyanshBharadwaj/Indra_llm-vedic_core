"""
Vedic-specialized tokenizer for INDRA LLM with Sanskrit and philosophical concept support
(c) Divyansh Bharadwaj
"""

import re
import json
from typing import List, Dict, Optional, Tuple, Set
from .sentencepiece_tokenizer import SentencePieceTokenizer

class VedicTokenizer(SentencePieceTokenizer):
    """Specialized tokenizer for Vedic texts with Sanskrit and philosophical concept handling."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_size: int = 75000,  # Larger vocab for Sanskrit
        vedic_concepts_path: Optional[str] = None,
        sanskrit_rules_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Vedic tokenizer.
        
        Args:
            model_path: Path to trained SentencePiece model
            vocab_size: Vocabulary size (larger for Sanskrit support)
            vedic_concepts_path: Path to Vedic concepts dictionary
            sanskrit_rules_path: Path to Sanskrit tokenization rules
        """
        
        # Enhanced special tokens for Vedic content
        vedic_special_tokens = {
            # Base tokens
            "sep_token": "<sep>",
            "cls_token": "<cls>",
            "mask_token": "<mask>",
            
            # Language markers
            "sanskrit_start": "<sanskrit>",
            "sanskrit_end": "</sanskrit>",
            "hindi_start": "<hindi>",
            "hindi_end": "</hindi>",
            "english_start": "<english>", 
            "english_end": "</english>",
            
            # Vedic text markers
            "vedic_start": "<vedic>",
            "vedic_end": "</vedic>",
            "mantra_start": "<mantra>",
            "mantra_end": "</mantra>",
            "sloka_start": "<sloka>",
            "sloka_end": "</sloka>",
            
            # Philosophical concepts
            "dharma_token": "<dharma>",
            "karma_token": "<karma>",
            "ahimsa_token": "<ahimsa>",
            "satya_token": "<satya>",
            "moksha_token": "<moksha>",
            "brahman_token": "<brahman>",
            "atman_token": "<atman>",
            "yoga_token": "<yoga>",
            
            # Text structure
            "verse_start": "<verse>",
            "verse_end": "</verse>",
            "commentary_start": "<commentary>",
            "commentary_end": "</commentary>",
            "translation_start": "<translation>",
            "translation_end": "</translation>",
            
            # Vedic collections
            "rigveda_token": "<rigveda>",
            "yajurveda_token": "<yajurveda>",
            "samaveda_token": "<samaveda>",
            "atharvaveda_token": "<atharvaveda>",
            "upanishad_token": "<upanishad>",
            "brahmana_token": "<brahmana>",
        }
        
        super().__init__(
            model_path=model_path,
            vocab_size=vocab_size,
            special_tokens=vedic_special_tokens,
            **kwargs
        )
        
        # Load Vedic concepts and Sanskrit rules
        self.vedic_concepts = self._load_vedic_concepts(vedic_concepts_path)
        self.sanskrit_rules = self._load_sanskrit_rules(sanskrit_rules_path)
        
        # Sanskrit character patterns
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
        self.sanskrit_diacritics_pattern = re.compile(r'[āīūṛḷēōṃḥṅñṭḍṇṣśḥ]+')
        
        # Vedic concept patterns
        self.concept_patterns = self._build_concept_patterns()
    
    def _load_vedic_concepts(self, concepts_path: Optional[str]) -> Dict[str, List[str]]:
        """Load Vedic philosophical concepts dictionary."""
        default_concepts = {
            "dharma_related": [
                "dharma", "धर्म", "righteousness", "duty", "moral_law",
                "svadharma", "स्वधर्म", "rajadharma", "राजधर्म"
            ],
            "karma_related": [
                "karma", "कर्म", "action", "deed", "work", "ritual",
                "karman", "कर्मन्", "akarma", "अकर्म", "vikarma", "विकर्म"
            ],
            "moksha_related": [
                "moksha", "मोक्ष", "liberation", "release", "salvation",
                "mukti", "मुक्ति", "kaivalya", "कैवल्य"
            ],
            "brahman_related": [
                "brahman", "ब्रह्मन्", "absolute", "ultimate_reality",
                "parabrahman", "परब्रह्मन्", "saguna_brahman", "सगुण_ब्रह्मन्"
            ],
            "yoga_related": [
                "yoga", "योग", "union", "discipline", "practice",
                "raja_yoga", "राज_योग", "karma_yoga", "कर्म_योग",
                "bhakti_yoga", "भक्ति_योग", "jnana_yoga", "ज्ञान_योग"
            ]
        }
        
        if concepts_path and os.path.exists(concepts_path):
            try:
                with open(concepts_path, 'r', encoding='utf-8') as f:
                    loaded_concepts = json.load(f)
                    # Merge with defaults
                    for key, values in loaded_concepts.items():
                        if key in default_concepts:
                            default_concepts[key].extend(values)
                        else:
                            default_concepts[key] = values
            except Exception as e:
                logging.warning(f"Could not load Vedic concepts from {concepts_path}: {e}")
        
        return default_concepts
    
    def _load_sanskrit_rules(self, rules_path: Optional[str]) -> Dict[str, str]:
        """Load Sanskrit tokenization rules."""
        default_rules = {
            # Sandhi rules (basic patterns)
            "vowel_sandhi": r'([aāiīuūṛḷeēoō])\s+([aāiīuūṛḷeēoō])',
            "consonant_sandhi": r'([kgṅcjñṭḍṇtdnpbmyrlvśṣshḥ])\s+([kgṅcjñṭḍṇtdnpbmyrlvśṣsh])',
            "visarga_sandhi": r'([aāiīuūeēoō])ḥ\s+([kgṅcjñṭḍṇtdnpbmyrlvśṣsh])',
            
            # Common prefixes and suffixes
            "prefixes": r'^(pra|vi|sam|upa|ni|abhi|anu|pari|su|dur|a|an)',
            "suffixes": r'(ti|tā|tva|tvam|ya|ana|man|tra)$',
            
            # Devanagari normalization
            "normalize_anusvara": r'[ंॅ]',
            "normalize_visarga": r'[ःḥ]',
        }
        
        if rules_path and os.path.exists(rules_path):
            try:
                with open(rules_path, 'r', encoding='utf-8') as f:
                    loaded_rules = json.load(f)
                    default_rules.update(loaded_rules)
            except Exception as e:
                logging.warning(f"Could not load Sanskrit rules from {rules_path}: {e}")
        
        return default_rules
    
    def _build_concept_patterns(self) -> Dict[str, re.Pattern]:
        """Build regex patterns for Vedic concepts."""
        patterns = {}
        
        for concept_category, concepts in self.vedic_concepts.items():
            # Create pattern for each concept category
            escaped_concepts = [re.escape(concept) for concept in concepts]
            pattern_str = r'\b(?:' + '|'.join(escaped_concepts) + r')\b'
            patterns[concept_category] = re.compile(pattern_str, re.IGNORECASE | re.UNICODE)
        
        return patterns
    
    def preprocess_vedic_text(self, text: str) -> str:
        """
        Preprocess Vedic text with Sanskrit normalization and concept detection.
        
        Args:
            text: Input text (potentially mixed Sanskrit/Hindi/English)
            
        Returns:
            Preprocessed text with appropriate markers
        """
        # Normalize Sanskrit diacritics
        text = self._normalize_sanskrit(text)
        
        # Detect and mark languages
        text = self._mark_languages(text)
        
        # Mark Vedic concepts
        text = self._mark_vedic_concepts(text)
        
        # Handle Sanskrit compound words
        text = self._handle_sanskrit_compounds(text)
        
        return text
    
    def _normalize_sanskrit(self, text: str) -> str:
        """Normalize Sanskrit text for consistent tokenization."""
        # Normalize anusvara and visarga
        if "normalize_anusvara" in self.sanskrit_rules:
            pattern = self.sanskrit_rules["normalize_anusvara"]
            text = re.sub(pattern, 'ं', text)
        
        if "normalize_visarga" in self.sanskrit_rules:
            pattern = self.sanskrit_rules["normalize_visarga"]
            text = re.sub(pattern, 'ḥ', text)
        
        # Handle IAST to Devanagari conversion markers
        # This is a simplified version - full implementation would need comprehensive mapping
        # Complete IAST to Devanagari mapping
        iast_to_dev_basic = {
            # Vowels - Independent forms
            'a': 'अ', 'ā': 'आ', 'i': 'इ', 'ī': 'ई', 'u': 'उ', 'ū': 'ऊ',
            'ṛ': 'ऋ', 'ṝ': 'ॠ', 'ḷ': 'ऌ', 'ḹ': 'ॡ',
            'e': 'ए', 'ē': 'ए', 'o': 'ओ', 'ō': 'ओ',
            'ai': 'ऐ', 'au': 'औ',

            # Vowel marks/diacritics (मात्रा)
            'ā': 'ा', 'i': 'ि', 'ī': 'ी', 'u': 'ु', 'ū': 'ू',
            'ṛ': 'ृ', 'ṝ': 'ॄ', 'ḷ': 'ॢ', 'ḹ': 'ॣ',
            'e': 'े', 'o': 'ो', 'ai': 'ै', 'au': 'ौ',
            
            # Consonants - Gutturals (कण्ठ्य)
            'k': 'क', 'kh': 'ख', 'g': 'ग', 'gh': 'घ', 'ṅ': 'ङ',
            
            # Consonants - Palatals (तालव्य)
            'c': 'च', 'ch': 'छ', 'j': 'ज', 'jh': 'झ', 'ñ': 'ञ',
            
            # Consonants - Retroflexes (मूर्धन्य)
            'ṭ': 'ट', 'ṭh': 'ठ', 'ḍ': 'ड', 'ḍh': 'ढ', 'ṇ': 'ण',
            
            # Consonants - Dentals (दन्त्य)
            't': 'त', 'th': 'थ', 'd': 'द', 'dh': 'ध', 'n': 'न',
            
            # Consonants - Labials (ओष्ठ्य)
            'p': 'प', 'ph': 'फ', 'b': 'ब', 'bh': 'भ', 'm': 'म',
            
            # Semi-vowels (अन्तःस्थ)
            'y': 'य', 'r': 'र', 'l': 'ल', 'v': 'व', 'w': 'व',
            
            # Sibilants (ऊष्म)
            'ś': 'श', 'ṣ': 'ष', 's': 'स',
            
            # Aspirate
            'h': 'ह',
            
            # Special characters
            'ṃ': 'ं',     # Anusvara
            'ḥ': 'ः',     # Visarga
            'ṁ': 'ँ',     # Candrabindu
            '~': 'ँ',     # Candrabindu (alternative)
            'ẋ': 'ᳵ',     # Jihvamuliya (rare)
            'ẍ': 'ᳶ',     # Upadhmaniya (rare)
            '\'': 'ऽ',    # Avagraha
            
            # Om variations
            'oṃ': 'ॐ', 'oṁ': 'ॐ', 'auṃ': 'ॐ', 'auṁ': 'ॐ',
            
            # Numbers
            '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
            '5': '५', '6': '६', '7': '७', '8': '८', '9': '९',
            
            # Punctuation
            '.': '।',     # Danda
            '..': '॥',    # Double danda
            '|': '।',     # Alternative danda notation
            '||': '॥',    # Alternative double danda
            
            # Additional vowel variants and combinations
            'ê': 'ए',     # Alternative e notation
            'ô': 'ओ',     # Alternative o notation
            'æ': 'ऐ',     # Alternative ai notation
            'ö': 'औ',     # Alternative au notation
            
            # Rare/Vedic characters
            'ḫ': 'ᳵ',     # Alternative jihvamuliya
            'ḵ': 'ᳶ',     # Alternative upadhmaniya
            'f': 'फ़',     # For foreign words
            'z': 'ज़',     # For foreign words
            'ġ': 'ग़',     # For foreign words
            'q': 'क़',     # For foreign words
            'x': 'क्ष',    # Ksha compound
            
            # Compound consonants (common combinations)
            'kṣ': 'क्ष',   # Ksha
            'jñ': 'ज्ञ',   # Gya
            'tr': 'त्र',   # Tra
            'śr': 'श्र',   # Shra
        }
        
        for iast, dev in iast_to_dev_basic.items():
            text = text.replace(iast, dev)
        
        return text
    
    def _mark_languages(self, text: str) -> str:
        """Detect and mark different languages in the text."""
        # Simple language detection based on character sets
        parts = []
        current_lang = None
        current_text = ""
        
        for char in text:
            if '\u0900' <= char <= '\u097F':  # Devanagari
                if current_lang != 'sanskrit':
                    if current_text.strip():
                        parts.append(self._wrap_language(current_text, current_lang))
                    current_text = char
                    current_lang = 'sanskrit'
                else:
                    current_text += char
            elif char.isascii() and char.isalpha():  # ASCII letters
                if current_lang != 'english':
                    if current_text.strip():
                        parts.append(self._wrap_language(current_text, current_lang))
                    current_text = char
                    current_lang = 'english'
                else:
                    current_text += char
            else:
                current_text += char
        
        # Handle remaining text
        if current_text.strip():
            parts.append(self._wrap_language(current_text, current_lang))
        
        return ''.join(parts)
    
    def _wrap_language(self, text: str, lang: Optional[str]) -> str:
        """Wrap text with appropriate language markers."""
        if not text.strip():
            return text
        
        if lang == 'sanskrit':
            return f"{self.special_tokens['sanskrit_start']}{text}{self.special_tokens['sanskrit_end']}"
        elif lang == 'english':
            return f"{self.special_tokens['english_start']}{text}{self.special_tokens['english_end']}"
        else:
            return text
    
    def _mark_vedic_concepts(self, text: str) -> str:
        """Mark identified Vedic philosophical concepts."""
        for concept_category, pattern in self.concept_patterns.items():
            # Find all matches
            matches = list(pattern.finditer(text))
            
            # Replace matches with marked versions (from end to preserve indices)
            for match in reversed(matches):
                start, end = match.span()
                concept = match.group()
                
                # Get the appropriate marker token
                if concept_category == 'dharma_related':
                    marker = self.special_tokens['dharma_token']
                elif concept_category == 'karma_related':
                    marker = self.special_tokens['karma_token']
                elif concept_category == 'moksha_related':
                    marker = self.special_tokens['moksha_token']
                else:
                    marker = f"<{concept_category}>"
                
                marked_concept = f"{marker}{concept}"
                text = text[:start] + marked_concept + text[end:]
        
        return text
    
    def _handle_sanskrit_compounds(self, text: str) -> str:
        """Handle Sanskrit compound words with sandhi rules."""
        # Complete implementation of Sanskrit compound handling with extensive linguistic rules
        
        # VOWEL SANDHI RULES
        if "vowel_sandhi" in self.sanskrit_rules:
            pattern = self.sanskrit_rules["vowel_sandhi"]
            
            # Similar vowel combinations (Savarṇa)
            # Short + Short = Long
            text = re.sub(r'a\s+a\b', 'ā', text)
            text = re.sub(r'i\s+i\b', 'ī', text)
            text = re.sub(r'u\s+u\b', 'ū', text)
            text = re.sub(r'ṛ\s+ṛ\b', 'ṝ', text)
            text = re.sub(r'ḷ\s+ḷ\b', 'ḹ', text)
            
            # Short + Long = Long, Long + Short = Long, Long + Long = Long
            text = re.sub(r'a\s+ā\b', 'ā', text)
            text = re.sub(r'ā\s+a\b', 'ā', text)
            text = re.sub(r'ā\s+ā\b', 'ā', text)
            text = re.sub(r'i\s+ī\b', 'ī', text)
            text = re.sub(r'ī\s+i\b', 'ī', text)
            text = re.sub(r'ī\s+ī\b', 'ī', text)
            text = re.sub(r'u\s+ū\b', 'ū', text)
            text = re.sub(r'ū\s+u\b', 'ū', text)
            text = re.sub(r'ū\s+ū\b', 'ū', text)
            text = re.sub(r'ṛ\s+ṝ\b', 'ṝ', text)
            text = re.sub(r'ṝ\s+ṛ\b', 'ṝ', text)
            text = re.sub(r'ṝ\s+ṝ\b', 'ṝ', text)
            
            # Guna combinations (a/ā + dissimilar simple vowels)
            text = re.sub(r'a\s+i\b', 'e', text)
            text = re.sub(r'a\s+ī\b', 'e', text)
            text = re.sub(r'ā\s+i\b', 'e', text)
            text = re.sub(r'ā\s+ī\b', 'e', text)
            text = re.sub(r'a\s+u\b', 'o', text)
            text = re.sub(r'a\s+ū\b', 'o', text)
            text = re.sub(r'ā\s+u\b', 'o', text)
            text = re.sub(r'ā\s+ū\b', 'o', text)
            text = re.sub(r'a\s+ṛ\b', 'ar', text)
            text = re.sub(r'a\s+ṝ\b', 'ar', text)
            text = re.sub(r'ā\s+ṛ\b', 'ar', text)
            text = re.sub(r'ā\s+ṝ\b', 'ar', text)
            text = re.sub(r'a\s+ḷ\b', 'al', text)
            text = re.sub(r'ā\s+ḷ\b', 'al', text)
            text = re.sub(r'a\s+ḹ\b', 'al', text)
            text = re.sub(r'ā\s+ḹ\b', 'al', text)
            
            # Vrddhi combinations (a/ā + guna vowels)
            text = re.sub(r'a\s+e\b', 'ai', text)
            text = re.sub(r'ā\s+e\b', 'ai', text)
            text = re.sub(r'a\s+o\b', 'au', text)
            text = re.sub(r'ā\s+o\b', 'au', text)
            text = re.sub(r'a\s+ai\b', 'ai', text)
            text = re.sub(r'ā\s+ai\b', 'ai', text)
            text = re.sub(r'a\s+au\b', 'au', text)
            text = re.sub(r'ā\s+au\b', 'au', text)
            
            # Semi-vowel formation before vowels
            # i/ī + vowel = y + vowel
            text = re.sub(r'i\s+([aāuūṛṝḷḹeoaiauḥṃ])', r'y\1', text)
            text = re.sub(r'ī\s+([aāuūṛṝḷḹeoaiauḥṃ])', r'y\1', text)
            
            # u/ū + vowel = v + vowel
            text = re.sub(r'u\s+([aāiīṛṝḷḹeoaiauḥṃ])', r'v\1', text)
            text = re.sub(r'ū\s+([aāiīṛṝḷḹeoaiauḥṃ])', r'v\1', text)
            
            # ṛ/ṝ + vowel = r + vowel
            text = re.sub(r'ṛ\s+([aāiīuūḷḹeoaiauḥṃ])', r'r\1', text)
            text = re.sub(r'ṝ\s+([aāiīuūḷḹeoaiauḥṃ])', r'r\1', text)
            
            # ḷ + vowel = l + vowel
            text = re.sub(r'ḷ\s+([aāiīuūṛṝeoaiauḥṃ])', r'l\1', text)
            text = re.sub(r'ḹ\s+([aāiīuūṛṝeoaiauḥṃ])', r'l\1', text)
            
            # e/ai + vowel = ay + vowel
            text = re.sub(r'e\s+([aāiīuūṛṝḷḹoaiauḥṃ])', r'ay\1', text)
            text = re.sub(r'ai\s+([aāiīuūṛṝḷḹoauḥṃ])', r'āy\1', text)
            
            # o/au + vowel = av + vowel
            text = re.sub(r'o\s+([aāiīuūṛṝḷḹeaiauḥṃ])', r'av\1', text)
            text = re.sub(r'au\s+([aāiīuūṛṝḷḹeaiḥṃ])', r'āv\1', text)
        
        # CONSONANT SANDHI RULES
        if "consonant_sandhi" in self.sanskrit_rules:
            # Final consonant modifications
            # Final 't' changes before different consonant groups
            text = re.sub(r't\s+([kKgGṅ])', r'k\1', text)      # Before gutturals → k
            text = re.sub(r't\s+([cCjJñ])', r'c\1', text)      # Before palatals → c
            text = re.sub(r't\s+([ṭṬḍḌṇ])', r'ṭ\1', text)     # Before cerebrals → ṭ
            text = re.sub(r't\s+([pPbBm])', r'p\1', text)      # Before labials → p
            text = re.sub(r't\s+([ś])', r'c\1', text)          # Before ś → c
            text = re.sub(r't\s+([ṣ])', r'ṭ\1', text)          # Before ṣ → ṭ
            text = re.sub(r't\s+([s])', r't\1', text)          # Before s → t
            text = re.sub(r't\s+([h])', r'd\1', text)          # Before h → d
            text = re.sub(r't\s+([rlv])', r'd\1', text)        # Before liquids → d
            
            # Final 'd' changes to 't' before voiceless consonants
            text = re.sub(r'd\s+([kKcCṭṬtTpPśṣsh])', r't\1', text)
            
            # Final 'n' changes
            # Before gutturals → ṅ
            text = re.sub(r'n\s+([kKgGṅ])', r'ṅ\1', text)
            # Before palatals → ñ  
            text = re.sub(r'n\s+([cCjJñśy])', r'ñ\1', text)
            # Before cerebrals → ṇ
            text = re.sub(r'n\s+([ṭṬḍḌṇṣr])', r'ṇ\1', text)
            # Before labials → m
            text = re.sub(r'n\s+([pPbBmvf])', r'm\1', text)
            # Before other consonants → anusvara
            text = re.sub(r'n\s+([kKgGcCjJṭṬḍḌtTdDpPbBśṣs])', r'ṃ\1', text)
            
            # Final 'm' → anusvara before consonants
            text = re.sub(r'm\s+([kKgGṅcCjJñṭṬḍḌṇtTdDnpPbBmśṣshy])', r'ṃ\1', text)
            
            # Sibilant changes
            # Final 's' changes
            text = re.sub(r's\s+([kKgGṅpPbBm])', r'ḥ\1', text)   # → visarga before gutturals/labials
            text = re.sub(r's\s+([cCjJñśy])', r'ś\1', text)      # → ś before palatals
            text = re.sub(r's\s+([ṭṬḍḌṇṣr])', r'ṣ\1', text)     # → ṣ before cerebrals
            text = re.sub(r's\s+([tTdDnl])', r's\1', text)       # unchanged before dentals
            text = re.sub(r's\s+([h])', r'ṣ\1', text)            # → ṣ before h
            
            # Doubled consonants at word boundaries
            text = re.sub(r'([kgcjṭḍtdpb])\s+\1', r'\1\1\1', text)
            
            # Aspirated consonant simplification
            text = re.sub(r'([kgcjṭḍtdpb])h\s+([kgcjṭḍtdpb])', r'\1\2\2', text)
            
            # Special retroflex rules
            # Cerebral n (ṇ) retroflexion after r, ṛ, ṝ, ṣ
            text = re.sub(r'([rṛṝṣ][aāiīuūeoḥṃkKgGṅcCjJñṭṬḍḌtTdDpPbBmyrlvśs]*?)\s*n([aāiīuūeoḥṃ])', r'\1ṇ\2', text)
            
            # Anusvara assimilation
            text = re.sub(r'ṃ\s+([kKgGṅ])', r'ṅ\1', text)      # Before gutturals
            text = re.sub(r'ṃ\s+([cCjJñ])', r'ñ\1', text)      # Before palatals
            text = re.sub(r'ṃ\s+([ṭṬḍḌṇ])', r'ṇ\1', text)     # Before cerebrals
            text = re.sub(r'ṃ\s+([tTdDn])', r'n\1', text)      # Before dentals
            text = re.sub(r'ṃ\s+([pPbBm])', r'm\1', text)      # Before labials
        
        # VISARGA SANDHI RULES
        if "visarga_sandhi" in self.sanskrit_rules:
            # aḥ/āḥ + voiced consonants/vowels → r
            text = re.sub(r'([aā])ḥ\s+([aāiīuūṛṝḷḹeoaiaugGjJḍḌdDbBnñṇmrlvhy])', r'\1r\2', text)
            
            # aḥ + voiceless stops
            text = re.sub(r'aḥ\s+([kK])', r'aḥ \1', text)       # Unchanged before k, kh
            text = re.sub(r'aḥ\s+([cC])', r'aś \1', text)       # → ś before c, ch
            text = re.sub(r'aḥ\s+([ṭṬ])', r'aṣ \1', text)       # → ṣ before ṭ, ṭh
            text = re.sub(r'aḥ\s+([tT])', r'as \1', text)       # → s before t, th
            text = re.sub(r'aḥ\s+([pP])', r'aḥ \1', text)       # Unchanged before p, ph
            
            # āḥ follows same pattern as aḥ
            text = re.sub(r'āḥ\s+([kK])', r'āḥ \1', text)
            text = re.sub(r'āḥ\s+([cC])', r'āś \1', text)
            text = re.sub(r'āḥ\s+([ṭṬ])', r'āṣ \1', text)
            text = re.sub(r'āḥ\s+([tT])', r'ās \1', text)
            text = re.sub(r'āḥ\s+([pP])', r'āḥ \1', text)
            
            # iḥ/īḥ/uḥ/ūḥ/eḥ/oḥ/ṛḥ/ṝḥ + voiceless consonants → ṣ
            text = re.sub(r'([iīuūeṛṝoaiau])ḥ\s+([kKcCṭṬtTpP])', r'\1ṣ \2', text)
            
            # Final ḥ before vowels → r (except after a/ā)
            text = re.sub(r'([iīuūeṛṝoaiauḷḹ])ḥ\s+([aāiīuūṛṝḷḹeoau])', r'\1r \2', text)
            
            # Special visarga rules before sibilants
            text = re.sub(r'ḥ\s+([śṣs])', r'ḥ \1', text)        # Unchanged before sibilants
        
        # SPECIAL AND IRREGULAR RULES
        if "special_rules" in self.sanskrit_rules:
            # Common irregular compounds
            irregular_forms = {
                r'pra\s+ūḍha': 'prauḍha',
                r'pra\s+ugga': 'progga',
                r'su\s+ukta': 'sukta',
                r'su\s+āgata': 'svāgata',
                r'sam\s+ūha': 'samūha',
                r'sam\s+ṛddhi': 'samṛddhi',
                r'upa\s+iṣṭa': 'upeṣṭa',
                r'apa\s+ūpa': 'apopa',
                r'deva\s+indra': 'devendra',
                r'mahā\s+īśa': 'maheśa',
                r'mahā\s+īśvara': 'maheśvara',
                r'mahā\s+indra': 'mahendra',
                r'mahā\s+ṛṣi': 'maharṣi',
                r'mahā\s+uttama': 'mahottama',
                r'mahā\s+udaya': 'mahodaya',
                r'mahā\s+ātman': 'mahātman'
            }
            
            for pattern, replacement in irregular_forms.items():
                text = re.sub(pattern, replacement, text)
            
            # Handle prefix + root combinations with special vowel changes
            # Prefix ending in 'a' + root beginning with 'ṛ'
            text = re.sub(r'([^aā])a\s+ṛ([kgṅcjñṭḍṇtdnpbmyrlvśṣsh])', r'\1ār\2', text)
            
            # Reduplication sandhi
            text = re.sub(r'([kgcjṭḍtdpb])([aāiīuūṛṝḷḹeoau])\s+\1\2', r'\1\2\1\2', text)
            
            # Final cleanup - remove extra spaces
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        return text
    
    def encode_vedic_text(
        self,
        text: str,
        add_vedic_markers: bool = True,
        preserve_structure: bool = True,
        **kwargs
    ) -> List[int]:
        """
        Encode Vedic text with specialized preprocessing.
        
        Args:
            text: Input Vedic text
            add_vedic_markers: Whether to add Vedic content markers
            preserve_structure: Whether to preserve verse/sloka structure
            **kwargs: Additional arguments for base encode method
            
        Returns:
            List of token IDs
        """
        # Preprocess text
        if add_vedic_markers or preserve_structure:
            text = self.preprocess_vedic_text(text)
        
        # Add Vedic content markers
        if add_vedic_markers:
            text = f"{self.special_tokens['vedic_start']}{text}{self.special_tokens['vedic_end']}"
        
        # Use base tokenizer encoding
        return self.encode(text, **kwargs)
    
    def get_vedic_concept_ids(self) -> Dict[str, List[int]]:
        """Get token IDs for Vedic concepts."""
        concept_ids = {}
        
        for concept_category, concepts in self.vedic_concepts.items():
            category_ids = []
            for concept in concepts:
                # Try to get exact token ID
                if concept in self.token_to_id:
                    category_ids.append(self.token_to_id[concept])
                else:
                    # Encode as subwords
                    subword_ids = self.encode(concept, add_special_tokens=False)
                    category_ids.extend(subword_ids)
            
            concept_ids[concept_category] = list(set(category_ids))  # Remove duplicates
        
        return concept_ids
    
    def identify_vedic_concepts_in_sequence(self, token_ids: List[int]) -> Dict[str, List[int]]:
        """
        Identify Vedic concepts in a token sequence.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Dictionary mapping concept categories to positions in sequence
        """
        concept_ids = self.get_vedic_concept_ids()
        found_concepts = {category: [] for category in concept_ids}
        
        for i, token_id in enumerate(token_ids):
            for category, ids in concept_ids.items():
                if token_id in ids:
                    found_concepts[category].append(i)
        
        return found_concepts
    
    def create_vedic_attention_mask(
        self,
        token_ids: List[int],
        enhance_concepts: bool = True,
        enhance_sanskrit: bool = True
    ) -> List[float]:
        """
        Create attention mask that emphasizes Vedic concepts and Sanskrit text.
        
        Args:
            token_ids: List of token IDs
            enhance_concepts: Whether to enhance attention on Vedic concepts
            enhance_sanskrit: Whether to enhance attention on Sanskrit text
            
        Returns:
            List of attention weights (1.0 = normal, >1.0 = enhanced)
        """
        attention_weights = [1.0] * len(token_ids)
        
        if enhance_concepts:
            # Enhance attention on Vedic concepts
            concept_positions = self.identify_vedic_concepts_in_sequence(token_ids)
            for category, positions in concept_positions.items():
                for pos in positions:
                    attention_weights[pos] = 1.5  # Enhanced attention
        
        if enhance_sanskrit:
            # Enhance attention on Sanskrit markers and content
            sanskrit_start_id = self.token_to_id.get(self.special_tokens['sanskrit_start'])
            sanskrit_end_id = self.token_to_id.get(self.special_tokens['sanskrit_end'])
            
            in_sanskrit = False
            for i, token_id in enumerate(token_ids):
                if token_id == sanskrit_start_id:
                    in_sanskrit = True
                    attention_weights[i] = 1.3
                elif token_id == sanskrit_end_id:
                    in_sanskrit = False
                    attention_weights[i] = 1.3
                elif in_sanskrit:
                    attention_weights[i] = 1.2
        
        return attention_weights
