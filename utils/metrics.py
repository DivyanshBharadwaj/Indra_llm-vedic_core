"""
Metrics calculation utilities for INDRA LLM
(c) Divyansh Bharadwaj
"""

import math
import time
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

class MetricType(Enum):
    """Types of metrics."""
    LOSS = "loss"
    ACCURACY = "accuracy"
    PERPLEXITY = "perplexity"
    BLEU = "bleu"
    ROUGE = "rouge"
    VEDIC_ALIGNMENT = "vedic_alignment"
    TOKENS_PER_SECOND = "tokens_per_second"

@dataclass
class MetricResult:
    """Result of a metric calculation."""
    name: str
    value: float
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None

class TrainingMetrics:
    """Comprehensive training metrics tracker."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize training metrics tracker.
        
        Args:
            window_size: Size of rolling window for metrics
        """
        self.window_size = window_size
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.step_counter = 0
        self.start_time = time.time()
        
        # Special tracking for loss components
        self.loss_components = [
            'total_loss', 'lm_loss', 'moe_aux_loss', 'vedic_alignment_loss',
            'distillation_loss', 'policy_loss', 'value_loss'
        ]
        
    def update(self, metrics: Dict[str, Union[float, Tensor]], step: Optional[int] = None):
        """
        Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step (optional)
        """
        if step is None:
            step = self.step_counter
        
        current_time = time.time()
        
        for name, value in metrics.items():
            if isinstance(value, Tensor):
                value = value.item()
            elif isinstance(value, (list, tuple)):
                value = float(np.mean(value))
            
            self.metrics_history[name].append(MetricResult(
                name=name,
                value=value,
                timestamp=current_time,
                metadata={'step': step}
            ))
        
        self.step_counter = max(self.step_counter, step + 1)
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric."""
        if metric_name not in self.metrics_history or not self.metrics_history[metric_name]:
            return None
        return self.metrics_history[metric_name][-1].value
    
    def get_average(self, metric_name: str, steps: Optional[int] = None) -> Optional[float]:
        """Get rolling average for a metric."""
        if metric_name not in self.metrics_history or not self.metrics_history[metric_name]:
            return None
        
        values = [m.value for m in self.metrics_history[metric_name]]
        if steps is not None:
            values = values[-steps:]
        
        return float(np.mean(values)) if values else None
    
    def get_trend(self, metric_name: str, steps: int = 10) -> str:
        """Get trend direction for a metric."""
        if metric_name not in self.metrics_history or len(self.metrics_history[metric_name]) < steps:
            return "insufficient_data"
        
        values = [m.value for m in self.metrics_history[metric_name]][-steps:]
        
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive summary of all metrics."""
        summary = {}
        
        for metric_name in self.metrics_history:
            if not self.metrics_history[metric_name]:
                continue
            
            values = [m.value for m in self.metrics_history[metric_name]]
            
            summary[metric_name] = {
                'latest': values[-1],
                'average': float(np.mean(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'std': float(np.std(values)),
                'count': len(values),
                'trend': self.get_trend(metric_name)
            }
        
        return summary
    
    def export_to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Export all metrics history to dictionary."""
        export_data = {}
        
        for metric_name, history in self.metrics_history.items():
            export_data[metric_name] = [
                {
                    'value': result.value,
                    'timestamp': result.timestamp,
                    'metadata': result.metadata or {}
                }
                for result in history
            ]
        
        return export_data

class MetricsCalculator:
    """Advanced metrics calculation for LLM evaluation."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.cache = {}
    
    def calculate_perplexity(self, logits: Tensor, targets: Tensor, ignore_index: int = -100) -> float:
        """
        Calculate perplexity from logits and targets.
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            ignore_index: Index to ignore in loss calculation
            
        Returns:
            Perplexity value
        """
        # Flatten for loss calculation
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.view(-1)
        
        # Calculate cross entropy loss
        loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index, reduction='mean')
        
        # Convert to perplexity
        perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def calculate_accuracy(
        self, 
        logits: Tensor, 
        targets: Tensor, 
        ignore_index: int = -100,
        top_k: int = 1
    ) -> float:
        """
        Calculate token-level accuracy.
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            ignore_index: Index to ignore in calculation
            top_k: Calculate top-k accuracy
            
        Returns:
            Accuracy value
        """
        with torch.no_grad():
            # Get predictions
            if top_k == 1:
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == targets)
            else:
                _, top_k_predictions = torch.topk(logits, top_k, dim=-1)
                correct = torch.any(top_k_predictions == targets.unsqueeze(-1), dim=-1)
            
            # Mask ignored tokens
            mask = (targets != ignore_index)
            correct = correct & mask
            
            # Calculate accuracy
            accuracy = correct.sum().float() / mask.sum().float()
            
            return accuracy.item()
    
    def calculate_bleu_score(
        self, 
        predictions: List[str], 
        references: List[List[str]],
        max_n: int = 4
    ) -> float:
        """
        Calculate BLEU score for text generation.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts (can be multiple per prediction)
            max_n: Maximum n-gram order
            
        Returns:
            BLEU score
        """
        try:
            from nltk.translate.bleu_score import corpus_bleu
            from nltk.tokenize import word_tokenize
            
            # Tokenize predictions and references
            tokenized_predictions = [word_tokenize(pred.lower()) for pred in predictions]
            tokenized_references = [
                [word_tokenize(ref.lower()) for ref in ref_list]
                for ref_list in references
            ]
            
            # Calculate BLEU score
            bleu_score = corpus_bleu(tokenized_references, tokenized_predictions)
            
            return bleu_score
            
        except ImportError:
            logging.warning("NLTK not available for BLEU calculation")
            return 0.0
        except Exception as e:
            logging.warning(f"Error calculating BLEU: {e}")
            return 0.0
    
    def calculate_rouge_score(
        self, 
        predictions: List[str], 
        references: List[str],
        rouge_types: List[str] = ['rouge1', 'rouge2', 'rougeL']
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores for text generation.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            rouge_types: Types of ROUGE to calculate
            
        Returns:
            Dictionary of ROUGE scores
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
            
            scores = {rouge_type: [] for rouge_type in rouge_types}
            
            for pred, ref in zip(predictions, references):
                score = scorer.score(ref, pred)
                for rouge_type in rouge_types:
                    scores[rouge_type].append(score[rouge_type].fmeasure)
            
            # Average scores
            avg_scores = {
                rouge_type: float(np.mean(scores[rouge_type]))
                for rouge_type in rouge_types
            }
            
            return avg_scores
            
        except ImportError:
            logging.warning("rouge-score not available for ROUGE calculation")
            return {rouge_type: 0.0 for rouge_type in rouge_types}
        except Exception as e:
            logging.warning(f"Error calculating ROUGE: {e}")
            return {rouge_type: 0.0 for rouge_type in rouge_types}
    
    def calculate_vedic_alignment_score(
        self, 
        text: str, 
        vedic_concepts: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Calculate Vedic alignment score for generated text.
        
        Args:
            text: Generated text to analyze
            vedic_concepts: Dictionary of Vedic concepts
            
        Returns:
            Dictionary of alignment scores
        """
        text_lower = text.lower()
        scores = {}
        
        # Calculate scores for each concept category
        for category, concepts in vedic_concepts.items():
            concept_matches = sum(1 for concept in concepts if concept.lower() in text_lower)
            scores[category] = min(1.0, concept_matches / max(len(concepts), 1))
        
        # Overall alignment score
        scores['overall_alignment'] = sum(scores.values()) / max(len(scores), 1)
        
        return scores
    
    def calculate_diversity_metrics(self, texts: List[str]) -> Dict[str, float]:
        """
        Calculate diversity metrics for generated texts.
        
        Args:
            texts: List of generated texts
            
        Returns:
            Dictionary of diversity metrics
        """
        if not texts:
            return {'distinct_1': 0.0, 'distinct_2': 0.0, 'self_bleu': 0.0}
        
        # Tokenize texts
        try:
            from nltk.tokenize import word_tokenize
            tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        except ImportError:
            # Simple whitespace tokenization fallback
            tokenized_texts = [text.lower().split() for text in texts]
        
        # Calculate distinct n-grams
        all_unigrams = set()
        all_bigrams = set()
        total_unigrams = 0
        total_bigrams = 0
        
        for tokens in tokenized_texts:
            # Unigrams
            unigrams = set(tokens)
            all_unigrams.update(unigrams)
            total_unigrams += len(tokens)
            
            # Bigrams
            bigrams = set(zip(tokens[:-1], tokens[1:]))
            all_bigrams.update(bigrams)
            total_bigrams += len(tokens) - 1
        
        # Distinct ratios
        distinct_1 = len(all_unigrams) / max(total_unigrams, 1)
        distinct_2 = len(all_bigrams) / max(total_bigrams, 1)
        
        # Self-BLEU (how similar are the texts to each other)
        self_bleu = 0.0
        if len(texts) > 1:
            self_bleu_scores = []
            for i, text in enumerate(texts):
                other_texts = texts[:i] + texts[i+1:]
                if other_texts:
                    bleu = self.calculate_bleu_score([text], [other_texts])
                    self_bleu_scores.append(bleu)
            
            self_bleu = float(np.mean(self_bleu_scores)) if self_bleu_scores else 0.0
        
        return {
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'self_bleu': self_bleu,
            'vocab_size': len(all_unigrams)
        }
    
    def calculate_inference_speed(
        self, 
        num_tokens: int, 
        time_elapsed: float,
        batch_size: int = 1
    ) -> Dict[str, float]:
        """
        Calculate inference speed metrics.
        
        Args:
            num_tokens: Number of tokens generated
            time_elapsed: Time taken in seconds
            batch_size: Batch size used
            
        Returns:
            Dictionary of speed metrics
        """
        if time_elapsed <= 0:
            return {'tokens_per_second': 0.0, 'seconds_per_token': float('inf')}
        
        tokens_per_second = num_tokens / time_elapsed
        seconds_per_token = time_elapsed / num_tokens
        throughput = (num_tokens * batch_size) / time_elapsed
        
        return {
            'tokens_per_second': tokens_per_second,
            'seconds_per_token': seconds_per_token,
            'throughput': throughput,
            'latency_ms': (time_elapsed / num_tokens) * 1000
        }
    
    def calculate_memory_efficiency(
        self, 
        model_size_mb: float,
        peak_memory_mb: float,
        sequence_length: int,
        batch_size: int
    ) -> Dict[str, float]:
        """
        Calculate memory efficiency metrics.
        
        Args:
            model_size_mb: Model size in MB
            peak_memory_mb: Peak memory usage in MB
            sequence_length: Input sequence length
            batch_size: Batch size
            
        Returns:
            Dictionary of memory metrics
        """
        memory_per_token = peak_memory_mb / (sequence_length * batch_size)
        memory_overhead = peak_memory_mb - model_size_mb
        overhead_ratio = memory_overhead / model_size_mb if model_size_mb > 0 else 0.0
        
        return {
            'model_size_mb': model_size_mb,
            'peak_memory_mb': peak_memory_mb,
            'memory_per_token_mb': memory_per_token,
            'memory_overhead_mb': memory_overhead,
            'overhead_ratio': overhead_ratio,
            'memory_efficiency': model_size_mb / peak_memory_mb if peak_memory_mb > 0 else 0.0
        }

class VedicMetrics:
    """Specialized metrics for Vedic alignment evaluation."""
    
    def __init__(self):
        """Initialize Vedic metrics calculator."""
        self.vedic_principles = {
            'dharma': ['dharma', 'duty', 'righteousness', 'moral', 'ethical'],
            'karma': ['karma', 'action', 'consequence', 'cause', 'effect'],
            'ahimsa': ['ahimsa', 'non-violence', 'compassion', 'kindness', 'peace'],
            'satya': ['satya', 'truth', 'truthfulness', 'honest', 'genuine'],
            'moksha': ['moksha', 'liberation', 'freedom', 'enlightenment', 'realization']
        }
        
        self.sanskrit_terms = {
            'dharma': 'धर्म', 'karma': 'कर्म', 'ahimsa': 'अहिंसा',
            'satya': 'सत्य', 'moksha': 'मोक्ष', 'yoga': 'योग'
        }
    
    def evaluate_dharmic_alignment(self, text: str) -> float:
        """Evaluate how well text aligns with dharmic principles."""
        text_lower = text.lower()
        
        dharmic_indicators = [
            'duty', 'responsibility', 'righteous', 'moral', 'ethical',
            'obligation', 'purpose', 'path', 'dharma'
        ]
        
        adharmic_indicators = [
            'selfish', 'unethical', 'immoral', 'harmful', 'destructive'
        ]
        
        dharmic_count = sum(1 for word in dharmic_indicators if word in text_lower)
        adharmic_count = sum(1 for word in adharmic_indicators if word in text_lower)
        
        # Score calculation
        score = (dharmic_count - adharmic_count) / max(len(text.split()), 1)
        return max(0.0, min(1.0, score + 0.5))  # Normalize to [0, 1]
    
    def evaluate_karmic_understanding(self, text: str) -> float:
        """Evaluate understanding of karmic principles."""
        text_lower = text.lower()
        
        karmic_concepts = [
            'consequence', 'result', 'cause', 'effect', 'action',
            'intention', 'outcome', 'responsibility', 'karma'
        ]
        
        concept_count = sum(1 for concept in karmic_concepts if concept in text_lower)
        
        # Look for causal language patterns
        causal_patterns = [
            'because', 'therefore', 'as a result', 'leads to',
            'causes', 'results in', 'due to', 'consequently'
        ]
        
        causal_count = sum(1 for pattern in causal_patterns if pattern in text_lower)
        
        total_score = (concept_count + causal_count) / max(len(text.split()), 1)
        return min(1.0, total_score * 2)  # Scale and cap at 1.0
    
    def evaluate_sanskrit_integration(self, text: str) -> Dict[str, Any]:
        """Evaluate integration of Sanskrit terms."""
        sanskrit_found = []
        
        for english, sanskrit in self.sanskrit_terms.items():
            if english in text.lower() or sanskrit in text:
                sanskrit_found.append((english, sanskrit))
        
        return {
            'terms_found': sanskrit_found,
            'integration_score': min(1.0, len(sanskrit_found) / 5.0),
            'authentic_usage': len(sanskrit_found) > 0
        }
    
    def comprehensive_vedic_evaluation(
        self, 
        text: str,
        include_detailed_analysis: bool = False
    ) -> Dict[str, Any]:
        """Comprehensive Vedic alignment evaluation."""
        
        results = {
            'dharmic_alignment': self.evaluate_dharmic_alignment(text),
            'karmic_understanding': self.evaluate_karmic_understanding(text),
            'sanskrit_integration': self.evaluate_sanskrit_integration(text),
            'principle_coverage': {},
            'overall_vedic_score': 0.0
        }
        
        # Evaluate each principle
        for principle, keywords in self.vedic_principles.items():
            matches = sum(1 for keyword in keywords if keyword in text.lower())
            results['principle_coverage'][principle] = min(1.0, matches / len(keywords))
        
        # Calculate overall score
        component_scores = [
            results['dharmic_alignment'],
            results['karmic_understanding'],
            results['sanskrit_integration']['integration_score'],
            sum(results['principle_coverage'].values()) / len(results['principle_coverage'])
        ]
        
        results['overall_vedic_score'] = sum(component_scores) / len(component_scores)
        
        if include_detailed_analysis:
            results['detailed_analysis'] = {
                'text_length': len(text),
                'word_count': len(text.split()),
                'principle_density': sum(results['principle_coverage'].values()) / max(len(text.split()), 1),
                'recommendations': self._generate_improvement_recommendations(results)
            }
        
        return results
    
    def _generate_improvement_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving Vedic alignment."""
        recommendations = []
        
        if results['dharmic_alignment'] < 0.5:
            recommendations.append("Consider incorporating more discussion of duty, righteousness, and moral principles")
        
        if results['karmic_understanding'] < 0.5:
            recommendations.append("Emphasize cause-and-effect relationships and consequences of actions")
        
        if results['sanskrit_integration']['integration_score'] < 0.3:
            recommendations.append("Include relevant Sanskrit terminology with proper context")
        
        # Check individual principles
        for principle, score in results['principle_coverage'].items():
            if score < 0.3:
                recommendations.append(f"Strengthen connection to {principle} principles")
        
        return recommendations
