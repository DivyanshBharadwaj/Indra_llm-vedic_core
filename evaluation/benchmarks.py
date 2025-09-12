"""
# Standard benchmarks evaluation for INDRA LLM
# (c) Divyansh Bharadwaj
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from model import INDRATransformer
from inference import InferenceEngine, GenerationConfig
from utils.metrics import MetricsCalculator

@dataclass
class BenchmarkResult:
    """Result from a benchmark evaluation."""
    benchmark_name: str
    score: float
    num_samples: int
    time_taken: float
    detailed_results: Dict[str, Any]
    metadata: Dict[str, Any]

class StandardBenchmarks:
    """Standard LLM benchmarks implementation."""
    
    def __init__(self):
        """Initialize standard benchmarks."""
        self.metrics_calculator = MetricsCalculator()
        
        self.benchmark_configs = {
            'hellaswag': {
                'dataset_name': 'hellaswag',
                'split': 'validation',
                'task_type': 'multiple_choice',
                'metric': 'accuracy'
            },
            'arc_easy': {
                'dataset_name': 'ai2_arc',
                'dataset_config': 'ARC-Easy',
                'split': 'test',
                'task_type': 'multiple_choice',
                'metric': 'accuracy'
            },
            'arc_challenge': {
                'dataset_name': 'ai2_arc', 
                'dataset_config': 'ARC-Challenge',
                'split': 'test',
                'task_type': 'multiple_choice',
                'metric': 'accuracy'
            },
            'winogrande': {
                'dataset_name': 'winogrande',
                'dataset_config': 'winogrande_xl',
                'split': 'validation',
                'task_type': 'multiple_choice',
                'metric': 'accuracy'
            },
            'lambada': {
                'dataset_name': 'lambada',
                'split': 'test',
                'task_type': 'completion',
                'metric': 'accuracy'
            }
        }
    
    def evaluate_hellaswag(
        self,
        model: INDRATransformer,
        tokenizer,
        max_samples: int = 1000,
        device: Optional[torch.device] = None
    ) -> BenchmarkResult:
        """Evaluate on HellaSwag commonsense reasoning."""
        start_time = time.time()
        
        try:
            # Load dataset
            dataset = load_dataset('hellaswag', split='validation')
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            correct = 0
            total = 0
            detailed_results = []
            
            for example in dataset:
                context = example['ctx']
                endings = example['endings']
                correct_idx = int(example['label'])
                
                # Calculate likelihood for each ending
                scores = []
                for ending in endings:
                    full_text = context + ' ' + ending
                    score = self._calculate_text_likelihood(
                        full_text, model, tokenizer, device
                    )
                    scores.append(score)
                
                # Predict the most likely ending
                predicted_idx = scores.index(max(scores))
                
                is_correct = predicted_idx == correct_idx
                if is_correct:
                    correct += 1
                total += 1
                
                detailed_results.append({
                    'context': context,
                    'endings': endings,
                    'correct_idx': correct_idx,
                    'predicted_idx': predicted_idx,
                    'scores': scores,
                    'correct': is_correct
                })
            
            accuracy = correct / total if total > 0 else 0.0
            time_taken = time.time() - start_time
            
            return BenchmarkResult(
                benchmark_name='hellaswag',
                score=accuracy,
                num_samples=total,
                time_taken=time_taken,
                detailed_results={
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total,
                    'examples': detailed_results[:10]  # First 10 for inspection
                },
                metadata={'task_type': 'commonsense_reasoning'}
            )
            
        except Exception as e:
            logging.error(f"Error evaluating HellaSwag: {e}")
            return self._create_error_result('hellaswag', str(e))
    
    def evaluate_arc(
        self,
        model: INDRATransformer,
        tokenizer,
        variant: str = 'easy',
        max_samples: int = 500,
        device: Optional[torch.device] = None
    ) -> BenchmarkResult:
        """Evaluate on ARC science questions."""
        start_time = time.time()
        
        try:
            # Load dataset
            config_name = 'ARC-Easy' if variant == 'easy' else 'ARC-Challenge'
            dataset = load_dataset('ai2_arc', config_name, split='test')
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            correct = 0
            total = 0
            detailed_results = []
            
            for example in dataset:
                question = example['question']
                choices = example['choices']
                correct_answer = example['answerKey']
                
                # Format choices
                choice_texts = choices['text']
                choice_labels = choices['label']
                
                # Find correct index
                correct_idx = choice_labels.index(correct_answer)
                
                # Calculate likelihood for each choice
                scores = []
                for choice in choice_texts:
                    full_text = f"Question: {question}\nAnswer: {choice}"
                    score = self._calculate_text_likelihood(
                        full_text, model, tokenizer, device
                    )
                    scores.append(score)
                
                # Predict the most likely choice
                predicted_idx = scores.index(max(scores))
                
                is_correct = predicted_idx == correct_idx
                if is_correct:
                    correct += 1
                total += 1
                
                detailed_results.append({
                    'question': question,
                    'choices': choice_texts,
                    'correct_answer': correct_answer,
                    'correct_idx': correct_idx,
                    'predicted_idx': predicted_idx,
                    'scores': scores,
                    'correct': is_correct
                })
            
            accuracy = correct / total if total > 0 else 0.0
            time_taken = time.time() - start_time
            
            return BenchmarkResult(
                benchmark_name=f'arc_{variant}',
                score=accuracy,
                num_samples=total,
                time_taken=time_taken,
                detailed_results={
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total,
                    'examples': detailed_results[:10]
                },
                metadata={'task_type': 'science_qa', 'variant': variant}
            )
            
        except Exception as e:
            logging.error(f"Error evaluating ARC {variant}: {e}")
            return self._create_error_result(f'arc_{variant}', str(e))
    
    def evaluate_lambada(
        self,
        model: INDRATransformer,
        tokenizer,
        max_samples: int = 1000,
        device: Optional[torch.device] = None
    ) -> BenchmarkResult:
        """Evaluate on LAMBADA reading comprehension."""
        start_time = time.time()
        
        try:
            # Load dataset
            dataset = load_dataset('lambada', split='test')
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            correct = 0
            total = 0
            detailed_results = []
            
            inference_engine = InferenceEngine(model, tokenizer, device)
            
            for example in dataset:
                text = example['text']
                
                # Split context and target word
                words = text.split()
                context = ' '.join(words[:-1])
                target_word = words[-1].lower()
                
                # Generate next word
                generation_config = GenerationConfig(
                    max_new_tokens=5,
                    temperature=0.1,
                    do_sample=False,
                    use_cache=True
                )
                
                generated = inference_engine.generate(
                    context, generation_config, return_dict=True
                )
                
                predicted_text = generated['generated_texts'].strip().lower()
                predicted_word = predicted_text.split()[0] if predicted_text.split() else ""
                
                is_correct = predicted_word == target_word
                if is_correct:
                    correct += 1
                total += 1
                
                detailed_results.append({
                    'context': context,
                    'target_word': target_word,
                    'predicted_word': predicted_word,
                    'correct': is_correct
                })
            
            accuracy = correct / total if total > 0 else 0.0
            time_taken = time.time() - start_time
            
            return BenchmarkResult(
                benchmark_name='lambada',
                score=accuracy,
                num_samples=total,
                time_taken=time_taken,
                detailed_results={
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total,
                    'examples': detailed_results[:10]
                },
                metadata={'task_type': 'reading_comprehension'}
            )
            
        except Exception as e:
            logging.error(f"Error evaluating LAMBADA: {e}")
            return self._create_error_result('lambada', str(e))
    
    def evaluate_winogrande(
        self,
        model: INDRATransformer,
        tokenizer,
        max_samples: int = 1000,
        device: Optional[torch.device] = None
    ) -> BenchmarkResult:
        """Evaluate on WinoGrande commonsense reasoning."""
        start_time = time.time()
        
        try:
            # Load dataset
            dataset = load_dataset('winogrande', 'winogrande_xl', split='validation')
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            correct = 0
            total = 0
            detailed_results = []
            
            for example in dataset:
                sentence = example['sentence']
                option1 = example['option1']
                option2 = example['option2']
                answer = example['answer']  # '1' or '2'
                
                # Replace underscore with each option
                sentence1 = sentence.replace('_', option1)
                sentence2 = sentence.replace('_', option2)
                
                # Calculate likelihood for each completion
                score1 = self._calculate_text_likelihood(sentence1, model, tokenizer, device)
                score2 = self._calculate_text_likelihood(sentence2, model, tokenizer, device)
                
                # Predict based on higher likelihood
                predicted_answer = '1' if score1 > score2 else '2'
                
                is_correct = predicted_answer == answer
                if is_correct:
                    correct += 1
                total += 1
                
                detailed_results.append({
                    'sentence': sentence,
                    'option1': option1,
                    'option2': option2,
                    'correct_answer': answer,
                    'predicted_answer': predicted_answer,
                    'scores': [score1, score2],
                    'correct': is_correct
                })
            
            accuracy = correct / total if total > 0 else 0.0
            time_taken = time.time() - start_time
            
            return BenchmarkResult(
                benchmark_name='winogrande',
                score=accuracy,
                num_samples=total,
                time_taken=time_taken,
                detailed_results={
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total,
                    'examples': detailed_results[:10]
                },
                metadata={'task_type': 'commonsense_reasoning'}
            )
            
        except Exception as e:
            logging.error(f"Error evaluating WinoGrande: {e}")
            return self._create_error_result('winogrande', str(e))
    
    def evaluate_mmlu(
        self,
        model: INDRATransformer,
        tokenizer,
        subjects: Optional[List[str]] = None,
        max_samples_per_subject: int = 100,
        device: Optional[torch.device] = None
    ) -> BenchmarkResult:
        """Evaluate on MMLU (Massive Multitask Language Understanding)."""
        start_time = time.time()
        
        try:
            # Load dataset
            if subjects is None:
                # Use a subset of subjects for faster evaluation
                subjects = [
                    'abstract_algebra', 'college_mathematics', 'computer_security',
                    'econometrics', 'formal_logic', 'high_school_psychology',
                    'moral_scenarios', 'philosophy', 'world_religions'
                ]
            
            all_results = []
            subject_scores = {}
            
            for subject in subjects:
                try:
                    dataset = load_dataset('cais/mmlu', subject, split='test')
                    if max_samples_per_subject:
                        dataset = dataset.select(range(min(max_samples_per_subject, len(dataset))))
                    
                    subject_correct = 0
                    subject_total = 0
                    
                    for example in dataset:
                        question = example['question']
                        choices = example['choices']  # List of 4 choices
                        answer = example['answer']  # Integer 0-3
                        
                        # Format question with choices
                        formatted_choices = []
                        choice_labels = ['A', 'B', 'C', 'D']
                        
                        for i, choice in enumerate(choices):
                            formatted_choices.append(f"{choice_labels[i]}. {choice}")
                        
                        # Calculate likelihood for each choice
                        scores = []
                        for i, choice in enumerate(choices):
                            full_text = f"Question: {question}\nAnswer: {choice_labels[i]}. {choice}"
                            score = self._calculate_text_likelihood(full_text, model, tokenizer, device)
                            scores.append(score)
                        
                        # Predict the most likely choice
                        predicted_answer = scores.index(max(scores))
                        
                        is_correct = predicted_answer == answer
                        if is_correct:
                            subject_correct += 1
                        subject_total += 1
                        
                        all_results.append({
                            'subject': subject,
                            'question': question,
                            'choices': choices,
                            'correct_answer': answer,
                            'predicted_answer': predicted_answer,
                            'correct': is_correct
                        })
                    
                    subject_accuracy = subject_correct / subject_total if subject_total > 0 else 0.0
                    subject_scores[subject] = {
                        'accuracy': subject_accuracy,
                        'correct': subject_correct,
                        'total': subject_total
                    }
                    
                except Exception as e:
                    logging.warning(f"Error evaluating MMLU subject {subject}: {e}")
                    subject_scores[subject] = {'accuracy': 0.0, 'correct': 0, 'total': 0}
            
            # Calculate overall accuracy
            total_correct = sum(scores['correct'] for scores in subject_scores.values())
            total_samples = sum(scores['total'] for scores in subject_scores.values())
            overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            
            time_taken = time.time() - start_time
            
            return BenchmarkResult(
                benchmark_name='mmlu',
                score=overall_accuracy,
                num_samples=total_samples,
                time_taken=time_taken,
                detailed_results={
                    'overall_accuracy': overall_accuracy,
                    'subject_scores': subject_scores,
                    'total_correct': total_correct,
                    'total_samples': total_samples,
                    'subjects_evaluated': subjects,
                    'sample_results': all_results[:20]  # First 20 for inspection
                },
                metadata={
                    'task_type': 'knowledge_qa',
                    'subjects_count': len(subjects)
                }
            )
            
        except Exception as e:
            logging.error(f"Error evaluating MMLU: {e}")
            return self._create_error_result('mmlu', str(e))
    
    def evaluate_truthfulqa(
        self,
        model: INDRATransformer,
        tokenizer,
        max_samples: int = 500,
        device: Optional[torch.device] = None
    ) -> BenchmarkResult:
        """Evaluate on TruthfulQA for truthfulness."""
        start_time = time.time()
        
        try:
            # Load dataset
            dataset = load_dataset('truthful_qa', 'multiple_choice', split='validation')
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            correct = 0
            total = 0
            detailed_results = []
            
            inference_engine = InferenceEngine(model, tokenizer, device)
            generation_config = GenerationConfig(
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False
            )
            
            for example in dataset:
                question = example['question']
                mc1_targets = example['mc1_targets']  # Dict with choices and labels
                
                choices = mc1_targets['choices']
                labels = mc1_targets['labels']  # 1 for correct, 0 for incorrect
                
                # Find correct answer index
                correct_indices = [i for i, label in enumerate(labels) if label == 1]
                if not correct_indices:
                    continue  # Skip if no correct answer
                
                correct_idx = correct_indices[0]
                
                # Calculate likelihood for each choice
                scores = []
                for choice in choices:
                    full_text = f"Q: {question}\nA: {choice}"
                    score = self._calculate_text_likelihood(full_text, model, tokenizer, device)
                    scores.append(score)
                
                # Predict the most likely choice
                predicted_idx = scores.index(max(scores))
                
                is_correct = predicted_idx == correct_idx
                if is_correct:
                    correct += 1
                total += 1
                
                detailed_results.append({
                    'question': question,
                    'choices': choices,
                    'correct_idx': correct_idx,
                    'predicted_idx': predicted_idx,
                    'correct': is_correct
                })
            
            accuracy = correct / total if total > 0 else 0.0
            time_taken = time.time() - start_time
            
            return BenchmarkResult(
                benchmark_name='truthfulqa',
                score=accuracy,
                num_samples=total,
                time_taken=time_taken,
                detailed_results={
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total,
                    'examples': detailed_results[:10]
                },
                metadata={'task_type': 'truthfulness'}
            )
            
        except Exception as e:
            logging.error(f"Error evaluating TruthfulQA: {e}")
            return self._create_error_result('truthfulqa', str(e))
    
    def _calculate_text_likelihood(
        self,
        text: str,
        model: INDRATransformer,
        tokenizer,
        device: Optional[torch.device] = None
    ) -> float:
        """Calculate likelihood of a text under the model."""
        if device is None:
            device = next(model.parameters()).device
        
        # Tokenize text
        inputs = tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        
        if input_ids.size(1) == 0:
            return float('-inf')
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
        
        if logits.size(1) <= 1:
            return float('-inf')
        
        # Calculate log probability
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Get log probability of actual tokens
        target_log_probs = log_probs[0, :-1].gather(1, input_ids[0, 1:].unsqueeze(-1))
        
        # Average log probability
        avg_log_prob = target_log_probs.mean().item()
        
        return avg_log_prob
    
    def _create_error_result(self, benchmark_name: str, error_msg: str) -> BenchmarkResult:
        """Create error result for failed benchmark."""
        return BenchmarkResult(
            benchmark_name=benchmark_name,
            score=0.0,
            num_samples=0,
            time_taken=0.0,
            detailed_results={'error': error_msg},
            metadata={'status': 'error'}
        )

class Evaluator:
    """Main evaluator class for INDRA LLM."""
    
    def __init__(
        self,
        model: INDRATransformer,
        tokenizer,
        device: Optional[torch.device] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: INDRA transformer model
            tokenizer: Tokenizer instance
            device: Device for evaluation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize benchmark suites
        self.standard_benchmarks = StandardBenchmarks()
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        logging.info(f"Evaluator initialized on {self.device}")
    
    def evaluate_standard_benchmarks(
        self,
        benchmarks: Optional[List[str]] = None,
        max_samples_per_benchmark: int = 1000
    ) -> Dict[str, BenchmarkResult]:
        """
        Evaluate on standard benchmarks.
        
        Args:
            benchmarks: List of benchmark names to evaluate
            max_samples_per_benchmark: Maximum samples per benchmark
            
        Returns:
            Dictionary of benchmark results
        """
        if benchmarks is None:
            benchmarks = ['hellaswag', 'arc_easy', 'arc_challenge', 'lambada', 'winogrande', 'mmlu', 'truthfulqa']
        
        results = {}
        
        for benchmark_name in benchmarks:
            logging.info(f"Evaluating {benchmark_name}...")
            
            try:
                if benchmark_name == 'hellaswag':
                    result = self.standard_benchmarks.evaluate_hellaswag(
                        self.model, self.tokenizer, max_samples_per_benchmark, self.device
                    )
                elif benchmark_name == 'arc_easy':
                    result = self.standard_benchmarks.evaluate_arc(
                        self.model, self.tokenizer, 'easy', max_samples_per_benchmark, self.device
                    )
                elif benchmark_name == 'arc_challenge':
                    result = self.standard_benchmarks.evaluate_arc(
                        self.model, self.tokenizer, 'challenge', max_samples_per_benchmark, self.device
                    )
                elif benchmark_name == 'lambada':
                    result = self.standard_benchmarks.evaluate_lambada(
                        self.model, self.tokenizer, max_samples_per_benchmark, self.device
                    )
                elif benchmark_name == 'winogrande':
                    result = self.standard_benchmarks.evaluate_winogrande(
                        self.model, self.tokenizer, max_samples_per_benchmark, self.device
                    )
                elif benchmark_name == 'mmlu':
                    result = self.standard_benchmarks.evaluate_mmlu(
                        self.model, self.tokenizer, None, 100, self.device  # Reduced samples for MMLU
                    )
                elif benchmark_name == 'truthfulqa':
                    result = self.standard_benchmarks.evaluate_truthfulqa(
                        self.model, self.tokenizer, max_samples_per_benchmark, self.device
                    )
                else:
                    logging.warning(f"Unknown benchmark: {benchmark_name}")
                    continue
                
                results[benchmark_name] = result
                
                logging.info(f"{benchmark_name}: {result.score:.4f} ({result.num_samples} samples)")
                
            except Exception as e:
                logging.error(f"Error evaluating {benchmark_name}: {e}")
                results[benchmark_name] = self.standard_benchmarks._create_error_result(
                    benchmark_name, str(e)
                )
        
        return results
    
    def evaluate_generation_quality(
        self,
        prompts: List[str],
        reference_texts: Optional[List[str]] = None,
        generation_config: Optional[GenerationConfig] = None
    ) -> Dict[str, Any]:
        """
        Evaluate generation quality on custom prompts.
        
        Args:
            prompts: List of prompts to generate from
            reference_texts: Optional reference texts for comparison
            generation_config: Generation configuration
            
        Returns:
            Dictionary of generation quality metrics
        """
        if generation_config is None:
            generation_config = GenerationConfig(
                max_new_tokens=100,
                temperature=0.8,
                top_p=0.9,
                do_sample=True
            )
        
        from ..inference import InferenceEngine
        inference_engine = InferenceEngine(self.model, self.tokenizer, self.device)
        
        generated_texts = []
        generation_times = []
        
        # Generate responses
        for prompt in prompts:
            start_time = time.time()
            
            result = inference_engine.generate(
                prompt, generation_config, return_dict=True
            )
            
            generation_time = time.time() - start_time
            
            generated_texts.append(result['generated_texts'])
            generation_times.append(generation_time)
        
        # Calculate metrics
        results = {
            'num_prompts': len(prompts),
            'avg_generation_time': sum(generation_times) / len(generation_times),
            'total_generation_time': sum(generation_times),
        }
        
        # Diversity metrics
        diversity_metrics = self.metrics_calculator.calculate_diversity_metrics(generated_texts)
        results.update(diversity_metrics)
        
        # BLEU and ROUGE if references provided
        if reference_texts and len(reference_texts) == len(generated_texts):
            bleu_score = self.metrics_calculator.calculate_bleu_score(
                generated_texts, [[ref] for ref in reference_texts]
            )
            rouge_scores = self.metrics_calculator.calculate_rouge_score(
                generated_texts, reference_texts
            )
            
            results['bleu_score'] = bleu_score
            results.update(rouge_scores)
        
        # Speed metrics
        total_tokens = sum(len(text.split()) for text in generated_texts)
        speed_metrics = self.metrics_calculator.calculate_inference_speed(
            total_tokens, sum(generation_times)
        )
        results.update(speed_metrics)
        
        # Sample outputs for inspection
        results['sample_outputs'] = [
            {'prompt': prompt, 'generated': generated}
            for prompt, generated in zip(prompts[:5], generated_texts[:5])
        ]
        
        return results
    
    def evaluate_perplexity(
        self,
        test_texts: List[str],
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        Evaluate perplexity on test texts.
        
        Args:
            test_texts: List of texts to evaluate
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with perplexity metrics
        """
        total_log_likelihood = 0.0
        total_tokens = 0
        perplexities = []
        
        # Process in batches
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.model.config.n_positions
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            
            # Calculate perplexity for each sequence
            for j, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
                # Get valid tokens (not padding)
                valid_length = mask.sum().item()
                if valid_length <= 1:
                    continue
                    
                valid_ids = ids[:valid_length]
                valid_logits = logits[j, :valid_length - 1, :]  # Exclude last position
                
                # Calculate cross entropy loss
                targets = valid_ids[1:]  # Shift targets
                log_probs = torch.nn.functional.log_softmax(valid_logits, dim=-1)
                token_log_probs = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
                
                # Calculate perplexity for this sequence
                seq_log_likelihood = token_log_probs.sum().item()
                seq_perplexity = torch.exp(-token_log_probs.mean()).item()
                
                total_log_likelihood += seq_log_likelihood
                total_tokens += len(targets)
                perplexities.append(seq_perplexity)
        
        # Calculate overall metrics
        if not perplexities:
            return {
                'avg_perplexity': float('inf'),
                'overall_perplexity': float('inf'),
                'num_texts': 0,
                'total_tokens': 0,
                'perplexity_std': 0.0,
                'min_perplexity': float('inf'),
                'max_perplexity': float('inf')
            }
        
        avg_perplexity = sum(perplexities) / len(perplexities)
        overall_perplexity = torch.exp(-total_log_likelihood / total_tokens).item() if total_tokens > 0 else float('inf')
        
        return {
            'avg_perplexity': avg_perplexity,
            'overall_perplexity': overall_perplexity,
            'num_texts': len(test_texts),
            'total_tokens': total_tokens,
            'perplexity_std': float(torch.tensor(perplexities).std()),
            'min_perplexity': min(perplexities),
            'max_perplexity': max(perplexities)
        }
    
    def run_comprehensive_evaluation(
        self,
        output_dir: str,
        include_standard_benchmarks: bool = True,
        include_generation_eval: bool = True,
        include_perplexity_eval: bool = True,
        custom_prompts: Optional[List[str]] = None,
        custom_test_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation suite.
        
        Args:
            output_dir: Directory to save results
            include_standard_benchmarks: Whether to run standard benchmarks
            include_generation_eval: Whether to evaluate generation quality
            include_perplexity_eval: Whether to evaluate perplexity
            custom_prompts: Custom prompts for generation evaluation
            custom_test_texts: Custom texts for perplexity evaluation
            
        Returns:
            Comprehensive evaluation results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'model_info': {
                'model_name': 'INDRA LLM',
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'device': str(self.device),
                'evaluation_timestamp': time.time()
            }
        }
        
        # Standard benchmarks
        if include_standard_benchmarks:
            logging.info("Running standard benchmarks...")
            benchmark_results = self.evaluate_standard_benchmarks()
            results['standard_benchmarks'] = {
                name: {
                    'score': result.score,
                    'num_samples': result.num_samples,
                    'time_taken': result.time_taken,
                    'metadata': result.metadata
                }
                for name, result in benchmark_results.items()
            }
        
        # Generation quality evaluation
        if include_generation_eval:
            logging.info("Evaluating generation quality...")
            
            if custom_prompts is None:
                custom_prompts = [
                    "What is the meaning of life according to Vedic philosophy?",
                    "Explain the concept of artificial intelligence from a dharmic perspective.",
                    "Describe the importance of environmental conservation and ahimsa.",
                    "What are the benefits of meditation and yoga practice?",
                    "How can we achieve world peace through Vedic principles?"
                ]
            
            generation_results = self.evaluate_generation_quality(custom_prompts)
            results['generation_quality'] = generation_results
        
        # Perplexity evaluation
        if include_perplexity_eval and custom_test_texts:
            logging.info("Evaluating perplexity...")
            perplexity_results = self.evaluate_perplexity(custom_test_texts)
            results['perplexity'] = perplexity_results
        
        # Save results
        results_file = output_path / "evaluation_results.json"
        with open(results_file, 'w') as f:
            import json
            json.dump(results, f, indent=2, default=str)
        
        logging.info(f"Evaluation results saved to {results_file}")
        
        # Generate summary report
        self._generate_summary_report(results, output_path / "evaluation_summary.txt")
        
        return results
    
    def _generate_summary_report(self, results: Dict[str, Any], output_file: Path):
        """Generate human-readable summary report."""
        with open(output_file, 'w') as f:
            f.write("INDRA LLM Evaluation Summary\n")
            f.write("=" * 40 + "\n\n")
            
            # Model info
            model_info = results.get('model_info', {})
            f.write(f"Model: {model_info.get('model_name', 'Unknown')}\n")
            f.write(f"Parameters: {model_info.get('parameters', 0):,}\n")
            f.write(f"Device: {model_info.get('device', 'Unknown')}\n\n")
            
            # Standard benchmarks
            if 'standard_benchmarks' in results:
                f.write("Standard Benchmarks:\n")
                f.write("-" * 20 + "\n")
                
                for name, result in results['standard_benchmarks'].items():
                    score = result['score']
                    samples = result['num_samples']
                    f.write(f"{name.upper()}: {score:.4f} ({samples} samples)\n")
                
                f.write("\n")
            
            # Generation quality
            if 'generation_quality' in results:
                gen_results = results['generation_quality']
                f.write("Generation Quality:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Diversity (Distinct-1): {gen_results.get('distinct_1', 0):.4f}\n")
                f.write(f"Diversity (Distinct-2): {gen_results.get('distinct_2', 0):.4f}\n")
                f.write(f"Average Generation Time: {gen_results.get('avg_generation_time', 0):.2f}s\n")
                f.write(f"Tokens per Second: {gen_results.get('tokens_per_second', 0):.1f}\n")
                
                if 'bleu_score' in gen_results:
                    f.write(f"BLEU Score: {gen_results['bleu_score']:.4f}\n")
                
                f.write("\n")
            
            # Perplexity
            if 'perplexity' in results:
                perp_results = results['perplexity']
                f.write("Perplexity:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Average Perplexity: {perp_results.get('avg_perplexity', 0):.2f}\n")
                f.write(f"Overall Perplexity: {perp_results.get('overall_perplexity', 0):.2f}\n")
                f.write(f"Test Texts: {perp_results.get('num_texts', 0)}\n")
                f.write(f"Total Tokens: {perp_results.get('total_tokens', 0):,}\n")
        
        logging.info(f"Summary report saved to {output_file}")

# CLI interface for running benchmarks
def main():
    """Main function for CLI benchmark runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="INDRA LLM Benchmark Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--benchmarks", type=str, nargs="+", 
                       choices=['hellaswag', 'arc_easy', 'arc_challenge', 'lambada', 'winogrande', 'mmlu', 'truthfulqa'],
                       default=['hellaswag', 'arc_easy'], help="Benchmarks to run")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum samples per benchmark")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Load model and tokenizer (placeholder - you'd implement actual loading)
    logging.info(f"Loading model from {args.model_path}")
    logging.info(f"Loading tokenizer from {args.tokenizer_path}")
    
    # This would load actual model and tokenizer
    # model = load_model(args.model_path, device)
    # tokenizer = load_tokenizer(args.tokenizer_path)
    
    # For now, create placeholder
    from ..config import ModelConfig
    from ..model import INDRATransformer
    
    config = ModelConfig()
    model = INDRATransformer(config).to(device)
    tokenizer = None  # Would load actual tokenizer
    
    # Create evaluator
    evaluator = Evaluator(model, tokenizer, device)
    
    # Run benchmarks
    logging.info(f"Running benchmarks: {args.benchmarks}")
    results = evaluator.evaluate_standard_benchmarks(
        benchmarks=args.benchmarks,
        max_samples_per_benchmark=args.max_samples
    )
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "benchmark_results.json"
    with open(results_file, 'w') as f:
        import json
        # Convert BenchmarkResult objects to dictionaries
        serializable_results = {}
        for name, result in results.items():
            serializable_results[name] = {
                'benchmark_name': result.benchmark_name,
                'score': result.score,
                'num_samples': result.num_samples,
                'time_taken': result.time_taken,
                'detailed_results': result.detailed_results,
                'metadata': result.metadata
            }
        json.dump(serializable_results, f, indent=2, default=str)
    
    # Print summary
    print("\nBenchmark Results:")
    print("=" * 50)
    for name, result in results.items():
        print(f"{name.upper()}: {result.score:.4f} ({result.num_samples} samples, {result.time_taken:.1f}s)")
    
    print(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    main()
