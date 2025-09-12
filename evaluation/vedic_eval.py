"""
Vedic knowledge and philosophical reasoning evaluation for INDRA LLM
(c) Divyansh Bharadwaj
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from datasets import Dataset

from model import INDRATransformer
from inference import VedicInferenceEngine, VedicGenerationConfig, VedicMode
from utils.metrics import VedicMetrics
from .benchmarks import BenchmarkResult

@dataclass
class VedicQuestion:
    """Vedic knowledge question structure."""
    question: str
    correct_answer: str
    options: Optional[List[str]] = None
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard
    source: str = "unknown"
    explanation: Optional[str] = None

@dataclass 
class PhilosophicalScenario:
    """Philosophical reasoning scenario."""
    scenario: str
    question: str
    expected_principles: List[str]
    category: str = "ethics"
    context: Optional[str] = None

class PhilosophicalBenchmarks:
    """Vedic and philosophical knowledge benchmarks."""
    
    def __init__(self):
        """Initialize Vedic benchmarks."""
        self.vedic_metrics = VedicMetrics()
        
        # Create standard Vedic knowledge questions
        self.vedic_questions = self._create_vedic_questions()
        self.philosophical_scenarios = self._create_philosophical_scenarios()
        self.sanskrit_translations = self._create_sanskrit_tests()
        
    def _create_vedic_questions(self) -> List[VedicQuestion]:
        """Create standard Vedic knowledge questions."""
        questions = [
            # Dharma questions
            VedicQuestion(
                question="What is the primary meaning of 'dharma' in Vedic philosophy?",
                correct_answer="Righteous duty and natural law that maintains cosmic order",
                options=[
                    "Righteous duty and natural law that maintains cosmic order",
                    "Religious ritual and ceremony",
                    "Personal belief system",
                    "Social customs and traditions"
                ],
                category="dharma",
                difficulty="easy",
                source="Vedic texts",
                explanation="Dharma represents the fundamental principle of righteousness and cosmic order in Vedic philosophy."
            ),
            
            VedicQuestion(
                question="According to the Bhagavad Gita, what is svadharma?",
                correct_answer="One's own duty according to one's nature and position in life",
                options=[
                    "Universal religious law",
                    "One's own duty according to one's nature and position in life", 
                    "Social obligations to family",
                    "Personal moral preferences"
                ],
                category="dharma",
                difficulty="medium",
                source="Bhagavad Gita"
            ),
            
            # Karma questions
            VedicQuestion(
                question="What is the law of karma?",
                correct_answer="The principle that every action has consequences that affect future experiences",
                options=[
                    "The principle that every action has consequences that affect future experiences",
                    "The belief in fate and predestination",
                    "The practice of good deeds for rewards",
                    "The cycle of birth and death"
                ],
                category="karma",
                difficulty="easy",
                source="Upanishads"
            ),
            
            VedicQuestion(
                question="What are the three types of karma mentioned in Vedic texts?",
                correct_answer="Sanchita (accumulated), Prarabdha (destined), and Agami (future)",
                category="karma",
                difficulty="hard",
                source="Vedantic texts"
            ),
            
            # Ahimsa questions
            VedicQuestion(
                question="What does 'ahimsa' mean in Vedic philosophy?",
                correct_answer="Non-violence in thought, word, and deed",
                options=[
                    "Peaceful meditation",
                    "Non-violence in thought, word, and deed",
                    "Avoiding conflicts",
                    "Vegetarian diet"
                ],
                category="ahimsa",
                difficulty="easy",
                source="Yoga Sutras"
            ),
            
            # Moksha questions  
            VedicQuestion(
                question="What is moksha in Vedic philosophy?",
                correct_answer="Liberation from the cycle of birth, death, and rebirth",
                options=[
                    "Attainment of wealth and prosperity",
                    "Liberation from the cycle of birth, death, and rebirth",
                    "Achievement of social status",
                    "Completion of religious duties"
                ],
                category="moksha",
                difficulty="easy",
                source="Upanishads"
            ),
            
            # Upanishad questions
            VedicQuestion(
                question="What is the central teaching of the Upanishads regarding the Self?",
                correct_answer="The individual self (Atman) is identical with the universal Self (Brahman)",
                category="upanishads",
                difficulty="medium",
                source="Principal Upanishads"
            ),
            
            # Yoga questions
            VedicQuestion(
                question="According to Patanjali, what are the eight limbs of yoga?",
                correct_answer="Yamas, Niyamas, Asana, Pranayama, Pratyahara, Dharana, Dhyana, Samadhi",
                category="yoga",
                difficulty="hard",
                source="Yoga Sutras of Patanjali"
            ),
            
            # Vedic cosmology
            VedicQuestion(
                question="In Vedic cosmology, what are the three worlds (triloka)?",
                correct_answer="Bhuh (earth), Bhuvah (atmosphere), and Svah (heaven)",
                category="cosmology",
                difficulty="medium",
                source="Vedic texts"
            ),
            
            # Sanskrit terms
            VedicQuestion(
                question="What does 'Sat-Chit-Ananda' refer to in Vedantic philosophy?",
                correct_answer="The three aspects of Brahman: Being, Consciousness, and Bliss",
                category="vedanta",
                difficulty="medium",
                source="Upanishads"
            )
        ]
        
        return questions
    
    def _create_philosophical_scenarios(self) -> List[PhilosophicalScenario]:
        """Create philosophical reasoning scenarios."""
        scenarios = [
            PhilosophicalScenario(
                scenario="A person discovers that their colleague has been taking credit for their work. They have evidence to expose this but doing so would likely result in the colleague losing their job and affecting their family.",
                question="How should one respond according to Vedic principles?",
                expected_principles=["dharma", "ahimsa", "satya"],
                category="workplace_ethics",
                context="This tests the balance between truthfulness (satya), non-harm (ahimsa), and righteous duty (dharma)."
            ),
            
            PhilosophicalScenario(
                scenario="A leader must make a decision that will benefit the majority of people but will cause hardship to a small group. The decision is legal and economically sound but morally complex.",
                question="What factors should guide this decision from a dharmic perspective?",
                expected_principles=["dharma", "karma", "ahimsa"],
                category="leadership_ethics",
                context="Tests understanding of dharmic decision-making and consideration of consequences."
            ),
            
            PhilosophicalScenario(
                scenario="Someone asks you for advice about pursuing a career that their family disapproves of but which aligns with their personal values and talents.",
                question="What guidance would Vedic wisdom offer in this situation?",
                expected_principles=["svadharma", "dharma"],
                category="personal_development",
                context="Tests understanding of svadharma (individual duty) versus social expectations."
            ),
            
            PhilosophicalScenario(
                scenario="In a heated argument, someone makes false accusations against you in front of others. You have the opportunity to retaliate with equally damaging but true information about them.",
                question="How should one respond according to ahimsa and other Vedic principles?",
                expected_principles=["ahimsa", "satya", "karma"],
                category="conflict_resolution",
                context="Tests understanding of non-violence, truthfulness, and karmic consequences."
            ),
            
            PhilosophicalScenario(
                scenario="A wealthy person asks whether they should donate money to charity or use it to expand their business, which would create jobs but also increase their personal wealth.",
                question="What would Vedic philosophy say about this choice?",
                expected_principles=["dharma", "karma", "moksha"],
                category="wealth_ethics",
                context="Tests understanding of dharmic use of wealth and karmic implications of actions."
            )
        ]
        
        return scenarios
    
    def _create_sanskrit_tests(self) -> List[Dict[str, str]]:
        """Create Sanskrit translation and understanding tests."""
        tests = [
            {
                "sanskrit": "तत्त्वमसि",
                "transliteration": "Tat tvam asi", 
                "meaning": "Thou art That (You are That)",
                "context": "Great saying (Mahavakya) from Chandogya Upanishad",
                "significance": "Expresses the identity between individual self and universal Self"
            },
            {
                "sanskrit": "अहं ब्रह्मास्मि",
                "transliteration": "Aham Brahmasmi",
                "meaning": "I am Brahman",
                "context": "Mahavakya from Brihadaranyaka Upanishad",
                "significance": "Declaration of the ultimate reality of the Self"
            },
            {
                "sanskrit": "सर्वं खल्विदं ब्रह्म",
                "transliteration": "Sarvam khalvidam brahma",
                "meaning": "All this is indeed Brahman",
                "context": "Statement from Chandogya Upanishad",
                "significance": "Everything in existence is manifestation of Brahman"
            },
            {
                "sanskrit": "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत",
                "transliteration": "Yada yada hi dharmasya glanir bhavati bharata",
                "meaning": "Whenever dharma declines, O Bharata",
                "context": "From Bhagavad Gita",
                "significance": "Beginning of Krishna's statement about divine incarnation"
            }
        ]
        
        return tests

class VedicEvaluator:
    """Comprehensive Vedic knowledge and philosophical reasoning evaluator."""
    
    def __init__(
        self,
        model: INDRATransformer,
        tokenizer,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Vedic evaluator.
        
        Args:
            model: INDRA transformer model
            tokenizer: VedicTokenizer instance
            device: Device for evaluation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Vedic inference engine
        self.vedic_engine = VedicInferenceEngine(self.model, self.tokenizer, self.device)
        
        # Initialize benchmark suite
        self.benchmarks = PhilosophicalBenchmarks()
        
        # Initialize metrics
        self.vedic_metrics = VedicMetrics()
        
        logging.info("Vedic evaluator initialized")
    
    def evaluate_vedic_knowledge(
        self,
        max_questions: Optional[int] = None
    ) -> BenchmarkResult:
        """
        Evaluate Vedic knowledge using question-answer format.
        
        Args:
            max_questions: Maximum number of questions to evaluate
            
        Returns:
            BenchmarkResult with Vedic knowledge scores
        """
        start_time = time.time()
        
        questions = self.benchmarks.vedic_questions
        if max_questions:
            questions = questions[:max_questions]
        
        correct_answers = 0
        total_questions = 0
        category_scores = {}
        detailed_results = []
        
        config = VedicGenerationConfig(
            vedic_mode=VedicMode.BALANCED,
            max_new_tokens=100,
            temperature=0.3,
            philosophical_depth=4
        )
        
        for question in questions:
            try:
                # Generate answer
                prompt = f"Question: {question.question}\nAnswer:"
                result = self.vedic_engine.generate_vedic(prompt, config)
                
                generated_answer = result['generated_text'].strip()
                
                # Evaluate answer
                is_correct = self._evaluate_answer(
                    question, generated_answer
                )
                
                if is_correct:
                    correct_answers += 1
                total_questions += 1
                
                # Track category performance
                category = question.category
                if category not in category_scores:
                    category_scores[category] = {'correct': 0, 'total': 0}
                
                category_scores[category]['total'] += 1
                if is_correct:
                    category_scores[category]['correct'] += 1
                
                detailed_results.append({
                    'question': question.question,
                    'category': question.category,
                    'difficulty': question.difficulty,
                    'correct_answer': question.correct_answer,
                    'generated_answer': generated_answer,
                    'is_correct': is_correct,
                    'vedic_analysis': result.get('vedic_analysis', {})
                })
                
            except Exception as e:
                logging.warning(f"Error evaluating question: {e}")
                continue
        
        # Calculate category accuracies
        for category in category_scores:
            if category_scores[category]['total'] > 0:
                category_scores[category]['accuracy'] = (
                    category_scores[category]['correct'] / 
                    category_scores[category]['total']
                )
        
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        time_taken = time.time() - start_time
        
        return BenchmarkResult(
            benchmark_name="vedic_knowledge",
            score=accuracy,
            num_samples=total_questions,
            time_taken=time_taken,
            detailed_results={
                'accuracy': accuracy,
                'correct_answers': correct_answers,
                'total_questions': total_questions,
                'category_scores': category_scores,
                'sample_results': detailed_results[:10]
            },
            metadata={
                'task_type': 'vedic_knowledge',
                'categories_tested': list(category_scores.keys())
            }
        )
    
    def evaluate_philosophical_reasoning(
        self,
        max_scenarios: Optional[int] = None
    ) -> BenchmarkResult:
        """
        Evaluate philosophical reasoning with ethical scenarios.
        
        Args:
            max_scenarios: Maximum number of scenarios to evaluate
            
        Returns:
            BenchmarkResult with philosophical reasoning scores
        """
        start_time = time.time()
        
        scenarios = self.benchmarks.philosophical_scenarios
        if max_scenarios:
            scenarios = scenarios[:max_scenarios]
        
        total_score = 0.0
        total_scenarios = 0
        category_scores = {}
        detailed_results = []
        
        config = VedicGenerationConfig(
            vedic_mode=VedicMode.BALANCED,
            max_new_tokens=200,
            temperature=0.7,
            philosophical_depth=5
        )
        
        for scenario in scenarios:
            try:
                # Create comprehensive prompt
                prompt = f"""Scenario: {scenario.scenario}

Question: {scenario.question}

Please provide a thoughtful response based on Vedic philosophical principles such as dharma, karma, ahimsa, and satya. Consider the ethical implications and provide reasoning for your guidance."""
                
                result = self.vedic_engine.generate_vedic(prompt, config)
                generated_response = result['generated_text']
                vedic_analysis = result['vedic_analysis']
                
                # Evaluate response
                score = self._evaluate_philosophical_response(
                    scenario, generated_response, vedic_analysis
                )
                
                total_score += score
                total_scenarios += 1
                
                # Track category performance
                category = scenario.category
                if category not in category_scores:
                    category_scores[category] = {'scores': [], 'count': 0}
                
                category_scores[category]['scores'].append(score)
                category_scores[category]['count'] += 1
                
                detailed_results.append({
                    'scenario': scenario.scenario,
                    'question': scenario.question,
                    'category': scenario.category,
                    'expected_principles': scenario.expected_principles,
                    'generated_response': generated_response,
                    'score': score,
                    'vedic_analysis': vedic_analysis
                })
                
            except Exception as e:
                logging.warning(f"Error evaluating scenario: {e}")
                continue
        
        # Calculate category averages
        for category in category_scores:
            scores = category_scores[category]['scores']
            if scores:
                category_scores[category]['average_score'] = sum(scores) / len(scores)
        
        avg_score = total_score / total_scenarios if total_scenarios > 0 else 0.0
        time_taken = time.time() - start_time
        
        return BenchmarkResult(
            benchmark_name="philosophical_reasoning",
            score=avg_score,
            num_samples=total_scenarios,
            time_taken=time_taken,
            detailed_results={
                'average_score': avg_score,
                'total_scenarios': total_scenarios,
                'category_scores': category_scores,
                'sample_results': detailed_results[:5]
            },
            metadata={
                'task_type': 'philosophical_reasoning',
                'score_range': '0.0 to 1.0',
                'categories_tested': list(category_scores.keys())
            }
        )
    
    def evaluate_sanskrit_understanding(self) -> BenchmarkResult:
        """Evaluate Sanskrit term understanding and translation."""
        start_time = time.time()
        
        tests = self.benchmarks.sanskrit_translations
        
        total_score = 0.0
        total_tests = 0
        detailed_results = []
        
        config = VedicGenerationConfig(
            vedic_mode=VedicMode.BALANCED,
            max_new_tokens=150,
            temperature=0.2,
            sanskrit_preference=0.8,
            philosophical_depth=3
        )
        
        for test in tests:
            try:
                # Test translation understanding
                prompt = f"""Sanskrit: {test['sanskrit']}
Transliteration: {test['transliteration']}

Please explain the meaning and philosophical significance of this Sanskrit phrase."""
                
                result = self.vedic_engine.generate_vedic(prompt, config)
                generated_explanation = result['generated_text']
                
                # Evaluate understanding
                score = self._evaluate_sanskrit_understanding(
                    test, generated_explanation
                )
                
                total_score += score
                total_tests += 1
                
                detailed_results.append({
                    'sanskrit': test['sanskrit'],
                    'transliteration': test['transliteration'],
                    'expected_meaning': test['meaning'],
                    'expected_significance': test['significance'],
                    'generated_explanation': generated_explanation,
                    'score': score
                })
                
            except Exception as e:
                logging.warning(f"Error evaluating Sanskrit test: {e}")
                continue
        
        avg_score = total_score / total_tests if total_tests > 0 else 0.0
        time_taken = time.time() - start_time
        
        return BenchmarkResult(
            benchmark_name="sanskrit_understanding",
            score=avg_score,
            num_samples=total_tests,
            time_taken=time_taken,
            detailed_results={
                'average_score': avg_score,
                'total_tests': total_tests,
                'sample_results': detailed_results
            },
            metadata={
                'task_type': 'sanskrit_understanding',
                'score_range': '0.0 to 1.0'
            }
        )
    
    def evaluate_vedic_dialogue_generation(
        self,
        topics: Optional[List[str]] = None
    ) -> BenchmarkResult:
        """Evaluate quality of Vedic philosophical dialogue generation."""
        start_time = time.time()
        
        if topics is None:
            topics = [
                "The nature of reality and consciousness",
                "The relationship between individual and universal Self", 
                "The path to liberation from suffering",
                "The role of duty in spiritual development",
                "The balance between action and renunciation"
            ]
        
        total_score = 0.0
        total_dialogues = 0
        detailed_results = []
        
        for topic in topics:
            try:
                # Generate philosophical dialogue
                dialogue_result = self.vedic_engine.generate_vedic_dialogue(
                    topic=topic,
                    turns=6
                )
                
                # Evaluate dialogue quality
                score = self._evaluate_dialogue_quality(dialogue_result)
                
                total_score += score
                total_dialogues += 1
                
                detailed_results.append({
                    'topic': topic,
                    'dialogue': dialogue_result['dialogue'][:3],  # First 3 turns
                    'analysis': dialogue_result['analysis'],
                    'score': score
                })
                
            except Exception as e:
                logging.warning(f"Error generating dialogue for topic '{topic}': {e}")
                continue
        
        avg_score = total_score / total_dialogues if total_dialogues > 0 else 0.0
        time_taken = time.time() - start_time
        
        return BenchmarkResult(
            benchmark_name="vedic_dialogue",
            score=avg_score,
            num_samples=total_dialogues,
            time_taken=time_taken,
            detailed_results={
                'average_score': avg_score,
                'total_dialogues': total_dialogues,
                'sample_results': detailed_results
            },
            metadata={
                'task_type': 'dialogue_generation',
                'score_range': '0.0 to 1.0'
            }
        )
    
    def _evaluate_answer(self, question: VedicQuestion, generated_answer: str) -> bool:
        """Evaluate if generated answer is correct."""
        generated_lower = generated_answer.lower()
        correct_lower = question.correct_answer.lower()
        
        # Simple keyword matching for now
        # In production, you might use more sophisticated matching
        if question.options:
            # Multiple choice - check if generated answer matches any option
            for option in question.options:
                if option.lower() == correct_lower:
                    # Check if key phrases from correct option appear in generated answer
                    key_words = option.lower().split()[:5]  # First 5 words
                    if any(word in generated_lower for word in key_words if len(word) > 3):
                        return True
        else:
            # Open-ended - check for key concepts
            key_concepts = correct_lower.split()
            concept_matches = sum(1 for concept in key_concepts 
                                if len(concept) > 3 and concept in generated_lower)
            
            # Consider correct if at least 30% of key concepts are present
            return concept_matches / max(len(key_concepts), 1) >= 0.3
        
        return False
    
    def _evaluate_philosophical_response(
        self,
        scenario: PhilosophicalScenario,
        response: str,
        vedic_analysis: Dict[str, Any]
    ) -> float:
        """Evaluate quality of philosophical reasoning response."""
        score = 0.0
        
        # Check for expected principles (40% of score)
        principles_mentioned = 0
        response_lower = response.lower()
        
        for principle in scenario.expected_principles:
            if principle.lower() in response_lower:
                principles_mentioned += 1
        
        principle_score = principles_mentioned / max(len(scenario.expected_principles), 1)
        score += 0.4 * principle_score
        
        # Check Vedic alignment from analysis (30% of score)
        overall_alignment = vedic_analysis.get('overall_alignment', 0.0)
        score += 0.3 * overall_alignment
        
        # Check for philosophical depth (20% of score)
        depth_indicators = [
            'consequence', 'principle', 'ethics', 'moral', 'dharma', 'karma',
            'right', 'wrong', 'consider', 'balance', 'wisdom', 'guidance'
        ]
        depth_matches = sum(1 for indicator in depth_indicators 
                           if indicator in response_lower)
        depth_score = min(1.0, depth_matches / 6.0)  # Normalize to max 1.0
        score += 0.2 * depth_score
        
        # Check response length and structure (10% of score)
        word_count = len(response.split())
        length_score = 1.0 if 50 <= word_count <= 300 else 0.5
        score += 0.1 * length_score
        
        return min(1.0, score)
    
    def _evaluate_sanskrit_understanding(
        self, 
        test: Dict[str, str], 
        explanation: str
    ) -> float:
        """Evaluate Sanskrit understanding and explanation."""
        score = 0.0
        explanation_lower = explanation.lower()
        
        # Check if meaning concepts are explained (50% of score)
        meaning_words = test['meaning'].lower().split()
        meaning_matches = sum(1 for word in meaning_words 
                            if len(word) > 3 and word in explanation_lower)
        meaning_score = meaning_matches / max(len(meaning_words), 1)
        score += 0.5 * meaning_score
        
        # Check if significance is addressed (30% of score)
        significance_words = test['significance'].lower().split()
        significance_matches = sum(1 for word in significance_words 
                                 if len(word) > 3 and word in explanation_lower)
        significance_score = significance_matches / max(len(significance_words), 1)
        score += 0.3 * significance_score
        
        # Check for philosophical context (20% of score)
        context_indicators = ['upanishad', 'vedic', 'brahman', 'self', 'reality', 'consciousness']
        context_matches = sum(1 for indicator in context_indicators 
                            if indicator in explanation_lower)
        context_score = min(1.0, context_matches / 3.0)
        score += 0.2 * context_score
        
        return min(1.0, score)
    
    def _evaluate_dialogue_quality(self, dialogue_result: Dict[str, Any]) -> float:
        """Evaluate quality of generated philosophical dialogue."""
        analysis = dialogue_result.get('analysis', {})
        
        # Use dialogue analysis metrics
        coherence = analysis.get('coherence_score', 0.0)
        philosophical_depth = analysis.get('philosophical_depth', 0.0)
        vedic_integration = analysis.get('vedic_integration', 0.0)
        
        # Calculate weighted average
        score = (0.4 * coherence + 0.3 * philosophical_depth + 0.3 * vedic_integration)
        
        return min(1.0, score)
    
    def run_comprehensive_vedic_evaluation(
        self,
        output_dir: str,
        max_questions: int = 50,
        max_scenarios: int = 10
    ) -> Dict[str, Any]:
        """
        Run comprehensive Vedic evaluation suite.
        
        Args:
            output_dir: Directory to save results
            max_questions: Maximum questions for knowledge test
            max_scenarios: Maximum scenarios for reasoning test
            
        Returns:
            Comprehensive evaluation results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logging.info("Starting comprehensive Vedic evaluation...")
        
        results = {}
        
        # Vedic knowledge evaluation
        logging.info("Evaluating Vedic knowledge...")
        knowledge_result = self.evaluate_vedic_knowledge(max_questions)
        results['vedic_knowledge'] = {
            'score': knowledge_result.score,
            'num_samples': knowledge_result.num_samples,
            'time_taken': knowledge_result.time_taken,
            'detailed_results': knowledge_result.detailed_results
        }
        
        # Philosophical reasoning evaluation
        logging.info("Evaluating philosophical reasoning...")
        reasoning_result = self.evaluate_philosophical_reasoning(max_scenarios)
        results['philosophical_reasoning'] = {
            'score': reasoning_result.score,
            'num_samples': reasoning_result.num_samples, 
            'time_taken': reasoning_result.time_taken,
            'detailed_results': reasoning_result.detailed_results
        }
        
        # Sanskrit understanding evaluation
        logging.info("Evaluating Sanskrit understanding...")
        sanskrit_result = self.evaluate_sanskrit_understanding()
        results['sanskrit_understanding'] = {
            'score': sanskrit_result.score,
            'num_samples': sanskrit_result.num_samples,
            'time_taken': sanskrit_result.time_taken,
            'detailed_results': sanskrit_result.detailed_results
        }
        
        #

        # Dialogue generation evaluation
        logging.info("Evaluating Vedic dialogue generation...")
        dialogue_result = self.evaluate_vedic_dialogue_generation()
        results['vedic_dialogue'] = {
            'score': dialogue_result.score,
            'num_samples': dialogue_result.num_samples,
            'time_taken': dialogue_result.time_taken,
            'detailed_results': dialogue_result.detailed_results
        }
        
        # Calculate overall Vedic score
        scores = [
            results['vedic_knowledge']['score'],
            results['philosophical_reasoning']['score'], 
            results['sanskrit_understanding']['score'],
            results['vedic_dialogue']['score']
        ]
        overall_vedic_score = sum(scores) / len(scores)
        
        results['overall_vedic_evaluation'] = {
            'overall_score': overall_vedic_score,
            'component_scores': {
                'knowledge': results['vedic_knowledge']['score'],
                'reasoning': results['philosophical_reasoning']['score'],
                'sanskrit': results['sanskrit_understanding']['score'],
                'dialogue': results['vedic_dialogue']['score']
            },
            'total_time': sum(r.get('time_taken', 0) for r in [
                knowledge_result, reasoning_result, sanskrit_result, dialogue_result
            ]),
            'evaluation_timestamp': time.time()
        }
        
        # Save results
        results_file = output_path / "vedic_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logging.info(f"Vedic evaluation results saved to {results_file}")
        
        # Generate detailed report
        self._generate_vedic_report(results, output_path / "vedic_evaluation_report.txt")
        
        return results
    
    def _generate_vedic_report(self, results: Dict[str, Any], output_file: Path):
        """Generate human-readable Vedic evaluation report."""
        with open(output_file, 'w') as f:
            f.write("INDRA LLM - Vedic Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall summary
            overall = results['overall_vedic_evaluation']
            f.write(f"Overall Vedic Score: {overall['overall_score']:.4f}\n")
            f.write(f"Total Evaluation Time: {overall['total_time']:.1f} seconds\n\n")
            
            # Component scores
            f.write("Component Scores:\n")
            f.write("-" * 20 + "\n")
            components = overall['component_scores']
            f.write(f"Vedic Knowledge: {components['knowledge']:.4f}\n")
            f.write(f"Philosophical Reasoning: {components['reasoning']:.4f}\n")
            f.write(f"Sanskrit Understanding: {components['sanskrit']:.4f}\n")
            f.write(f"Dialogue Generation: {components['dialogue']:.4f}\n\n")
            
            # Detailed breakdown
            f.write("Detailed Results:\n")
            f.write("-" * 20 + "\n\n")
            
            # Vedic Knowledge
            if 'vedic_knowledge' in results:
                knowledge = results['vedic_knowledge']
                f.write(f"Vedic Knowledge Assessment:\n")
                f.write(f"  Accuracy: {knowledge['detailed_results']['accuracy']:.4f}\n")
                f.write(f"  Questions: {knowledge['detailed_results']['total_questions']}\n")
                f.write(f"  Correct: {knowledge['detailed_results']['correct_answers']}\n")
                
                # Category breakdown
                if 'category_scores' in knowledge['detailed_results']:
                    f.write("  Category Performance:\n")
                    for cat, scores in knowledge['detailed_results']['category_scores'].items():
                        acc = scores.get('accuracy', 0.0)
                        f.write(f"    {cat.title()}: {acc:.3f} ({scores['correct']}/{scores['total']})\n")
                f.write("\n")
            
            # Philosophical Reasoning
            if 'philosophical_reasoning' in results:
                reasoning = results['philosophical_reasoning']
                f.write(f"Philosophical Reasoning Assessment:\n")
                f.write(f"  Average Score: {reasoning['detailed_results']['average_score']:.4f}\n")
                f.write(f"  Scenarios: {reasoning['detailed_results']['total_scenarios']}\n")
                
                # Category breakdown
                if 'category_scores' in reasoning['detailed_results']:
                    f.write("  Category Performance:\n")
                    for cat, scores in reasoning['detailed_results']['category_scores'].items():
                        avg = scores.get('average_score', 0.0)
                        f.write(f"    {cat.replace('_', ' ').title()}: {avg:.3f}\n")
                f.write("\n")
            
            # Sanskrit Understanding
            if 'sanskrit_understanding' in results:
                sanskrit = results['sanskrit_understanding']
                f.write(f"Sanskrit Understanding Assessment:\n")
                f.write(f"  Average Score: {sanskrit['detailed_results']['average_score']:.4f}\n")
                f.write(f"  Tests: {sanskrit['detailed_results']['total_tests']}\n\n")
            
            # Dialogue Generation
            if 'vedic_dialogue' in results:
                dialogue = results['vedic_dialogue']
                f.write(f"Dialogue Generation Assessment:\n")
                f.write(f"  Average Score: {dialogue['detailed_results']['average_score']:.4f}\n")
                f.write(f"  Dialogues: {dialogue['detailed_results']['total_dialogues']}\n\n")
            
            # Recommendations
            f.write("Recommendations for Improvement:\n")
            f.write("-" * 30 + "\n")
            
            # Generate recommendations based on scores
            recommendations = self._generate_recommendations(results)
            for rec in recommendations:
                f.write(f"• {rec}\n")
        
        logging.info(f"Vedic evaluation report saved to {output_file}")
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on results."""
        recommendations = []
        
        # Check component scores for weaknesses
        components = results['overall_vedic_evaluation']['component_scores']
        
        if components['knowledge'] < 0.7:
            recommendations.append(
                "Enhance training on core Vedic concepts and terminology from primary texts"
            )
        
        if components['reasoning'] < 0.7:
            recommendations.append(
                "Improve philosophical reasoning capabilities through case-based learning"
            )
        
        if components['sanskrit'] < 0.6:
            recommendations.append(
                "Strengthen Sanskrit language understanding and philosophical context"
            )
        
        if components['dialogue'] < 0.6:
            recommendations.append(
                "Develop more sophisticated dialogue generation with multi-perspective reasoning"
            )
        
        # Check for specific category weaknesses
        if 'vedic_knowledge' in results:
            cat_scores = results['vedic_knowledge']['detailed_results'].get('category_scores', {})
            weak_categories = [cat for cat, scores in cat_scores.items() 
                             if scores.get('accuracy', 0) < 0.6]
            
            if weak_categories:
                recommendations.append(
                    f"Focus additional training on: {', '.join(weak_categories)}"
                )
        
        # Overall performance recommendations
        overall_score = results['overall_vedic_evaluation']['overall_score']
        
        if overall_score < 0.5:
            recommendations.append(
                "Consider fundamental retraining with expanded Vedic curriculum"
            )
        elif overall_score < 0.7:
            recommendations.append(
                "Implement targeted fine-tuning on identified weak areas"
            )
        else:
            recommendations.append(
                "Model shows strong Vedic alignment - consider advanced philosophical reasoning tasks"
            )
        
        return recommendations

# Additional evaluation utilities

class VedicBenchmarkSuite:
    """Complete Vedic benchmark suite with multiple evaluation modes."""
    
    def __init__(self, model: INDRATransformer, tokenizer, device: Optional[torch.device] = None):
        """Initialize benchmark suite."""
        self.evaluator = VedicEvaluator(model, tokenizer, device)
        
    def quick_evaluation(self) -> Dict[str, Any]:
        """Run quick Vedic evaluation (reduced samples)."""
        return self.evaluator.run_comprehensive_vedic_evaluation(
            output_dir="./quick_vedic_eval",
            max_questions=20,
            max_scenarios=5
        )
    
    def full_evaluation(self) -> Dict[str, Any]:
        """Run full Vedic evaluation."""
        return self.evaluator.run_comprehensive_vedic_evaluation(
            output_dir="./full_vedic_eval",
            max_questions=100,
            max_scenarios=20
        )
    
    def custom_evaluation(
        self, 
        custom_questions: List[VedicQuestion],
        custom_scenarios: List[PhilosophicalScenario],
        output_dir: str = "./custom_vedic_eval"
    ) -> Dict[str, Any]:
        """Run evaluation with custom questions and scenarios."""
        # Temporarily replace benchmark questions
        original_questions = self.evaluator.benchmarks.vedic_questions
        original_scenarios = self.evaluator.benchmarks.philosophical_scenarios
        
        self.evaluator.benchmarks.vedic_questions = custom_questions
        self.evaluator.benchmarks.philosophical_scenarios = custom_scenarios
