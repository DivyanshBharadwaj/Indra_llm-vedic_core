"""
Vedic-guided inference engine for INDRA LLM with philosophical reasoning
(c) Divyansh Bharadwaj
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F

from .generation import InferenceEngine, GenerationConfig
from model import INDRATransformer

class VedicMode(Enum):
    """Vedic reasoning modes."""
    DHARMIC = "dharmic"  # Focus on righteousness and duty
    KARMIC = "karmic"    # Focus on action and consequences
    AHIMSA = "ahimsa"    # Focus on non-violence
    SATYA = "satya"      # Focus on truthfulness
    MOKSHA = "moksha"    # Focus on liberation
    BALANCED = "balanced"  # Balanced across all principles

@dataclass
class VedicGenerationConfig(GenerationConfig):
    """Extended generation config with Vedic parameters."""
    vedic_mode: VedicMode = VedicMode.BALANCED
    vedic_weight: float = 1.0
    dharmic_threshold: float = 0.7
    use_vedic_memory: bool = True
    sanskrit_preference: float = 0.2
    philosophical_depth: int = 3  # 1-5 scale
    include_sources: bool = False

class VedicInferenceEngine(InferenceEngine):
    """Vedic-guided inference engine with philosophical reasoning."""
    
    def __init__(
        self,
        model: INDRATransformer,
        tokenizer,
        device: Optional[torch.device] = None,
        vedic_concepts_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Vedic inference engine.
        
        Args:
            model: INDRA transformer model with Vedic core
            tokenizer: VedicTokenizer instance
            device: Device for inference
            vedic_concepts_path: Path to Vedic concepts database
        """
        super().__init__(model, tokenizer, device, **kwargs)
        
        # Load Vedic concepts
        self.vedic_concepts = self._load_vedic_concepts(vedic_concepts_path)
        
        # Vedic reasoning prompts
        self.vedic_prompts = {
            VedicMode.DHARMIC: "From the perspective of dharma (righteousness and duty), ",
            VedicMode.KARMIC: "Considering the law of karma (action and consequence), ",
            VedicMode.AHIMSA: "Following the principle of ahimsa (non-violence), ",
            VedicMode.SATYA: "Adhering to satya (truthfulness and reality), ",
            VedicMode.MOKSHA: "With the goal of moksha (liberation and self-realization), ",
            VedicMode.BALANCED: "Drawing from Vedic wisdom, ",
        }
        
        # Sanskrit translations for common philosophical terms
        self.sanskrit_terms = {
            "dharma": "धर्म",
            "karma": "कर्म", 
            "ahimsa": "अहिंसा",
            "satya": "सत्य",
            "moksha": "मोक्ष",
            "brahman": "ब्रह्मन्",
            "atman": "आत्मन्",
            "yoga": "योग",
            "meditation": "ध्यान",
            "wisdom": "ज्ञान",
        }
        
        logging.info("Vedic inference engine initialized")
    
    def _load_vedic_concepts(self, concepts_path: Optional[str]) -> Dict[str, List[str]]:
        """Load Vedic concepts database."""
        # Default concepts if no file provided
        default_concepts = {
            "dharma_concepts": [
                "righteous duty", "moral law", "natural order", "ethical conduct",
                "personal duty", "social responsibility", "cosmic order"
            ],
            "karma_concepts": [
                "action", "consequence", "cause and effect", "moral causation",
                "deed", "intention", "result", "cosmic justice"
            ],
            "ahimsa_concepts": [
                "non-violence", "non-harm", "compassion", "kindness",
                "peaceful action", "protection of life", "gentleness"
            ],
            "moksha_concepts": [
                "liberation", "freedom", "self-realization", "enlightenment",
                "release", "spiritual goal", "ultimate purpose"
            ]
        }
        
        if concepts_path:
            try:
                import json
                with open(concepts_path, 'r', encoding='utf-8') as f:
                    loaded_concepts = json.load(f)
                    default_concepts.update(loaded_concepts)
            except Exception as e:
                logging.warning(f"Could not load Vedic concepts from {concepts_path}: {e}")
        
        return default_concepts
    
    def generate_vedic(
        self,
        prompt: str,
        vedic_config: Optional[VedicGenerationConfig] = None,
        context_texts: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with Vedic guidance and reasoning.
        
        Args:
            prompt: Input prompt
            vedic_config: Vedic generation configuration
            context_texts: Additional Vedic texts for context
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generated text and Vedic analysis
        """
        if vedic_config is None:
            vedic_config = VedicGenerationConfig(**kwargs)
        
        # Enhance prompt with Vedic context
        enhanced_prompt = self._enhance_prompt_with_vedic_context(
            prompt, vedic_config, context_texts
        )
        
        # Generate with Vedic guidance
        generation_result = self.generate(
            enhanced_prompt,
            generation_config=vedic_config,
            return_dict=True
        )
        
        generated_text = generation_result['generated_texts']
        
        # Analyze generated text for Vedic alignment
        vedic_analysis = self._analyze_vedic_alignment(
            prompt, generated_text, vedic_config
        )
        
        # Post-process for Sanskrit integration
        if vedic_config.sanskrit_preference > 0:
            generated_text = self._integrate_sanskrit_terms(
                generated_text, vedic_config.sanskrit_preference
            )
        
        return {
            'generated_text': generated_text,
            'enhanced_prompt': enhanced_prompt,
            'vedic_analysis': vedic_analysis,
            'vedic_config': vedic_config,
            'generation_stats': generation_result['generation_stats']
        }
    
    def _enhance_prompt_with_vedic_context(
        self,
        prompt: str,
        config: VedicGenerationConfig,
        context_texts: Optional[List[str]] = None
    ) -> str:
        """Enhance prompt with Vedic philosophical context."""
        enhanced_parts = []
        
        # Add Vedic mode-specific framing
        mode_prompt = self.vedic_prompts.get(config.vedic_mode, "")
        if mode_prompt:
            enhanced_parts.append(mode_prompt)
        
        # Add relevant Vedic concepts
        relevant_concepts = self._get_relevant_vedic_concepts(prompt, config.vedic_mode)
        if relevant_concepts:
            concept_text = f"considering the concepts of {', '.join(relevant_concepts[:3])}, "
            enhanced_parts.append(concept_text)
        
        # Add context texts if provided
        if context_texts:
            context_intro = "Drawing from the following Vedic wisdom:\n"
            context_content = "\n".join(f"• {text}" for text in context_texts[:2])
            enhanced_parts.extend([context_intro, context_content, "\n\n"])
        
        # Add philosophical depth instruction
        if config.philosophical_depth > 3:
            enhanced_parts.append("Please provide a deep philosophical analysis. ")
        elif config.philosophical_depth > 1:
            enhanced_parts.append("Please explain with philosophical insight. ")
        
        # Add the original prompt
        enhanced_parts.append(prompt)
        
        # Add Vedic markers for tokenizer
        if hasattr(self.tokenizer, 'special_tokens'):
            vedic_start = self.tokenizer.special_tokens.get('vedic_start', '<vedic>')
            vedic_end = self.tokenizer.special_tokens.get('vedic_end', '</vedic>')
            return f"{vedic_start}{''.join(enhanced_parts)}{vedic_end}"
        
        return ''.join(enhanced_parts)
    
    def _get_relevant_vedic_concepts(
        self,
        prompt: str,
        mode: VedicMode
    ) -> List[str]:
        """Get relevant Vedic concepts for the prompt and mode."""
        prompt_lower = prompt.lower()
        relevant_concepts = []
        
        # Mode-specific concepts
        if mode == VedicMode.DHARMIC or mode == VedicMode.BALANCED:
            if any(word in prompt_lower for word in ['duty', 'right', 'moral', 'ethical', 'should']):
                relevant_concepts.extend(self.vedic_concepts.get('dharma_concepts', [])[:2])
        
        if mode == VedicMode.KARMIC or mode == VedicMode.BALANCED:
            if any(word in prompt_lower for word in ['action', 'consequence', 'result', 'cause']):
                relevant_concepts.extend(self.vedic_concepts.get('karma_concepts', [])[:2])
        
        if mode == VedicMode.AHIMSA or mode == VedicMode.BALANCED:
            if any(word in prompt_lower for word in ['violence', 'harm', 'peace', 'compassion']):
                relevant_concepts.extend(self.vedic_concepts.get('ahimsa_concepts', [])[:2])
        
        if mode == VedicMode.MOKSHA or mode == VedicMode.BALANCED:
            if any(word in prompt_lower for word in ['freedom', 'liberation', 'enlightenment', 'spiritual']):
                relevant_concepts.extend(self.vedic_concepts.get('moksha_concepts', [])[:2])
        
        return list(set(relevant_concepts))  # Remove duplicates
    
    def _analyze_vedic_alignment(
        self,
        prompt: str,
        generated_text: str,
        config: VedicGenerationConfig
    ) -> Dict[str, Any]:
        """Analyze generated text for Vedic philosophical alignment."""
        analysis = {
            'dharma_score': 0.0,
            'karma_awareness': 0.0,
            'ahimsa_compliance': 0.0,
            'satya_adherence': 0.0,
            'overall_alignment': 0.0,
            'philosophical_concepts_used': [],
            'sanskrit_terms_used': [],
            'vedic_principles_applied': []
        }
        
        text_lower = generated_text.lower()
        
        # Analyze dharma alignment
        dharma_indicators = ['duty', 'right', 'moral', 'ethical', 'righteousness', 'dharma']
        dharma_count = sum(1 for word in dharma_indicators if word in text_lower)
        analysis['dharma_score'] = min(1.0, dharma_count / 5.0)
        
        # Analyze karma awareness
        karma_indicators = ['consequence', 'result', 'action', 'effect', 'karma', 'cause']
        karma_count = sum(1 for word in karma_indicators if word in text_lower)
        analysis['karma_awareness'] = min(1.0, karma_count / 5.0)
        
        # Analyze ahimsa compliance
        ahimsa_positive = ['peace', 'compassion', 'kindness', 'non-violence', 'ahimsa']
        ahimsa_negative = ['violence', 'harm', 'hurt', 'destroy', 'kill']
        ahimsa_pos_count = sum(1 for word in ahimsa_positive if word in text_lower)
        ahimsa_neg_count = sum(1 for word in ahimsa_negative if word in text_lower)
        analysis['ahimsa_compliance'] = max(0.0, min(1.0, (ahimsa_pos_count - ahimsa_neg_count) / 3.0))
        
        # Analyze satya adherence (truthfulness indicators)
        satya_indicators = ['truth', 'honest', 'genuine', 'authentic', 'real', 'satya']
        satya_count = sum(1 for word in satya_indicators if word in text_lower)
        analysis['satya_adherence'] = min(1.0, satya_count / 3.0)
        
        # Find philosophical concepts
        for concept_category, concepts in self.vedic_concepts.items():
            for concept in concepts:
                if concept.lower() in text_lower:
                    analysis['philosophical_concepts_used'].append(concept)
        
        # Find Sanskrit terms
        for english_term, sanskrit_term in self.sanskrit_terms.items():
            if english_term in text_lower or sanskrit_term in generated_text:
                analysis['sanskrit_terms_used'].append((english_term, sanskrit_term))
        
        # Calculate overall alignment
        scores = [
            analysis['dharma_score'],
            analysis['karma_awareness'],
            analysis['ahimsa_compliance'],
            analysis['satya_adherence']
        ]
        analysis['overall_alignment'] = sum(scores) / len(scores)
        
        # Identify applied principles
        if analysis['dharma_score'] > 0.3:
            analysis['vedic_principles_applied'].append('dharma')
        if analysis['karma_awareness'] > 0.3:
            analysis['vedic_principles_applied'].append('karma')
        if analysis['ahimsa_compliance'] > 0.3:
            analysis['vedic_principles_applied'].append('ahimsa')
        if analysis['satya_adherence'] > 0.3:
            analysis['vedic_principles_applied'].append('satya')
        
        return analysis
    
    def _integrate_sanskrit_terms(
        self,
        text: str,
        sanskrit_preference: float
    ) -> str:
        """Integrate Sanskrit terms into generated text."""
        if sanskrit_preference <= 0:
            return text
        
        modified_text = text
        
        for english_term, sanskrit_term in self.sanskrit_terms.items():
            if english_term in text.lower():
                # Replace some occurrences with Sanskrit
                import re
                pattern = r'\b' + re.escape(english_term) + r'\b'
                matches = list(re.finditer(pattern, modified_text, re.IGNORECASE))
                
                # Replace based on preference probability
                for match in matches:
                    if torch.rand(1).item() < sanskrit_preference:
                        replacement = f"{match.group()} ({sanskrit_term})"
                        modified_text = (
                            modified_text[:match.start()] + 
                            replacement + 
                            modified_text[match.end():]
                        )
        
        return modified_text
    
    def explain_vedic_reasoning(
        self,
        question: str,
        principles: Optional[List[str]] = None,
        depth: int = 3
    ) -> Dict[str, Any]:
        """
        Provide Vedic reasoning explanation for a question.
        
        Args:
            question: Question to analyze
            principles: Specific Vedic principles to apply
            depth: Depth of explanation (1-5)
            
        Returns:
            Detailed Vedic reasoning and explanation
        """
        if principles is None:
            principles = ['dharma', 'karma', 'ahimsa', 'satya']
        
        explanations = {}
        
        for principle in principles:
            if principle == 'dharma':
                mode = VedicMode.DHARMIC
            elif principle == 'karma':
                mode = VedicMode.KARMIC
            elif principle == 'ahimsa':
                mode = VedicMode.AHIMSA
            elif principle == 'satya':
                mode = VedicMode.SATYA
            else:
                mode = VedicMode.BALANCED
            
            # Generate explanation from this principle's perspective
            config = VedicGenerationConfig(
                vedic_mode=mode,
                philosophical_depth=depth,
                max_new_tokens=200,
                temperature=0.7
            )
            
            explanation_prompt = f"How would the Vedic principle of {principle} guide us in understanding: {question}"
            
            result = self.generate_vedic(explanation_prompt, config)
            explanations[principle] = {
                'explanation': result['generated_text'],
                'alignment_score': result['vedic_analysis']['overall_alignment'],
                'concepts_used': result['vedic_analysis']['philosophical_concepts_used']
            }
        
        # Generate synthesis
        synthesis_prompt = f"Synthesizing the perspectives of {', '.join(principles)}, provide a unified Vedic understanding of: {question}"
        
        synthesis_config = VedicGenerationConfig(
            vedic_mode=VedicMode.BALANCED,
            philosophical_depth=depth,
            max_new_tokens=300,
            temperature=0.8
        )
        
        synthesis_result = self.generate_vedic(synthesis_prompt, synthesis_config)
        
        return {
            'question': question,
            'principle_explanations': explanations,
            'unified_understanding': synthesis_result['generated_text'],
            'overall_analysis': synthesis_result['vedic_analysis'],
            'synthesis_quality': synthesis_result['vedic_analysis']['overall_alignment']
        }
    
    def compare_responses(
        self,
        prompt: str,
        responses: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple responses for Vedic alignment.
        
        Args:
            prompt: Original prompt
            responses: List of responses to compare
            
        Returns:
            Comparison analysis with rankings
        """
        comparisons = []
        
        for i, response in enumerate(responses):
            config = VedicGenerationConfig()  # Default config for analysis
            analysis = self._analyze_vedic_alignment(prompt, response, config)
            
            comparisons.append({
                'response_id': i,
                'response_text': response,
                'vedic_analysis': analysis,
                'overall_score': analysis['overall_alignment']
            })
        
        # Sort by overall alignment score
        comparisons.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Add rankings
        for rank, comparison in enumerate(comparisons, 1):
            comparison['rank'] = rank
        
        # Generate comparison summary
        best_response = comparisons[0]
        worst_response = comparisons[-1]
        
        summary = {
            'total_responses': len(responses),
            'best_response': {
                'rank': 1,
                'score': best_response['overall_score'],
                'strengths': self._identify_strengths(best_response['vedic_analysis'])
            },
            'worst_response': {
                'rank': len(responses),
                'score': worst_response['overall_score'],
                'weaknesses': self._identify_weaknesses(worst_response['vedic_analysis'])
            },
            'average_score': sum(c['overall_score'] for c in comparisons) / len(comparisons)
        }
        
        return {
            'prompt': prompt,
            'detailed_comparisons': comparisons,
            'summary': summary,
            'recommendations': self._generate_improvement_recommendations(comparisons)
        }
    
    def _identify_strengths(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify strengths in Vedic analysis."""
        strengths = []
        
        if analysis['dharma_score'] > 0.7:
            strengths.append("Strong dharmic reasoning")
        if analysis['karma_awareness'] > 0.7:
            strengths.append("Clear understanding of karmic principles")
        if analysis['ahimsa_compliance'] > 0.7:
            strengths.append("Excellent non-violence adherence")
        if analysis['satya_adherence'] > 0.7:
            strengths.append("High truthfulness and authenticity")
        
        if len(analysis['philosophical_concepts_used']) > 3:
            strengths.append("Rich philosophical vocabulary")
        
        if len(analysis['sanskrit_terms_used']) > 2:
            strengths.append("Good integration of Sanskrit terminology")
        
        return strengths
    
    def _identify_weaknesses(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify weaknesses in Vedic analysis."""
        weaknesses = []
        
        if analysis['dharma_score'] < 0.3:
            weaknesses.append("Lacks dharmic reasoning")
        if analysis['karma_awareness'] < 0.3:
            weaknesses.append("Limited understanding of karmic principles")
        if analysis['ahimsa_compliance'] < 0.3:
            weaknesses.append("May contain harmful or violent elements")
        if analysis['satya_adherence'] < 0.3:
            weaknesses.append("Lacks truthfulness or authenticity")
        
        if len(analysis['philosophical_concepts_used']) == 0:
            weaknesses.append("No philosophical concepts referenced")
        
        if len(analysis['vedic_principles_applied']) < 2:
            weaknesses.append("Limited application of Vedic principles")
        
        return weaknesses
    
    def _generate_improvement_recommendations(
        self, 
        comparisons: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations for improving Vedic alignment."""
        recommendations = []
        
        # Analyze common weaknesses
        avg_dharma = sum(c['vedic_analysis']['dharma_score'] for c in comparisons) / len(comparisons)
        avg_karma = sum(c['vedic_analysis']['karma_awareness'] for c in comparisons) / len(comparisons)
        avg_ahimsa = sum(c['vedic_analysis']['ahimsa_compliance'] for c in comparisons) / len(comparisons)
        avg_satya = sum(c['vedic_analysis']['satya_adherence'] for c in comparisons) / len(comparisons)
        
        if avg_dharma < 0.5:
            recommendations.append("Incorporate more discussion of duty, righteousness, and moral principles")
        
        if avg_karma < 0.5:
            recommendations.append("Emphasize cause-and-effect relationships and consequences of actions")
        
        if avg_ahimsa < 0.5:
            recommendations.append("Focus on non-violent, compassionate approaches and solutions")
        
        if avg_satya < 0.5:
            recommendations.append("Emphasize truthfulness, authenticity, and genuine understanding")
        
        # Check for Sanskrit integration
        sanskrit_usage = sum(len(c['vedic_analysis']['sanskrit_terms_used']) for c in comparisons) / len(comparisons)
        if sanskrit_usage < 1:
            recommendations.append("Consider including relevant Sanskrit terminology with explanations")
        
        # Check philosophical depth
        concept_usage = sum(len(c['vedic_analysis']['philosophical_concepts_used']) for c in comparisons) / len(comparisons)
        if concept_usage < 2:
            recommendations.append("Draw upon more Vedic philosophical concepts and teachings")
        
        return recommendations
    
    def generate_vedic_dialogue(
        self,
        topic: str,
        participants: Optional[List[str]] = None,
        turns: int = 6
    ) -> Dict[str, Any]:
        """
        Generate a philosophical dialogue on a topic from Vedic perspectives.
        
        Args:
            topic: Topic for dialogue
            participants: List of philosophical perspectives (or use defaults)
            turns: Number of dialogue turns
            
        Returns:
            Generated dialogue with analysis
        """
        if participants is None:
            participants = ["Dharmic Sage", "Karmic Philosopher", "Ahimsa Advocate", "Satya Seeker"]
        
        # Map participants to Vedic modes
        participant_modes = {
            "Dharmic Sage": VedicMode.DHARMIC,
            "Karmic Philosopher": VedicMode.KARMIC,
            "Ahimsa Advocate": VedicMode.AHIMSA,
            "Satya Seeker": VedicMode.SATYA,
        }
        
        dialogue = []
        context = f"Topic for philosophical discussion: {topic}\n\n"
        
        for turn in range(turns):
            participant = participants[turn % len(participants)]
            mode = participant_modes.get(participant, VedicMode.BALANCED)
            
            # Create prompt for this participant's turn
            if turn == 0:
                prompt = f"{participant}, please share your initial thoughts on {topic}."
            else:
                recent_dialogue = "\n".join([
                    f"{entry['speaker']}: {entry['text']}" 
                    for entry in dialogue[-2:] if dialogue
                ])
                prompt = f"Given this discussion:\n{recent_dialogue}\n\n{participant}, please respond:"
            
            # Generate response
            config = VedicGenerationConfig(
                vedic_mode=mode,
                max_new_tokens=150,
                temperature=0.8,
                philosophical_depth=4
            )
            
            full_prompt = context + prompt
            result = self.generate_vedic(full_prompt, config)
            
            dialogue.append({
                'turn': turn + 1,
                'speaker': participant,
                'mode': mode.value,
                'text': result['generated_text'],
                'vedic_analysis': result['vedic_analysis']
            })
            
            # Update context with new dialogue
            context += f"{participant}: {result['generated_text']}\n\n"
        
        # Analyze dialogue quality
        dialogue_analysis = self._analyze_dialogue_quality(dialogue, topic)
        
        return {
            'topic': topic,
            'participants': participants,
            'dialogue': dialogue,
            'analysis': dialogue_analysis,
            'summary': self._generate_dialogue_summary(dialogue, topic)
        }
    
    def _analyze_dialogue_quality(
        self, 
        dialogue: List[Dict[str, Any]], 
        topic: str
    ) -> Dict[str, Any]:
        """Analyze the quality of generated dialogue."""
        analysis = {
            'coherence_score': 0.0,
            'philosophical_depth': 0.0,
            'vedic_integration': 0.0,
            'participant_balance': {},
            'concept_coverage': set(),
            'dialogue_flow': []
        }
        
        # Analyze each turn
        for entry in dialogue:
            vedic_analysis = entry['vedic_analysis']
            
            # Track concepts covered
            analysis['concept_coverage'].update(vedic_analysis['philosophical_concepts_used'])
            
            # Track participant contributions
            speaker = entry['speaker']
            if speaker not in analysis['participant_balance']:
                analysis['participant_balance'][speaker] = {
                    'turns': 0,
                    'avg_alignment': 0.0,
                    'concepts_introduced': 0
                }
            
            analysis['participant_balance'][speaker]['turns'] += 1
            analysis['participant_balance'][speaker]['avg_alignment'] += vedic_analysis['overall_alignment']
            analysis['participant_balance'][speaker]['concepts_introduced'] += len(vedic_analysis['philosophical_concepts_used'])
        
        # Finalize participant statistics
        for speaker_stats in analysis['participant_balance'].values():
            speaker_stats['avg_alignment'] /= speaker_stats['turns']
        
        # Calculate overall scores
        all_alignments = [entry['vedic_analysis']['overall_alignment'] for entry in dialogue]
        analysis['vedic_integration'] = sum(all_alignments) / len(all_alignments)
        analysis['philosophical_depth'] = len(analysis['concept_coverage']) / 10.0  # Normalize
        analysis['coherence_score'] = min(1.0, len(analysis['concept_coverage']) / 15.0)  # More concepts = better coherence
        
        return analysis
    
    def _generate_dialogue_summary(
        self, 
        dialogue: List[Dict[str, Any]], 
        topic: str
    ) -> str:
        """Generate a summary of the philosophical dialogue."""
        # Extract key points from each perspective
        key_points = []
        
        for entry in dialogue:
            if len(entry['vedic_analysis']['philosophical_concepts_used']) > 0:
                concepts = ', '.join(entry['vedic_analysis']['philosophical_concepts_used'][:2])
                key_points.append(f"{entry['speaker']} emphasized {concepts}")
        
        # Create summary prompt
        dialogue_text = "\n".join([
            f"{entry['speaker']}: {entry['text']}" for entry in dialogue
        ])
        
        summary_prompt = f"""Please provide a concise summary of this Vedic philosophical dialogue on {topic}:

{dialogue_text}

Summary:"""
        
        # Generate summary
        config = VedicGenerationConfig(
            vedic_mode=VedicMode.BALANCED,
            max_new_tokens=200,
            temperature=0.7
        )
        
        result = self.generate_vedic(summary_prompt, config)
        return result['generated_text']
    
    def get_vedic_insights(self, text: str) -> Dict[str, Any]:
        """Extract Vedic insights and wisdom from any text."""
        config = VedicGenerationConfig()
        analysis = self._analyze_vedic_alignment("", text, config)
        
        # Generate insights prompt
        insights_prompt = f"""From a Vedic philosophical perspective, what insights and wisdom can be drawn from this text?

Text: {text}

Vedic Insights:"""
        
        insight_config = VedicGenerationConfig(
            vedic_mode=VedicMode.BALANCED,
            philosophical_depth=4,
            max_new_tokens=250
        )
        
        insights_result = self.generate_vedic(insights_prompt, insight_config)
        
        return {
            'original_text': text,
            'vedic_analysis': analysis,
            'philosophical_insights': insights_result['generated_text'],
            'applicable_principles': analysis['vedic_principles_applied'],
            'suggested_practices': self._suggest_vedic_practices(analysis),
            'related_teachings': self._find_related_teachings(text)
        }
    
    def _suggest_vedic_practices(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest Vedic practices based on analysis."""
        practices = []
        
        if 'dharma' in analysis['vedic_principles_applied']:
            practices.append("Reflect on your dharma (life purpose and duties)")
            practices.append("Study dharmic texts and teachings")
        
        if 'karma' in analysis['vedic_principles_applied']:
            practices.append("Practice mindful action with awareness of consequences")
            practices.append("Engage in karma yoga (selfless action)")
        
        if 'ahimsa' in analysis['vedic_principles_applied']:
            practices.append("Cultivate compassion through loving-kindness meditation")
            practices.append("Practice non-violence in thought, word, and deed")
        
        if 'satya' in analysis['vedic_principles_applied']:
            practices.append("Cultivate truthfulness in all communications")
            practices.append("Practice self-inquiry to understand your true nature")
        
        return practices
    
    def _find_related_teachings(self, text: str) -> List[str]:
        """Find related Vedic teachings based on content."""
        text_lower = text.lower()
        related_teachings = []
        
        # Map keywords to teachings
        teaching_keywords = {
            "duty": ["Bhagavad Gita on svadharma", "Dharmashastra teachings"],
            "action": ["Bhagavad Gita on karma yoga", "Upanishadic teachings on action"],
            "truth": ["Satyameva Jayate principle", "Upanishadic teachings on truth"],
            "peace": ["Shanti mantras", "Ahimsa teachings of Jainism and Buddhism"],
            "wisdom": ["Vedantic teachings", "Upanishadic wisdom"],
            "meditation": ["Dhyana yoga teachings", "Patanjali's Yoga Sutras"]
        }
        
        for keyword, teachings in teaching_keywords.items():
            if keyword in text_lower:
                related_teachings.extend(teachings)
        
        return list(set(related_teachings))  # Remove duplicates
