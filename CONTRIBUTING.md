# Contributing to INDRA LLM

*"यज्ञार्थात्कर्मणोऽन्यत्र लोकोऽयं कर्मबन्धनः" - Work performed as service to others liberates; all other work binds.*

We welcome contributions that align with INDRA LLM's mission of integrating ancient Vedic wisdom with cutting-edge AI technology.

## 🌟 Philosophy of Contribution

All contributions should embody Vedic principles:
- **Dharma**: Righteous purpose in advancing beneficial AI
- **Ahimsa**: Non-harmful applications and ethical considerations
- **Satya**: Truthful documentation and honest reporting of results
- **Karma**: Mindful action with consideration of consequences

## 🛠️ How to Contribute

### 1. Areas for Contribution

#### Core Architecture
- **Model Optimizations**: Improvements to attention mechanisms, MoE routing
- **Memory Efficiency**: Better KV caching, gradient checkpointing
- **Training Optimization**: Curriculum learning improvements, loss functions

#### Vedic Integration
- **Philosophical Concepts**: Enhanced Vedic concept representation
- **Sanskrit Processing**: Better tokenization and linguistic handling
- **Ethical Alignment**: Improved dharmic decision-making algorithms
- **Knowledge Retrieval**: Enhanced Vedic memory systems

#### Language Support
- **Indic Languages**: Support for Tamil, Telugu, Bengali, etc.
- **Script Support**: Better handling of different writing systems
- **Translation**: Sanskrit-to-modern language translation improvements

#### Evaluation & Benchmarks
- **Philosophical Reasoning**: New benchmarks for ethical decision-making
- **Sanskrit Understanding**: Comprehensive Sanskrit comprehension tests
- **Cross-cultural Wisdom**: Integration with other wisdom traditions

### 2. Getting Started

#### Prerequisites
```bash
# Install development dependencies
pip install -e ".[dev,all]"

# Setup pre-commit hooks
pre-commit install
```

#### Development Setup
```bash
# Clone the repository
git clone https://github.com/DivyanshBharadwaj/indra-llm.git
cd indra-llm

# Create development branch
git checkout -b feature/your-feature-name

# Install in development mode
pip install -e .
```

### 3. Code Standards

#### Code Quality
- **Type Hints**: All functions must have comprehensive type annotations
- **Docstrings**: Use Google-style docstrings for all public methods
- **Error Handling**: Robust exception handling with meaningful messages
- **Testing**: Unit tests for all new functionality

#### Vedic Code Principles
- **Clarity (Satya)**: Code should be clear and truthful in its intent
- **Non-harm (Ahimsa)**: Avoid harmful patterns or resource waste
- **Righteousness (Dharma)**: Follow ethical coding practices
- **Mindful Action (Karma)**: Consider the consequences of code changes

#### Example Code Structure
```python
def vedic_aligned_function(
    input_data: torch.Tensor,
    vedic_context: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Process input with Vedic philosophical alignment.
    
    Args:
        input_data: Input tensor to process
        vedic_context: Optional Vedic contextual information
        
    Returns:
        Tuple of (processed_output, vedic_metrics)
        
    Raises:
        ValueError: If input_data is invalid
        RuntimeError: If processing fails
    """
    # Implementation follows dharmic principles
    # Clear, non-harmful, truthful processing
    pass
```

### 4. Testing Guidelines

#### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Vedic Alignment Tests**: Philosophical principle adherence
- **Performance Tests**: Efficiency and scalability testing

#### Running Tests
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/vedic/
pytest tests/performance/

# Run with coverage
pytest --cov=indra_llm --cov-report=html
```

### 5. Documentation Standards

#### Code Documentation
- All public APIs must have comprehensive docstrings
- Include examples for complex functions
- Document Vedic philosophical rationale where applicable

#### Philosophical Documentation
- Explain the Vedic principles behind design decisions
- Provide Sanskrit terms with proper transliteration
- Include references to source texts when applicable

### 6. Submission Process

#### Pull Request Guidelines
1. **Branch Naming**: Use descriptive names like `feature/vedic-memory-enhancement`
2. **Commit Messages**: Follow conventional commits format
3. **Description**: Explain the philosophical and technical rationale
4. **Testing**: Ensure all tests pass and add new tests for new features
5. **Documentation**: Update relevant documentation

#### PR Template
```markdown
## Summary
Brief description of changes and their purpose.

## Vedic Alignment
How do these changes align with Vedic principles?

## Technical Details
- Architecture changes
- Performance implications
- Breaking changes (if any)

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Vedic alignment tests pass
- [ ] Performance tests (if applicable)

## Documentation
- [ ] Code documented
- [ ] README updated (if needed)
- [ ] Philosophical rationale explained

## Checklist
- [ ] Code follows project standards
- [ ] All tests pass
- [ ] No ethical concerns
- [ ] Respects Vedic principles
```

### 7. Review Process

#### Technical Review
- Code quality and efficiency
- Architectural consistency
- Test coverage and quality
- Documentation completeness

#### Philosophical Review
- Alignment with Vedic principles
- Ethical implications assessment
- Cultural sensitivity verification
- Sanskrit accuracy (for language-related contributions)

### 8. Community Guidelines

#### Respectful Collaboration
- Treat all contributors with respect and kindness
- Value diverse perspectives while maintaining philosophical alignment
- Provide constructive feedback focused on improvement
- Acknowledge the contributions of others

#### Knowledge Sharing
- Share learnings from Vedic texts and philosophy
- Explain Sanskrit concepts for non-Sanskrit speakers
- Provide context for philosophical decisions
- Mentor newcomers to both AI and Vedic concepts

### 9. Specialized Contributions

#### Sanskrit Scholars
- Textual accuracy verification
- Concept relationship validation
- Translation quality assessment
- Philosophical interpretation guidance

#### AI Researchers
- Architectural improvements
- Training optimization
- Evaluation methodology
- Performance enhancement

#### Philosophers & Ethicists
- Ethical framework development
- Philosophical consistency checking
- Cross-tradition dialogue facilitation
- Wisdom integration strategies

### 10. Recognition

#### Contributor Acknowledgment
- All contributors will be acknowledged in the project
- Significant contributions will be highlighted in releases
- Academic contributors will be included in research publications
- Vedic scholars will be specially recognized for wisdom contributions

#### Spiritual Credit
*"न हि ज्ञानेन सदृशं पवित्रमिह विद्यते" - There is nothing in this world as purifying as knowledge.*

Contributors to INDRA LLM participate in the noble work of preserving and sharing ancient wisdom through modern technology.

## 🚀 Getting Help

### Resources
- **Documentation**: Full API and philosophy documentation
- **Discord**: Join our community discussions
- **Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas

### Contact
- **Lead Developer**: Divyansh Bharadwaj (your-email@domain.com)
- **Sanskrit Advisor**: [Sanskrit Scholar Name]
- **Ethics Committee**: [Ethics Team]

### Learning Resources
- **Vedic Philosophy Primer**: Introduction for AI practitioners
- **Sanskrit Basics**: Essential Sanskrit for contributors
- **AI Ethics**: Vedic approaches to AI alignment
- **Technical Docs**: Architecture and implementation guides

---

*"सहनाववतु सहनौ भुनक्तु सहवीर्यं करवावहै।*
*तेजस्विनावधीतमस्तु मा विद्विषावहै॥"*

*"May we together be protected, may we together be nourished,*
*may we work together with vigor, may our study be enlightening,*
*may we not quarrel with each other."*

Thank you for contributing to the synthesis of ancient wisdom and modern intelligence!
