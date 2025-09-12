# INDRA LLM - Vedic-aligned Decoder-only Transformer

**A production-ready Large Language Model that combines cutting-edge transformer architecture with ancient Vedic wisdom**

*© Divyansh Bharadwaj*

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
![License](https://img.shields.io/badge/license-Custom-green.svg)
![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)

## 🌟 Overview

INDRA LLM is a revolutionary language model that integrates:

- **Advanced Architecture**: Decoder-only transformer with FlashAttention, Grouped Query Attention (GQA), and Mixture of Experts (MoE)
- **Vedic Philosophical Core**: Built-in Vedic memory system, dharmic alignment, and Sanskrit language support  
- **Production-Ready Features**: Efficient KV caching, mixed precision training, distributed computing support
- **Multi-Language Support**: Sanskrit, Hindi, English with specialized Vedic tokenization
- **Scalable Design**: From 2M to 13B+ parameters with modular architecture

## 🏗️ Architecture

### Core Components

- **Transformer Architecture**: GPT-style decoder-only model with modern optimizations
- **FlashAttention**: Memory-efficient attention computation
- **Grouped Query Attention**: Reduced KV cache memory usage
- **Mixture of Experts**: Sparse activation for increased capacity
- **Vedic Integration Layer**: Philosophical alignment and knowledge retrieval

### Vedic Integration Features

- **Vedic Expert Network**: Dedicated expert trained on Vedic corpus
- **Retrieval-Augmented Memory**: Vector database of Vedic concepts
- **Dharmic Reward Model**: Alignment with Vedic principles (Dharma, Ahimsa, Satya, Karma)
- **Sanskrit Support**: Advanced tokenization and processing for Sanskrit texts

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/indra-llm.git
cd indra-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 0. Quick Runs
##### tokenizer creation
```bash
python main.py --mode tokenizer_train --train_data dataset/vedic_texts/ dataset/indian_languages/ dataset/global_languages --output_dir ./tokenizers --tokenizer_type vedic --tokenizer_vocab_size 75000
```

##### Other Pre-train
```bash
python main.py --mode pretrain --train_data ./dataset --val_data ./dataset/validation/ --tokenizer_path ./tokenizers/tokenizer.model --n_embd 768 --n_layer 12 --n_head 12 --batch_size 2 --max_steps 100000 --use_vedic_core --use_flash_attention --output_dir ./checkpoints
```

```bash
python main.py --mode pretrain --train_data ./dataset --tokenizer_type vedic --tokenizer_vocab_size 75000 --n_embd 768 --n_layer 12 --n_head 12 --batch_size 2 --max_steps 100000 --use_vedic_core --use_flash_attention --output_dir ./checkpoints
```

##### Main Pre-Train
```bash
python main.py --mode pretrain --train_data ./dataset --tokenizer_type vedic --tokenizer_vocab_size 262151 --vocab_size 262151 --n_embd 256 --n_layer 4 --n_head 4 --n_kv_head 2 --batch_size 2 --max_steps 100000 --use_vedic_core --use_flash_attention --output_dir ./checkpoints
```

#### 1. Train Tokenizer

```bash
python main.py --mode tokenizer_train \
    --train_data dataset/vedic_texts/ dataset/indian_languages/ \
    --output_dir ./tokenizers \
    --tokenizer_type vedic \
    --tokenizer_vocab_size 75000
```

#### 2. Pre-training

```bash
python main.py --mode pretrain \
    --train_data dataset/vedic_texts/ \
    --val_data dataset/vedic_texts/validation/ \
    --tokenizer_path ./tokenizers/indra_tokenizer.model \
    --n_embd 768 \
    --n_layer 12 \
    --n_head 12 \
    --batch_size 8 \
    --max_steps 100000 \
    --use_vedic_core \
    --use_flash_attention \
    --output_dir ./checkpoints
```

#### 3. Hybrid Training (with Teacher Model)

```bash
python main.py --mode hybrid_pretrain \
    --train_data dataset/ \
    --teacher_model_path Qwen/Qwen1.5-MoE-A2.7B \
    --use_direct_transfer \
    --distillation_alpha 0.7 \
    --output_dir ./checkpoints
```

#### 4. Inference

```bash
python main.py --mode inference \
    --resume_from_checkpoint ./checkpoints/checkpoint-step-100000.pt \
    --tokenizer_path ./tokenizers/indra_tokenizer.model \
    --input_text "What is the meaning of Dharma according to Vedic philosophy?" \
    --max_new_tokens 200 \
    --temperature 0.8
```

## 📁 Project Structure

```
indra-llm/
├── config/
│   ├── __init__.py
│   ├── model_config.py      # Model architecture configuration
│   └── training_config.py   # Training hyperparameters
├── model/
│   ├── __init__.py
│   ├── attention.py         # FlashAttention + GQA implementation
│   ├── moe.py              # Mixture of Experts
│   ├── vedic_core.py       # Vedic integration components
│   ├── transformer.py      # Main transformer architecture
│   └── kv_cache.py         # KV caching system
├── tokenization/
│   ├── __init__.py
│   ├── sentencepiece_tokenizer.py
│   └── vedic_tokenizer.py  # Specialized Vedic tokenizer
├── data/
│   ├── __init__.py
│   ├── dataset.py          # Dataset classes
│   ├── dataloader.py       # Efficient data loading
│   └── vedic_corpus.py     # Vedic text processing
├── training/
│   ├── __init__.py
│   ├── pretrain.py         # Pre-training loop
│   ├── hybrid_pretrain.py  # Hybrid training with teacher
│   ├── sft.py             # Supervised fine-tuning
│   ├── rlhf.py            # RLHF training
│   └── trainer_utils.py    # Training utilities
├── inference/
│   ├── __init__.py
│   ├── generation.py       # Text generation
│   ├── vedic_inference.py  # Vedic-guided inference
│   └── api.py             # API endpoints
├── evaluation/
│   ├── __init__.py
│   ├── benchmarks.py       # Standard benchmarks
│   └── vedic_eval.py      # Vedic knowledge evaluation
├── dataset/
│   ├── vedic_texts/        # Vedic corpus
│   ├── indian_languages/   # Hindi, Sanskrit, etc.
│   └── global_languages/   # English, multilingual
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py       # Model saving/loading
│   ├── metrics.py         # Training metrics
│   └── logging.py         # Logging utilities
├── main.py                 # Main entry point
├── config.yml             # Configuration file
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🔧 Configuration

The model supports extensive configuration through `config.yml`:

### Model Sizes

| Size | Parameters | Layers | Embedding | Heads | KV Heads |
|------|-----------|---------|-----------|-------|----------|
| Tiny | 2M | 4 | 256 | 4 | 2 |
| Small | 10M | 6 | 512 | 8 | 2 |
| Medium | 100M | 12 | 768 | 12 | 4 |
| Large | 500M | 18 | 1024 | 16 | 4 |
| XL | 1B | 20 | 1536 | 24 | 6 |
| XXL | 7B | 32 | 4096 | 32 | 8 |

### Key Features

- **Vedic Integration**: Configurable Vedic memory size and retrieval parameters
- **MoE Configuration**: Flexible expert count and routing strategies
- **Attention Mechanisms**: FlashAttention, GQA, and standard attention options
- **Training Strategies**: Curriculum learning with Vedic phases
- **Multi-Language Support**: Sanskrit, Hindi, English tokenization

## 📊 Training Phases

### 1. Curriculum Learning Phases

1. **Vedic Phase**: Pure Vedic content training (50,000 steps)
2. **General Phase**: Diverse content training (200,000 steps) 
3. **Mixed Phase**: Balanced Vedic + general content

### 2. Training Modes

- **Pre-training**: Standard language modeling on large corpus
- **Hybrid Pre-training**: Knowledge transfer from teacher models (Qwen1.5-MoE)
- **Supervised Fine-tuning**: Instruction following and task-specific training
- **RLHF**: Reinforcement learning with Vedic-aligned rewards

## 🎯 Unique Features

### Vedic Philosophical Integration

- **Dharmic Alignment**: Built-in evaluation of righteousness in responses
- **Karma Awareness**: Understanding of action-consequence relationships
- **Ahimsa Integration**: Non-violence principle in content generation
- **Satya Commitment**: Truthfulness verification mechanisms

### Sanskrit Language Support

- **Advanced Tokenization**: Handles Sanskrit compounds, sandhi rules
- **Script Support**: Both Devanagari and IAST (romanized) scripts
- **Vedic Accents**: Support for Vedic accent marks and special characters
- **Concept Recognition**: Automatic identification of philosophical terms

### Production-Ready Optimizations

- **FlashAttention**: Up to 3x memory efficiency improvement
- **KV Caching**: Efficient inference with cached key-value pairs
- **Mixed Precision**: FP16/BF16 training for reduced memory usage
- **Distributed Training**: DDP and FSDP support for multi-GPU training

## 📈 Performance & Benchmarks

### Standard Benchmarks

- **GLUE/SuperGLUE**: Language understanding tasks
- **HellaSwag**: Commonsense reasoning
- **ARC**: Science question answering
- **MMLU**: Multitask language understanding

### Vedic-Specific Evaluations

- **Vedic Knowledge**: Understanding of Vedic concepts and texts
- **Sanskrit Translation**: Accuracy in Sanskrit-English translation
- **Philosophical Reasoning**: Logical reasoning with Vedic principles
- **Dharmic Alignment**: Adherence to Vedic ethical guidelines

## 🛠️ Advanced Usage

### Custom Model Configuration

```python
from config import ModelConfig
from model import INDRATransformer

# Create custom configuration
config = ModelConfig(
    vocab_size=75000,
    n_embd=1024,
    n_layer=16,
    n_head=16,
    n_kv_head=4,
    use_vedic_core=True,
    use_flash_attention=True,
    use_moe=True
)

# Initialize model
model = INDRATransformer(config)
```

### Vedic-Guided Generation

```python
from inference import InferenceEngine

engine = InferenceEngine(model, tokenizer)

# Generate with Vedic guidance
response = engine.generate_vedic(
    prompt="Explain the concept of dharma",
    vedic_context="upanishads",
    dharmic_weight=1.5
)
```

### Custom Training Loop

```python
from training import PretrainTrainer
from data import VedicDataset

# Create Vedic dataset
dataset = VedicDataset(
    data_path="path/to/vedic/texts",
    tokenizer=tokenizer,
    vedic_weight=2.0
)

# Initialize trainer
trainer = PretrainTrainer(model, config, dataset)

# Custom curriculum
trainer.set_curriculum_phases({
    "vedic": 30000,
    "general": 70000,
    "mixed": 50000
})

# Start training
results = trainer.train()
```

## 🔍 Evaluation & Analysis

### Model Analysis Tools

```bash
# Evaluate on standard benchmarks
python main.py --mode evaluate \
    --resume_from_checkpoint ./checkpoints/model.pt \
    --eval_dataset glue

# Vedic knowledge evaluation  
python main.py --mode evaluate \
    --eval_dataset vedic_knowledge \
    --vedic_concepts_path ./data/vedic_concepts.json

# Generate evaluation report
python utils/generate_report.py \
    --checkpoint ./checkpoints/model.pt \
    --output_dir ./evaluation_results
```

### Interpretability Features

- **Attention Visualization**: Analyze attention patterns on Vedic texts
- **Concept Activation**: Track activation of Vedic philosophical concepts
- **Dharmic Scoring**: Real-time evaluation of response alignment
- **Expert Routing Analysis**: MoE routing patterns for different content types

## 🚀 Deployment

### Model Export Formats

- **PyTorch (.pt)**: Native PyTorch format
- **SafeTensors**: Safe, efficient serialization format  
- **ONNX**: Cross-platform deployment format
- **TensorRT**: NVIDIA GPU optimization

### Serving Options

```bash
# FastAPI server
python inference/api.py --model_path ./checkpoints/model.pt

# Gradio interface
python scripts/gradio_demo.py --checkpoint ./checkpoints/model.pt
```

## 📝 Citation

```bibtex
@software{indra_llm_2024,
  title={INDRA LLM: Vedic-aligned Decoder-only Transformer},
  author={Bharadwaj, Divyansh},
  year={2024},
  url={https://github.com/your-username/indra-llm}
}
```

## 📄 License

© Divyansh Bharadwaj. All rights reserved.

This project is licensed under a custom license. See the `LICENSE` file for details.

## 🤝 Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

### Areas for Contribution

- **Vedic Corpus Expansion**: Additional Sanskrit texts and commentaries
- **Language Support**: More Indic languages (Tamil, Telugu, Bengali)
- **Evaluation Metrics**: Novel benchmarks for philosophical reasoning
- **Optimization**: Performance improvements and memory optimizations

## 🙏 Acknowledgments

- **Vedic Tradition**: For the timeless wisdom that guides this project
- **Open Source Community**: For the foundational tools and libraries
- **Research Community**: For advancing transformer architecture
- **Sanskrit Scholars**: For linguistic expertise and guidance

## 📞 Contact

For questions, suggestions, or collaborations:

- **Author**: Divyansh Bharadwaj
- **Email**: [divyanshbharadwaj008@gmail.com]
- **GitHub**: [@divyanshbharadwaj]
- **Project**: [https://github.com/DivyanshBharadwaj/indra-llm]

---

*"As knowledge is unified in the Vedas, so shall wisdom be unified in artificial intelligence."*

---
