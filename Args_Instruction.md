# INDRA LLM: Complete Tutorial and Reference Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Core Concepts](#core-concepts)
4. [Command-Line Interface Reference](#command-line-interface-reference)
5. [Training Modes](#training-modes)
6. [Model Configuration](#model-configuration)
7. [Tokenizer Management](#tokenizer-management)
8. [Data Management](#data-management)
9. [Inference and Generation](#inference-and-generation)
10. [Evaluation](#evaluation)
11. [Advanced Features](#advanced-features)
12. [Best Practices](#best-practices)
13. [Troubleshooting](#troubleshooting)
14. [Examples](#examples)

---

## Introduction

INDRA LLM is a Vedic-aligned Decoder-only Transformer that combines modern language modeling techniques with ancient wisdom traditions. This comprehensive guide will walk you through every aspect of using the INDRA LLM system, from basic setup to advanced configurations.

### Key Features

- **Vedic Integration**: Native support for Sanskrit and Vedic texts with specialized tokenization
- **Hierarchical Reasoning**: Multi-level reasoning capabilities inspired by Vedic philosophy
- **Mixture of Experts (MoE)**: Efficient scaling with expert routing
- **Modern Optimizations**: FlashAttention, KV caching, and distributed training
- **Flexible Training Modes**: Pre-training, hybrid training, supervised fine-tuning, and RLHF
- **Advanced Inference**: Reasoning-enabled generation with confidence scoring

---

## Getting Started

### Quick Start

The simplest way to get started with INDRA LLM is to run inference with a pre-trained model:

```bash
python main.py --mode inference \
    --input_text "What is the meaning of dharma?" \
    --resume_from_checkpoint ./checkpoints/model.pt \
    --tokenizer_type vedic
```

### Basic Training

To start training from scratch:

```bash
python main.py --mode pretrain \
    --train_data ./data/corpus.txt \
    --tokenizer_type vedic \
    --output_dir ./my_model
```

---

## Core Concepts

### The INDRA Architecture

INDRA LLM is built on several foundational concepts:

1. **Vedic Core**: Integration of Vedic principles and Sanskrit understanding
2. **Hierarchical Reasoning**: Multi-level reasoning that mirrors Vedic philosophical frameworks
3. **Adaptive Learning**: Curriculum learning that progresses from Vedic texts to general knowledge
4. **Efficient Scaling**: MoE and attention optimizations for large-scale training

### Training Philosophy

INDRA follows a three-phase training approach:

1. **Vedic Phase**: Focus on Sanskrit and Vedic texts
2. **General Phase**: Expansion to general knowledge
3. **Alignment Phase**: Fine-tuning for instruction following and safety

---

## Command-Line Interface Reference

### Mode Selection

The `--mode` argument determines the primary operation:

| Mode | Description | Required Arguments |
|------|-------------|-------------------|
| `train` / `pretrain` | Standard pre-training | `--train_data` |
| `hybrid_pretrain` | Training with teacher model | `--train_data`, `--teacher_model_path` |
| `sft` | Supervised fine-tuning | `--train_data`, `--resume_from_checkpoint` |
| `rlhf` | Reinforcement learning from human feedback | (Not yet implemented) |
| `inference` | Text generation | `--input_text`, `--resume_from_checkpoint` |
| `evaluate` | Model evaluation | `--resume_from_checkpoint` |
| `tokenizer_train` | Train tokenizer | `--train_data` |

### Essential Arguments

#### Model Architecture
- `--vocab_size`: Vocabulary size (default: 50257)
- `--n_positions`: Maximum sequence length (default: 2048)
- `--n_embd`: Embedding dimension (default: 768)
- `--n_layer`: Number of transformer layers (default: 12)
- `--n_head`: Number of attention heads (default: 12)
- `--n_kv_head`: Number of key-value heads for grouped-query attention (default: 4)

#### Training Configuration
- `--batch_size`: Training batch size (default: 8)
- `--max_steps`: Maximum training steps (default: 100000)
- `--max_lr`: Maximum learning rate (default: 6e-4)
- `--warmup_steps`: Learning rate warmup steps (default: 2000)

#### Data and Output
- `--train_data`: Training data paths (multiple files supported)
- `--val_data`: Validation data paths
- `--output_dir`: Directory for checkpoints and logs (default: ./checkpoints)

---

## Training Modes

### Pre-training Mode

Pre-training is the foundation phase where the model learns language understanding from large corpora.

#### Basic Pre-training

```bash
python main.py --mode pretrain \
    --train_data ./data/general_corpus.txt \
    --val_data ./data/validation.txt \
    --vocab_size 50000 \
    --n_layer 24 \
    --n_embd 1024 \
    --batch_size 16 \
    --max_steps 500000 \
    --output_dir ./models/base_pretrained
```

#### Vedic Pre-training

For Vedic-specific pre-training with specialized tokenization:

```bash
python main.py --mode pretrain \
    --train_data ./data/vedic_corpus/ \
    --vedic_data ./data/sanskrit_texts/ \
    --tokenizer_type vedic \
    --use_vedic_core \
    --vedic_memory_size 20000 \
    --vedic_retrieval_top_k 10 \
    --output_dir ./models/vedic_pretrained
```

### Hybrid Pre-training Mode

Hybrid pre-training leverages a teacher model to accelerate learning:

```bash
python main.py --mode hybrid_pretrain \
    --train_data ./data/mixed_corpus.txt \
    --teacher_model_path "Qwen/Qwen1.5-MoE-A2.7B" \
    --use_direct_transfer \
    --distillation_alpha 0.7 \
    --distillation_temperature 4.0 \
    --vedic_phase_steps 30000 \
    --general_phase_steps 150000 \
    --vedic_data_ratio 0.3
```

#### Key Parameters for Hybrid Training

- `--teacher_model_path`: Path or HuggingFace model name for the teacher
- `--distillation_alpha`: Weight for distillation loss (0.0-1.0)
- `--distillation_temperature`: Temperature for knowledge distillation
- `--use_direct_transfer`: Enable direct parameter transfer from teacher

### Supervised Fine-tuning (SFT)

SFT adapts a pre-trained model for instruction following:

```bash
python main.py --mode sft \
    --train_data ./data/instruction_dataset.json \
    --resume_from_checkpoint ./models/pretrained/model.pt \
    --max_lr 2e-5 \
    --max_steps 10000 \
    --batch_size 4 \
    --gradient_accumulation_steps 16
```

---

## Model Configuration

### Architecture Options

#### Standard Configuration
```bash
# Small model (similar to GPT-2 small)
--vocab_size 50257 --n_layer 12 --n_embd 768 --n_head 12

# Medium model
--vocab_size 50257 --n_layer 24 --n_embd 1024 --n_head 16

# Large model
--vocab_size 50257 --n_layer 36 --n_embd 1280 --n_head 20
```

#### Mixture of Experts (MoE)
```bash
# Enable MoE for efficient scaling
--use_moe \
--num_experts 8 \
--expert_top_k 2
```

#### Hierarchical Reasoning
```bash
# Enable multi-level reasoning
--use_hierarchical_reasoning \
--reasoning_depth 3 \
--enable_reasoning_in_training \
--reasoning_weight 0.1
```

### Performance Optimizations

#### Attention Optimizations
```bash
# Use FlashAttention and KV caching
--use_flash_attention \
--use_kv_cache
```

#### Memory Optimizations
```bash
# Enable gradient checkpointing for memory efficiency
--gradient_checkpointing
```

#### Mixed Precision Training
```bash
# Use BF16 for modern GPUs
--use_bf16

# Use FP16 for older GPUs
--use_fp16
```

### Distributed Training

#### Data Parallel Training
```bash
# Single-node multi-GPU
torchrun --nproc_per_node=4 main.py \
    --mode pretrain \
    --use_ddp \
    --train_data ./data/corpus.txt
```

#### Fully Sharded Data Parallel (FSDP)
```bash
# For very large models
torchrun --nproc_per_node=8 main.py \
    --mode pretrain \
    --use_fsdp \
    --train_data ./data/corpus.txt
```

---

## Tokenizer Management

### Vedic Tokenizer

The Vedic tokenizer is optimized for Sanskrit and Indian languages:

```bash
# Train a new Vedic tokenizer
python main.py --mode tokenizer_train \
    --tokenizer_type vedic \
    --train_data ./data/sanskrit_corpus/ \
    --tokenizer_vocab_size 75000 \
    --output_dir ./tokenizers/vedic_v1
```

#### Vedic Tokenizer Features

- **Sanskrit Support**: Proper handling of Devanagari script
- **Compound Word Recognition**: Understanding of Sanskrit compound structures
- **Vedic Concepts**: Special tokens for philosophical and spiritual concepts
- **Multi-script Support**: Support for multiple Indian scripts

### SentencePiece Tokenizer

For general-purpose tokenization:

```bash
# Train SentencePiece tokenizer
python main.py --mode tokenizer_train \
    --tokenizer_type sentencepiece \
    --train_data ./data/general_corpus.txt \
    --tokenizer_vocab_size 50000
```

### Using Pre-trained Tokenizers

```bash
# Use existing tokenizer
--tokenizer_path ./tokenizers/my_tokenizer.model \
--tokenizer_type sentencepiece
```

---

## Data Management

### Data Formats

INDRA supports multiple data formats:

#### Text Files (.txt)
```bash
--train_data ./data/corpus.txt \
--data_type txt
```

#### JSON Lines
```bash
--train_data ./data/dataset.jsonl \
--data_type json
```

#### HuggingFace Datasets
```bash
--train_data "dataset_name" \
--data_type hf
```

#### Auto-detection
```bash
--data_type auto  # Automatically detects format
```

### Multi-file Training

```bash
# Multiple training files
--train_data ./data/file1.txt ./data/file2.txt ./data/directory/
```

### Vedic Data Integration

```bash
# Combine general and Vedic data
--train_data ./data/general_corpus.txt \
--vedic_data ./data/sanskrit_texts/ ./data/upanishads/
```

---

## Inference and Generation

### Basic Text Generation

```bash
python main.py --mode inference \
    --input_text "Explain the concept of dharma" \
    --resume_from_checkpoint ./models/trained/model.pt \
    --max_new_tokens 200 \
    --temperature 0.8 \
    --top_p 0.9
```

### Generation Parameters

#### Sampling Control
- `--temperature`: Controls randomness (0.1-2.0)
  - Lower values (0.1-0.7): More focused, deterministic
  - Higher values (1.0-2.0): More creative, diverse
- `--top_k`: Consider only top-k tokens (1-100)
- `--top_p`: Nucleus sampling threshold (0.1-1.0)
- `--repetition_penalty`: Avoid repetition (1.0-1.5)

#### Length Control
- `--max_new_tokens`: Maximum tokens to generate
- `--do_sample`: Enable sampling (vs greedy decoding)

### Hierarchical Reasoning Inference

Enable advanced reasoning capabilities:

```bash
python main.py --mode inference \
    --input_text "How does karma relate to dharma?" \
    --resume_from_checkpoint ./models/reasoning_model.pt \
    --enable_reasoning_inference \
    --reasoning_depth 3 \
    --return_reasoning_chain
```

#### Reasoning Parameters
- `--reasoning_depth`: How many reasoning levels to use (1-5)
- `--return_reasoning_chain`: Show step-by-step reasoning process

---

## Evaluation

### Standard Benchmarks

```bash
python main.py --mode evaluate \
    --resume_from_checkpoint ./models/trained/model.pt \
    --eval_batch_size 16
```

### Custom Dataset Evaluation

```bash
python main.py --mode evaluate \
    --resume_from_checkpoint ./models/trained/model.pt \
    --eval_dataset ./data/test_set.jsonl \
    --eval_batch_size 8
```

---

## Advanced Features

### Mixture of Experts (MoE)

MoE allows efficient scaling by routing different inputs to specialized experts:

```bash
python main.py --mode pretrain \
    --use_moe \
    --num_experts 16 \
    --expert_top_k 2 \
    --train_data ./data/large_corpus/
```

#### MoE Configuration Guidelines

- **num_experts**: Start with 8, scale to 16+ for larger models
- **expert_top_k**: Usually 1-2, higher values increase computation
- **Load Balancing**: Automatically handled by the routing algorithm

### Hierarchical Reasoning System

The hierarchical reasoning system implements multi-level thinking:

```bash
# Training with reasoning
python main.py --mode pretrain \
    --use_hierarchical_reasoning \
    --reasoning_depth 3 \
    --enable_reasoning_in_training \
    --reasoning_weight 0.15
```

#### Reasoning Levels

1. **Surface Level**: Immediate pattern recognition
2. **Semantic Level**: Meaning and context understanding
3. **Conceptual Level**: Abstract concept relationships
4. **Philosophical Level**: Deep principle application
5. **Transcendental Level**: Highest-order reasoning

### Vedic Core Integration

```bash
# Full Vedic integration
python main.py --mode pretrain \
    --use_vedic_core \
    --vedic_memory_size 50000 \
    --vedic_retrieval_top_k 10 \
    --vedic_data ./data/vedic_corpus/
```

#### Vedic Memory System

- **vedic_memory_size**: Number of Vedic concepts to store
- **vedic_retrieval_top_k**: Retrieved concepts per query
- Automatically indexes Sanskrit terms and philosophical concepts

---

## Best Practices

### Training Recommendations

#### For Sanskrit/Vedic Focus
1. Start with Vedic tokenizer training
2. Use curriculum learning (Vedic → General)
3. Enable Vedic core integration
4. Consider hierarchical reasoning for philosophical texts

```bash
# Recommended Vedic training pipeline
python main.py --mode tokenizer_train \
    --tokenizer_type vedic \
    --train_data ./data/sanskrit_corpus/ \
    --output_dir ./tokenizers/

python main.py --mode hybrid_pretrain \
    --tokenizer_type vedic \
    --use_vedic_core \
    --use_hierarchical_reasoning \
    --vedic_data ./data/vedic_texts/ \
    --train_data ./data/general_corpus.txt \
    --teacher_model_path "Qwen/Qwen1.5-MoE-A2.7B"
```

#### For General-purpose Models
1. Use standard tokenizer with large vocabulary
2. Enable MoE for efficiency
3. Use hybrid training for faster convergence

```bash
# Recommended general training pipeline
python main.py --mode hybrid_pretrain \
    --use_moe \
    --num_experts 8 \
    --train_data ./data/diverse_corpus/ \
    --teacher_model_path "microsoft/DialoGPT-medium" \
    --use_flash_attention \
    --use_bf16
```

### Hardware Recommendations

#### Single GPU (8-16GB)
```bash
# Memory-efficient configuration
--batch_size 2 \
--micro_batch_size 1 \
--gradient_accumulation_steps 32 \
--gradient_checkpointing \
--use_fp16
```

#### Multi-GPU Setup (32GB+ each)
```bash
# Distributed training configuration
--batch_size 8 \
--micro_batch_size 2 \
--use_ddp \
--use_bf16 \
--use_flash_attention
```

#### High-Memory Setup (80GB+ each)
```bash
# Large model configuration
--batch_size 16 \
--n_layer 48 \
--n_embd 2048 \
--n_head 32 \
--use_fsdp \
--use_moe \
--num_experts 16
```

---

## Examples

### Example 1: Training a Vedic Scholar Model

Goal: Create a model specialized in Sanskrit texts and Vedic philosophy.

```bash
# Step 1: Train Vedic tokenizer
python main.py --mode tokenizer_train \
    --tokenizer_type vedic \
    --train_data ./data/mahabharata.txt ./data/ramayana.txt ./data/upanishads/ \
    --tokenizer_vocab_size 100000 \
    --output_dir ./tokenizers/vedic_scholar

# Step 2: Pre-train with hierarchical reasoning
python main.py --mode pretrain \
    --tokenizer_type vedic \
    --tokenizer_path ./tokenizers/vedic_scholar/vedic_tokenizer.model \
    --use_vedic_core \
    --use_hierarchical_reasoning \
    --reasoning_depth 4 \
    --vedic_data ./data/sanskrit_complete/ \
    --train_data ./data/philosophy_texts.txt \
    --n_layer 32 \
    --n_embd 1536 \
    --vedic_memory_size 100000 \
    --max_steps 200000 \
    --output_dir ./models/vedic_scholar

# Step 3: Fine-tune for instruction following
python main.py --mode sft \
    --train_data ./data/vedic_qa_pairs.json \
    --resume_from_checkpoint ./models/vedic_scholar/model.pt \
    --max_lr 1e-5 \
    --max_steps 5000 \
    --output_dir ./models/vedic_scholar_chat
```

### Example 2: Efficient General-purpose Model

Goal: Create a general-purpose model using MoE for efficiency.

```bash
# Hybrid training with MoE
python main.py --mode hybrid_pretrain \
    --use_moe \
    --num_experts 16 \
    --expert_top_k 2 \
    --teacher_model_path "microsoft/DialoGPT-large" \
    --train_data ./data/diverse_corpus/ \
    --n_layer 24 \
    --n_embd 1024 \
    --use_flash_attention \
    --use_bf16 \
    --batch_size 12 \
    --max_steps 300000 \
    --output_dir ./models/general_moe
```

### Example 3: Research Model with Advanced Reasoning

Goal: Create a model for research applications with deep reasoning.

```bash
# Advanced reasoning model
python main.py --mode pretrain \
    --use_hierarchical_reasoning \
    --reasoning_depth 5 \
    --enable_reasoning_in_training \
    --reasoning_weight 0.2 \
    --train_data ./data/academic_papers/ ./data/research_corpus/ \
    --n_layer 40 \
    --n_embd 2048 \
    --n_head 32 \
    --use_flash_attention \
    --gradient_checkpointing \
    --max_steps 400000 \
    --output_dir ./models/research_reasoning
```

### Example 4: Inference with Different Strategies

#### Creative Generation
```bash
python main.py --mode inference \
    --input_text "Write a story about a sage who discovers..." \
    --resume_from_checkpoint ./models/creative/model.pt \
    --temperature 1.2 \
    --top_p 0.9 \
    --max_new_tokens 500 \
    --do_sample
```

#### Precise Q&A
```bash
python main.py --mode inference \
    --input_text "What are the four noble truths?" \
    --resume_from_checkpoint ./models/vedic_scholar/model.pt \
    --temperature 0.3 \
    --top_k 10 \
    --max_new_tokens 200 \
    --enable_reasoning_inference \
    --return_reasoning_chain
```

---

## Configuration Files

### YAML Configuration

Instead of long command lines, use YAML configuration files:

```yaml
# config.yml
model:
  vocab_size: 75000
  n_positions: 4096
  n_embd: 1536
  n_layer: 28
  n_head: 24
  use_moe: true
  num_experts: 12
  use_hierarchical_reasoning: true

training:
  batch_size: 16
  max_steps: 500000
  max_lr: 4e-4
  warmup_steps: 5000
  use_bf16: true
  use_flash_attention: true

data:
  train_data: ["./data/corpus1.txt", "./data/corpus2.txt"]
  vedic_data: ["./data/vedic_texts/"]
  tokenizer_type: "vedic"
```

Use with:
```bash
python main.py --mode pretrain --model_config config.yml
```

---

## Monitoring and Logging

### Weights & Biases Integration

```bash
python main.py --mode pretrain \
    --use_wandb \
    --wandb_project "indra-experiments" \
    --wandb_run_name "vedic_large_v1" \
    --train_data ./data/corpus.txt
```

### Local Logging

```bash
# Detailed logging
--log_level DEBUG \
--logging_steps 50
```

---

## Troubleshooting

### Common Issues and Solutions

#### Out of Memory Errors
1. Reduce batch size: `--batch_size 1 --micro_batch_size 1`
2. Enable gradient checkpointing: `--gradient_checkpointing`
3. Use gradient accumulation: `--gradient_accumulation_steps 32`
4. Switch to FP16: `--use_fp16`

#### Slow Training
1. Enable FlashAttention: `--use_flash_attention`
2. Use mixed precision: `--use_bf16`
3. Increase batch size if memory allows
4. Use distributed training: `--use_ddp`

#### Poor Generation Quality
1. Adjust temperature: Lower for focused, higher for creative
2. Check tokenizer compatibility
3. Verify model checkpoint loading
4. Try different sampling strategies

#### Convergence Issues
1. Adjust learning rate: Start with `--max_lr 3e-4`
2. Increase warmup steps: `--warmup_steps 10000`
3. Check gradient accumulation setup
4. Verify data preprocessing

---

## Performance Tuning

### Memory Optimization Strategies

```bash
# Maximum memory efficiency
python main.py --mode pretrain \
    --batch_size 1 \
    --micro_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --gradient_checkpointing \
    --use_fp16 \
    --max_seq_length 1024
```

### Speed Optimization

```bash
# Maximum training speed
python main.py --mode pretrain \
    --use_flash_attention \
    --use_kv_cache \
    --use_bf16 \
    --batch_size 32 \
    --use_ddp
```

### Quality vs Efficiency Trade-offs

| Priority | Configuration | Use Case |
|----------|---------------|----------|
| Quality | Large model, no MoE, reasoning enabled | Research, philosophical applications |
| Efficiency | MoE, FlashAttention, mixed precision | Production, real-time applications |
| Balanced | Medium model, selective MoE, optimizations | General-purpose applications |

---

## Advanced Workflows

### Multi-stage Training Pipeline

```bash
#!/bin/bash
# Complete training pipeline

# Stage 1: Tokenizer training
python main.py --mode tokenizer_train \
    --tokenizer_type vedic \
    --train_data ./data/full_corpus/ \
    --output_dir ./pipeline/tokenizer

# Stage 2: Vedic pre-training
python main.py --mode pretrain \
    --tokenizer_type vedic \
    --tokenizer_path ./pipeline/tokenizer/vedic_tokenizer.model \
    --vedic_data ./data/vedic_texts/ \
    --use_vedic_core \
    --max_steps 100000 \
    --output_dir ./pipeline/vedic_pretrained

# Stage 3: General pre-training
python main.py --mode pretrain \
    --resume_from_checkpoint ./pipeline/vedic_pretrained/model.pt \
    --train_data ./data/general_corpus/ \
    --max_steps 200000 \
    --output_dir ./pipeline/general_pretrained

# Stage 4: Instruction tuning
python main.py --mode sft \
    --resume_from_checkpoint ./pipeline/general_pretrained/model.pt \
    --train_data ./data/instruction_dataset.json \
    --max_steps 10000 \
    --output_dir ./pipeline/final_model
```

### Curriculum Learning

```bash
# Automated curriculum learning
python main.py --mode hybrid_pretrain \
    --vedic_phase_steps 50000 \    # Pure Vedic training
    --general_phase_steps 200000 \ # General knowledge
    --vedic_data_ratio 0.2 \       # 20% Vedic in mixed phase
    --train_data ./data/general/ \
    --vedic_data ./data/vedic/
```

---

## Integration Examples

### Research Workflow

For academic research involving Sanskrit texts:

```bash
# 1. Prepare specialized tokenizer
python main.py --mode tokenizer_train \
    --tokenizer_type vedic \
    --train_data ./research_data/sanskrit_corpus/ \
    --tokenizer_vocab_size 150000

# 2. Train research model
python main.py --mode pretrain \
    --tokenizer_type vedic \
    --use_hierarchical_reasoning \
    --reasoning_depth 5 \
    --vedic_data ./research_data/primary_sources/ \
    --train_data ./research_data/secondary_sources/ \
    --n_layer 36 \
    --n_embd 2048

# 3. Interactive research queries
python main.py --mode inference \
    --input_text "Analyze the concept of moksha in Advaita Vedanta" \
    --enable_reasoning_inference \
    --return_reasoning_chain \
    --temperature 0.5
```

### Production Deployment

For production applications requiring efficiency:

```bash
# Optimized production model
python main.py --mode hybrid_pretrain \
    --use_moe \
    --num_experts 8 \
    --expert_top_k 2 \
    --use_flash_attention \
    --use_kv_cache \
    --use_bf16 \
    --teacher_model_path "microsoft/DialoGPT-medium" \
    --batch_size 24 \
    --output_dir ./production_model
```

---

## Monitoring and Debugging

### Training Metrics

Key metrics to monitor during training:

1. **Loss Curves**: Train/validation loss
2. **Learning Rate**: Should follow warmup schedule
3. **Gradient Norms**: Check for gradient explosion
4. **Memory Usage**: Monitor GPU utilization
5. **Throughput**: Tokens per second

### Debugging Commands

```bash
# Debug mode with detailed logging
python main.py --mode pretrain \
    --log_level DEBUG \
    --logging_steps 10 \
    --eval_steps 100 \
    --train_data ./data/small_test.txt \
    --max_steps 1000
```

### Performance Profiling

```bash
# Profile memory and compute usage
python -m torch.utils.bottleneck main.py \
    --mode pretrain \
    --batch_size 4 \
    --max_steps 100
```

---

## Conclusion

This tutorial has covered the comprehensive usage of INDRA LLM, from basic training to advanced reasoning capabilities. The system's flexibility allows for customization based on specific requirements, whether for Sanskrit scholarship, general-purpose language modeling, or research applications.

### Key Takeaways

1. **Start Simple**: Begin with basic configurations and gradually add complexity
2. **Use Appropriate Hardware**: Match configuration to available resources
3. **Monitor Training**: Use logging and evaluation to track progress
4. **Experiment**: The modular design encourages experimentation with different components
5. **Leverage Vedic Features**: The unique Vedic integration offers capabilities not found in standard LLMs

### Next Steps

1. Experiment with different model sizes and configurations
2. Explore the hierarchical reasoning capabilities
3. Develop custom evaluation metrics for your use case
4. Contribute to the development of new Vedic language understanding features

For additional support and advanced configurations, refer to the individual module documentation and consider contributing to the INDRA LLM project.

---

*© Divyansh Bharadwaj - INDRA LLM Project*
