"""
Main entry point for INDRA LLM - Vedic-aligned Decoder-only Transformer
Modified to use streaming datasets for memory efficiency
(c) Divyansh Bharadwaj
"""

import argparse
import logging
import os
import glob
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import ModelConfig, TrainingConfig
from model import INDRATransformer, add_hierarchical_reasoning_to_model
from tokenization import SentencePieceTokenizer, VedicTokenizer, VedicTokenizerManager
from data import StreamingINDRADataset, StreamingVedicDataset, StreamingInstructionDataset, create_streaming_dataset
from training import PretrainTrainer, HybridPretrainTrainer, SFTTrainer, RLHFTrainer
from inference import InferenceEngine, VedicInferenceEngine
from evaluation import Evaluator, VedicEvaluator

def debug_data_pipeline(args, train_dataset_obj):
    """A helper function to debug the data loading process."""
    print("\n" + "="*50)
    print("--- STREAMING DATA PIPELINE DEBUGGER ---")
    
    # 1. Check paths from command-line arguments
    train_path = args.train_data[0] if args.train_data else "Not Provided"
    vedic_path = args.vedic_data[0] if args.vedic_data else "Not Provided"

    print(f"\n[1] Checking General Training Data Path: '{train_path}'")
    if os.path.exists(train_path):
        print(f"    -> Path EXISTS.")
        general_files = glob.glob(os.path.join(train_path, '**/*.txt'), recursive=True)
        print(f"    -> Found {len(general_files)} '.txt' files in this directory.")
    else:
        print(f"    -> ERROR: Path DOES NOT EXIST.")

    print(f"\n[2] Checking Vedic Training Data Path: '{vedic_path}'")
    if os.path.exists(vedic_path):
        print(f"    -> Path EXISTS.")
        vedic_files = glob.glob(os.path.join(vedic_path, '**/*.txt'), recursive=True)
        print(f"    -> Found {len(vedic_files)} '.txt' files in this directory.")
    else:
        print(f"    -> ERROR: Path DOES NOT EXIST.")

    # 2. Check the created dataset object itself
    print(f"\n[3] Checking the created Streaming Dataset object:")
    if train_dataset_obj:
        try:
            print(f"    -> Dataset type: {type(train_dataset_obj).__name__}")
            print(f"    -> Total files to process: {train_dataset_obj.total_files}")
            print(f"    -> Batch size (MB): {train_dataset_obj.batch_size_mb}")
            print(f"    -> Examples per batch: {train_dataset_obj.examples_per_batch}")
            
            # Test iterator (just peek at first item)
            iterator = iter(train_dataset_obj)
            first_item = next(iterator)
            print(f"    -> Successfully created iterator and got first item")
            print(f"    -> First item keys: {list(first_item.keys()) if isinstance(first_item, dict) else 'Not a dict'}")
        except Exception as e:
            print(f"    -> Could not test streaming dataset. Error: {e}")
    else:
        print("    -> The created dataset object is None.")

    print("\n" + "="*50 + "\n")

def get_args():
    """Comprehensive argument parser supporting all configuration options."""
    parser = argparse.ArgumentParser(description="INDRA LLM - Vedic-aligned Transformer")
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True,
                       choices=["train", "pretrain", "hybrid_pretrain", "sft", "rlhf", 
                               "inference", "evaluate", "tokenizer_train"],
                       help="Training/inference mode")
    
    # Model configuration
    parser.add_argument("--model_config", type=str, default="config.yml",
                       help="Path to model configuration file")
    parser.add_argument("--vocab_size", type=int, default=262154,
                       help="Vocabulary size")
    parser.add_argument("--n_positions", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--n_embd", type=int, default=768,
                       help="Embedding dimension")
    parser.add_argument("--n_layer", type=int, default=12,
                       help="Number of layers")
    parser.add_argument("--n_head", type=int, default=12,
                       help="Number of attention heads")
    parser.add_argument("--n_kv_head", type=int, default=4,
                       help="Number of key-value heads for GQA")
    
    # MoE configuration
    parser.add_argument("--use_moe", action="store_true",
                       help="Use Mixture of Experts")
    parser.add_argument("--num_experts", type=int, default=8,
                       help="Number of experts in MoE")
    parser.add_argument("--expert_top_k", type=int, default=2,
                       help="Top-k experts to route to")
    
    # Vedic configuration
    parser.add_argument("--use_vedic_core", action="store_true", default=True,
                       help="Use Vedic integration")
    parser.add_argument("--vedic_memory_size", type=int, default=10000,
                       help="Size of Vedic memory")
    parser.add_argument("--vedic_retrieval_top_k", type=int, default=5,
                       help="Top-k for Vedic retrieval")
    
    # Hierarchical reasoning configuration
    parser.add_argument("--use_hierarchical_reasoning", action="store_true",
                       help="Enable hierarchical reasoning module")
    parser.add_argument("--reasoning_depth", type=int, default=3, choices=[1, 2, 3, 4, 5],
                       help="Depth of hierarchical reasoning")
    parser.add_argument("--enable_reasoning_in_training", action="store_true",
                       help="Enable reasoning during training")
    parser.add_argument("--reasoning_weight", type=float, default=0.1,
                       help="Weight for reasoning loss component")
    
    # Efficiency features
    parser.add_argument("--use_flash_attention", action="store_true", default=True,
                       help="Use FlashAttention")
    parser.add_argument("--use_kv_cache", action="store_true", default=True,
                       help="Use KV caching")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Use gradient checkpointing")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--micro_batch_size", type=int, default=1,
                       help="Micro batch size for gradient accumulation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=100000,
                       help="Maximum training steps")
    parser.add_argument("--eval_steps", type=int, default=1000,
                       help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=5000,
                       help="Save checkpoint frequency")
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Logging frequency")
    
    # Learning rate configuration
    parser.add_argument("--max_lr", type=float, default=6e-4,
                       help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5,
                       help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                       help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=1e-1,
                       help="Weight decay")
    
    # Mixed precision
    parser.add_argument("--use_fp16", action="store_true",
                       help="Use FP16 mixed precision")
    parser.add_argument("--use_bf16", action="store_true",
                       help="Use BF16 mixed precision")
    
    # Distributed training
    parser.add_argument("--use_ddp", action="store_true",
                       help="Use DistributedDataParallel")
    parser.add_argument("--use_fsdp", action="store_true",
                       help="Use FullyShardedDataParallel")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training")
    
    # Data configuration
    parser.add_argument("--train_data", type=str, nargs="+", required=False,
                       help="Training data paths")
    parser.add_argument("--val_data", type=str, nargs="+",
                       help="Validation data paths")
    parser.add_argument("--vedic_data", type=str, nargs="+",
                       help="Vedic corpus paths")
    parser.add_argument("--data_type", type=str, default="auto",
                       choices=["txt", "json", "csv", "hf", "auto"],
                       help="Data format")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="Maximum sequence length")
    
    # Streaming dataset configuration
    parser.add_argument("--batch_size_mb", type=float, default=5.0,
                       help="Memory batch size in MB for streaming")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                       help="Cache directory for processed data")
    parser.add_argument("--shuffle_buffer_size", type=int, default=1000,
                       help="Size of shuffle buffer for streaming")
    
    # Tokenizer configuration
    parser.add_argument("--tokenizer_path", type=str,
                       help="Path to trained tokenizer")
    parser.add_argument("--tokenizer_type", type=str, default="vedic",
                       choices=["sentencepiece", "vedic"],
                       help="Tokenizer type")
    parser.add_argument("--tokenizer_vocab_size", type=int, default=262154,
                       help="Tokenizer vocabulary size")
    
    # Teacher model configuration (for hybrid training)
    parser.add_argument("--teacher_model_path", type=str, default="Qwen/Qwen1.5-MoE-A2.7B",
                       help="Teacher model path for hybrid training")
    parser.add_argument("--use_direct_transfer", action="store_true", default=True,
                       help="Use direct knowledge transfer")
    parser.add_argument("--distillation_alpha", type=float, default=0.7,
                       help="Distillation loss weight")
    parser.add_argument("--distillation_temperature", type=float, default=4.0,
                       help="Distillation temperature")
    
    # Curriculum learning
    parser.add_argument("--vedic_phase_steps", type=int, default=50000,
                       help="Steps for Vedic-only phase")
    parser.add_argument("--general_phase_steps", type=int, default=200000,
                       help="Steps for general phase")
    parser.add_argument("--vedic_data_ratio", type=float, default=0.3,
                       help="Ratio of Vedic data in mixed phase")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                       help="Output directory")
    parser.add_argument("--resume_from_checkpoint", type=str,
                       help="Resume from checkpoint path")
    parser.add_argument("--save_total_limit", type=int, default=5,
                       help="Maximum number of checkpoints to keep")
    
    # Inference configuration
    parser.add_argument("--input_text", type=str,
                       help="Input text for inference")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Nucleus sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                       help="Repetition penalty")
    parser.add_argument("--do_sample", action="store_true", default=True,
                       help="Use sampling for generation")
    parser.add_argument("--enable_reasoning_inference", action="store_true",
                       help="Enable hierarchical reasoning during inference")
    parser.add_argument("--return_reasoning_chain", action="store_true",
                       help="Return detailed reasoning chain")
    
    # Evaluation configuration
    parser.add_argument("--eval_dataset", type=str,
                       help="Evaluation dataset")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                       help="Evaluation batch size")
    
    # Logging configuration
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="indra-llm",
                       help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str,
                       help="Wandb run name")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def create_model_config(args) -> ModelConfig:
    """Create model configuration from arguments."""
    return ModelConfig(
        vocab_size=args.vocab_size,
        n_positions=args.n_positions,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        use_moe=args.use_moe,
        use_flash_attention=args.use_flash_attention,
        use_kv_cache=args.use_kv_cache,
        gradient_checkpointing=args.gradient_checkpointing,
        use_hierarchical_reasoning=args.use_hierarchical_reasoning,
        reasoning_levels=5,  # Always use all 5 levels when enabled
    )

def create_training_config(args) -> TrainingConfig:
    """Create training configuration from arguments."""
    config = TrainingConfig(
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        use_fp16=args.use_fp16,
        use_bf16=args.use_bf16,
        use_ddp=args.use_ddp,
        use_fsdp=args.use_fsdp,
        local_rank=args.local_rank,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        save_total_limit=args.save_total_limit,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        teacher_model_path=args.teacher_model_path,
        use_direct_transfer=args.use_direct_transfer,
        distillation_alpha=args.distillation_alpha,
        distillation_temperature=args.distillation_temperature,
    )
    
    # Update pretrain config
    config.pretrain.max_lr = args.max_lr
    config.pretrain.min_lr = args.min_lr
    config.pretrain.warmup_steps = args.warmup_steps
    config.pretrain.weight_decay = args.weight_decay
    config.pretrain.vedic_phase_steps = args.vedic_phase_steps
    config.pretrain.general_phase_steps = args.general_phase_steps
    config.pretrain.vedic_data_ratio = args.vedic_data_ratio
    
    return config

def setup_logging(log_level: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_tokenizer(args):
    """Create tokenizer based on configuration."""
    if args.tokenizer_type == "vedic":
        manager = VedicTokenizerManager(
            base_dir="./tokenizers",
            gemma_dir="gemma3_indra_tokenizer"
        )
        tokenizer = manager.load_or_create(vocab_size=args.tokenizer_vocab_size)
    else:
        tokenizer = SentencePieceTokenizer(
            model_path=args.tokenizer_path,
            vocab_size=args.tokenizer_vocab_size,
        )
    return tokenizer

def create_datasets(args, tokenizer):
    """Create training and validation datasets using streaming implementation."""
    train_dataset = None
    val_dataset = None
    
    # Combine all training data paths
    all_train_paths = []
    if args.train_data:
        all_train_paths.extend(args.train_data)
    if args.vedic_data:
        all_train_paths.extend(args.vedic_data)
    
    if all_train_paths:
        # Determine dataset type based on data
        dataset_type = "vedic" if args.vedic_data else "base"
        if args.mode == "sft":
            dataset_type = "instruction"
        
        # Create streaming dataset
        train_dataset = create_streaming_dataset(
            data_path=all_train_paths,
            tokenizer=tokenizer,
            dataset_type=dataset_type,
            max_length=args.max_seq_length,
            data_type=args.data_type,
            batch_size_mb=args.batch_size_mb,
            cache_dir=args.cache_dir,
            shuffle_buffer_size=args.shuffle_buffer_size,
            # Vedic-specific parameters
            vedic_weight=2.0 if dataset_type == "vedic" else 1.0,
            preserve_structure=True,
            add_vedic_markers=True,
        )
    
    if args.val_data:
        # Create validation dataset (also streaming for consistency)
        val_dataset = create_streaming_dataset(
            data_path=args.val_data,
            tokenizer=tokenizer,
            dataset_type="base",
            max_length=args.max_seq_length,
            data_type=args.data_type,
            batch_size_mb=args.batch_size_mb,
            cache_dir=args.cache_dir,
            shuffle_buffer_size=100,  # Smaller for validation
        )
    
    return train_dataset, val_dataset

def train_tokenizer_mode(args):
    """Train tokenizer on corpus."""
    logging.info("Training tokenizer...")
    
    if args.tokenizer_type == "vedic":
        tokenizer = VedicTokenizer(vocab_size=args.tokenizer_vocab_size)
    else:
        tokenizer = SentencePieceTokenizer(vocab_size=args.tokenizer_vocab_size)
    
    # Prepare input files
    input_files = []
    for data_path in (args.train_data or []):
        if os.path.isfile(data_path):
            input_files.append(data_path)
        elif os.path.isdir(data_path):
            # Add all text files in directory
            for file_path in Path(data_path).rglob("*.txt"):
                input_files.append(str(file_path))
    
    if not input_files:
        raise ValueError("No input files found for tokenizer training")
    
    # Train tokenizer
    output_prefix = os.path.join(args.output_dir, "indra_tokenizer")
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer.train_tokenizer(
        input_files=input_files,
        model_prefix=output_prefix,
    )
    
    logging.info(f"Tokenizer saved to {output_prefix}")

def pretrain_mode(args):
    """Pre-training mode with streaming datasets."""
    logging.info("Starting pre-training with streaming datasets...")
    
    # Create configurations
    model_config = create_model_config(args)
    training_config = create_training_config(args)
    
    # Create tokenizer and datasets
    tokenizer = create_tokenizer(args)
    train_dataset, val_dataset = create_datasets(args, tokenizer)

    debug_data_pipeline(args, train_dataset)
    
    if not train_dataset:
        raise ValueError("No training dataset provided")
    
    # Create model
    model = INDRATransformer(model_config)
    
    # Add hierarchical reasoning if requested
    if args.use_hierarchical_reasoning:
        model = add_hierarchical_reasoning_to_model(model)
        logging.info("Hierarchical reasoning module added to model")
    
    # Create trainer
    trainer = PretrainTrainer(
        model=model,
        config=training_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        trainer.resume_from_checkpoint(args.resume_from_checkpoint)
    
    # Start training
    results = trainer.train()
    logging.info(f"Pre-training completed: {results}")

def hybrid_pretrain_mode(args):
    """Hybrid pre-training mode with teacher model."""
    logging.info("Starting hybrid pre-training with streaming datasets...")
    
    # Create configurations
    model_config = create_model_config(args)
    training_config = create_training_config(args)
    
    # Create tokenizer and datasets
    tokenizer = create_tokenizer(args)
    train_dataset, val_dataset = create_datasets(args, tokenizer)
    
    if not train_dataset:
        raise ValueError("No training dataset provided")
    
    # Create model
    model = INDRATransformer(model_config)
    
    # Add hierarchical reasoning if requested
    if args.use_hierarchical_reasoning:
        model = add_hierarchical_reasoning_to_model(model)
        logging.info("Hierarchical reasoning module added to model")
    
    # Create hybrid trainer
    trainer = HybridPretrainTrainer(
        model=model,
        config=training_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        teacher_model_path=args.teacher_model_path,
    )
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        trainer.resume_from_checkpoint(args.resume_from_checkpoint)
    
    # Start training
    results = trainer.train()
    logging.info(f"Hybrid pre-training completed: {results}")

def sft_mode(args):
    """Supervised fine-tuning mode with streaming datasets."""
    logging.info("Starting supervised fine-tuning with streaming datasets...")
    
    # Create configurations
    model_config = create_model_config(args)
    training_config = create_training_config(args)
    
    # Create tokenizer and datasets
    tokenizer = create_tokenizer(args)
    train_dataset, val_dataset = create_datasets(args, tokenizer)
    
    if not train_dataset:
        raise ValueError("No training dataset provided")
    
    # Load pre-trained model
    if not args.resume_from_checkpoint:
        raise ValueError("SFT requires a pre-trained model checkpoint")
    
    model = INDRATransformer(model_config)
    
    # Create SFT trainer
    trainer = SFTTrainer(
        model=model,
        config=training_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Load pre-trained weights
    trainer.resume_from_checkpoint(args.resume_from_checkpoint)
    
    # Start fine-tuning
    results = trainer.train()
    logging.info(f"SFT completed: {results}")

def inference_mode(args):
    """Inference mode for text generation."""
    logging.info("Starting inference...")
    
    if not args.input_text:
        raise ValueError("No input text provided for inference")
    
    # Create model configuration
    model_config = create_model_config(args)
    
    # Create tokenizer
    tokenizer = create_tokenizer(args)
    
    # Load model
    model = INDRATransformer(model_config)
    
    if args.resume_from_checkpoint:
        # Load trained model
        from training.trainer_utils import TrainerUtils
        TrainerUtils.load_checkpoint(
            args.resume_from_checkpoint,
            model=model,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
    
    # Create inference engine
    inference_engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Generate text
    if args.enable_reasoning_inference and model.hierarchical_reasoning is not None:
        # Use hierarchical reasoning inference
        logging.info("Using hierarchical reasoning for generation...")
        
        # Tokenize input
        inputs = tokenizer(args.input_text, return_tensors='pt')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass with reasoning
        with torch.no_grad():
            outputs = model(
                **inputs,
                enable_reasoning=True,
                reasoning_depth=args.reasoning_depth,
                return_reasoning_chain=args.return_reasoning_chain,
                return_dict=True
            )
        
        print(f"Input: {args.input_text}")
        
        if args.return_reasoning_chain and 'reasoning_chain' in outputs:
            print("\nReasoning Chain:")
            for i, step in enumerate(outputs['reasoning_chain']):
                print(f"Level {i+1} ({step.get('level', 'Unknown')}):")
                print(f"  Principles: {step.get('vedic_principles', [])}")
                print(f"  Confidence: {step.get('confidence', 0.0):.3f}")
            print()
        
        # Generate response using standard generation with reasoning-enhanced states
        generated_text = inference_engine.generate(
            prompt=args.input_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=args.do_sample,
        )
        
        print(f"Generated (with reasoning): {generated_text}")
        
        if 'confidence_scores' in outputs:
            avg_confidence = outputs['confidence_scores']['overall_confidence'].mean().item()
            print(f"Average confidence: {avg_confidence:.3f}")
    
    else:
        # Standard generation
        generated_text = inference_engine.generate(
            prompt=args.input_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=args.do_sample,
        )
        
        print(f"Input: {args.input_text}")
        print(f"Generated: {generated_text}")

def evaluate_mode(args):
    """Evaluation mode."""
    logging.info("Starting evaluation...")
    
    # Create configurations
    model_config = create_model_config(args)
    
    # Create tokenizer
    tokenizer = create_tokenizer(args)
    
    # Load model
    model = INDRATransformer(model_config)
    
    if args.resume_from_checkpoint:
        from training.trainer_utils import TrainerUtils
        TrainerUtils.load_checkpoint(
            args.resume_from_checkpoint,
            model=model,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Run evaluation
    if args.eval_dataset:
        results = evaluator.evaluate_dataset(args.eval_dataset)
    else:
        results = evaluator.evaluate_standard_benchmarks()
    
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value}")

def main():
    """Main entry point."""
    args = get_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create output and cache directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    
    # Log configuration
    logging.info(f"INDRA LLM - Mode: {args.mode}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Cache directory: {args.cache_dir}")
    logging.info(f"Streaming batch size: {args.batch_size_mb}MB")
    logging.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Route to appropriate mode
    try:
        if args.mode == "tokenizer_train":
            train_tokenizer_mode(args)
        elif args.mode in ["pretrain", "train"]:
            pretrain_mode(args)
        elif args.mode == "hybrid_pretrain":
            hybrid_pretrain_mode(args)
        elif args.mode == "sft":
            sft_mode(args)
        elif args.mode == "rlhf":
            # RLHF mode would be implemented similarly
            raise NotImplementedError("RLHF mode not yet implemented")
        elif args.mode == "inference":
            inference_mode(args)
        elif args.mode == "evaluate":
            evaluate_mode(args)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
    
    except Exception as e:
        logging.error(f"Error in {args.mode} mode: {e}")
        raise

if __name__ == "__main__":
    main()


