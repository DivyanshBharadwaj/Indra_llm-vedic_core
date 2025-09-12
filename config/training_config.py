"""
Training configuration classes for INDRA LLM
(c) Divyansh Bharadwaj
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch

@dataclass
class PretrainConfig:
    """Pre-training configuration"""
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 2000
    lr_decay_iters: int = 320000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Curriculum learning
    vedic_phase_steps: int = 50000
    general_phase_steps: int = 200000
    vedic_data_ratio: float = 0.3

@dataclass
class SFTConfig:
    """Supervised Fine-tuning configuration"""
    max_lr: float = 5e-5
    min_lr: float = 5e-6
    warmup_steps: int = 100
    lr_decay_iters: int = 10000
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Instruction tuning specific
    max_seq_length: int = 2048
    response_loss_only: bool = True

@dataclass
class RLHFConfig:
    """RLHF training configuration"""
    ppo_epochs: int = 4
    learning_rate: float = 1.4e-5
    batch_size: int = 64
    mini_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    
    # PPO hyperparameters
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    vf_coef: float = 0.1
    ent_coef: float = 0.01
    kl_coef: float = 0.1
    target_kl: float = 6.0
    
    # Vedic alignment
    vedic_reward_weight: float = 0.3

@dataclass
class TrainingConfig:
    """Main training configuration"""
    # General training settings
    batch_size: int = 8
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 100000
    eval_steps: int = 1000
    save_steps: int = 5000
    logging_steps: int = 100
    
    # Mixed precision
    use_fp16: bool = False
    use_bf16: bool = True
    
    # Distributed training
    use_ddp: bool = False
    use_fsdp: bool = False
    local_rank: int = -1
    
    # Data settings
    max_seq_length: int = 2048
    dataloader_num_workers: int = 4
    pin_memory: bool = False
    
    # Checkpoint settings
    output_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None
    save_total_limit: int = 5
    
    # Evaluation settings
    eval_dataset: Optional[str] = None
    eval_batch_size: int = 8
    
    # Wandb logging
    use_wandb: bool = False
    wandb_project: str = "indra-llm"
    wandb_run_name: Optional[str] = None
    
    # Teacher model for hybrid training
    teacher_model_path: str = "Qwen/Qwen1.5-MoE-A2.7B"
    use_direct_transfer: bool = True
    distillation_alpha: float = 0.7
    distillation_temperature: float = 4.0
    
    # Phase-specific configs
    pretrain: PretrainConfig = None
    sft: SFTConfig = None
    rlhf: RLHFConfig = None
    
    def __post_init__(self):
        if self.pretrain is None:
            self.pretrain = PretrainConfig()
        if self.sft is None:
            self.sft = SFTConfig()
        if self.rlhf is None:
            self.rlhf = RLHFConfig()
        
        # Calculate effective batch size
        self.effective_batch_size = (
            self.batch_size * self.gradient_accumulation_steps
        )
