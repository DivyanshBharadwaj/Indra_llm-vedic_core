"""
Logging utilities for INDRA LLM with structured logging and monitoring
(c) Divyansh Bharadwaj
"""

import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, asdict

import colorlog

@dataclass
class LogConfig:
    """Configuration for logging setup."""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    colored_logs: bool = True
    file_logging: bool = True
    console_logging: bool = True
    log_dir: str = "logs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    json_logging: bool = False
    include_process_info: bool = True

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON support."""
    
    def __init__(
        self,
        include_extra: bool = True,
        json_format: bool = False,
        timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    ):
        """
        Initialize structured formatter.
        
        Args:
            include_extra: Whether to include extra fields
            json_format: Whether to output in JSON format
            timestamp_format: Timestamp format string
        """
        self.include_extra = include_extra
        self.json_format = json_format
        self.timestamp_format = timestamp_format
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record."""
        # Create base log data
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).strftime(self.timestamp_format),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add process information
        log_data.update({
            'process': os.getpid(),
            'thread': record.thread,
            'thread_name': record.threadName
        })
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                             'filename', 'module', 'lineno', 'funcName', 'created',
                             'msecs', 'relativeCreated', 'thread', 'threadName',
                             'processName', 'process', 'message', 'exc_info', 'exc_text',
                             'stack_info']:
                    extra_fields[key] = value
            
            if extra_fields:
                log_data['extra'] = extra_fields
        
        # Format output
        if self.json_format:
            return json.dumps(log_data, default=str, ensure_ascii=False)
        else:
            # Human-readable format
            base_msg = f"{log_data['timestamp']} - {log_data['logger']} - {log_data['level']} - {log_data['message']}"
            
            if 'exception' in log_data:
                base_msg += f"\n{log_data['exception']}"
            
            if 'extra' in log_data and log_data['extra']:
                extra_str = ", ".join(f"{k}={v}" for k, v in log_data['extra'].items())
                base_msg += f" [{extra_str}]"
            
            return base_msg

class VedicLogger:
    """Specialized logger for INDRA LLM with Vedic principles integration."""
    
    def __init__(self, name: str, config: Optional[LogConfig] = None):
        """
        Initialize Vedic logger.
        
        Args:
            name: Logger name
            config: Logging configuration
        """
        self.name = name
        self.config = config or LogConfig()
        self.logger = logging.getLogger(name)
        self._setup_logger()
        
        # Vedic principles tracking
        self.vedic_events = {
            'dharma_alignments': 0,
            'karma_considerations': 0,
            'ahimsa_practices': 0,
            'satya_adherence': 0
        }
    
    def _setup_logger(self):
        """Setup logger with handlers and formatters."""
        self.logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        if self.config.console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            
            if self.config.colored_logs:
                console_formatter = colorlog.ColoredFormatter(
                    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    log_colors={
                        'DEBUG': 'cyan',
                        'INFO': 'green',
                        'WARNING': 'yellow',
                        'ERROR': 'red',
                        'CRITICAL': 'red,bg_white',
                    }
                )
            else:
                console_formatter = StructuredFormatter(
                    json_format=self.config.json_logging
                )
            
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.file_logging:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"{self.name}.log"
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            
            file_formatter = StructuredFormatter(
                json_format=self.config.json_logging
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            # Separate JSON log file if requested
            if self.config.json_logging:
                json_log_file = log_dir / f"{self.name}_structured.jsonl"
                json_handler = logging.handlers.RotatingFileHandler(
                    json_log_file,
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count,
                    encoding='utf-8'
                )
                json_formatter = StructuredFormatter(json_format=True)
                json_handler.setFormatter(json_formatter)
                self.logger.addHandler(json_handler)
    
    def dharma_log(self, message: str, **kwargs):
        """Log with dharma (righteousness) context."""
        self.vedic_events['dharma_alignments'] += 1
        extra = {'vedic_principle': 'dharma', 'dharma_event': True}
        extra.update(kwargs)
        self.logger.info(f"[DHARMA] {message}", extra=extra)
    
    def karma_log(self, message: str, action: str, consequence: str, **kwargs):
        """Log with karma (action-consequence) context."""
        self.vedic_events['karma_considerations'] += 1
        extra = {
            'vedic_principle': 'karma',
            'action': action,
            'consequence': consequence,
            'karma_event': True
        }
        extra.update(kwargs)
        self.logger.info(f"[KARMA] {message}", extra=extra)
    
    def ahimsa_log(self, message: str, **kwargs):
        """Log with ahimsa (non-violence) context."""
        self.vedic_events['ahimsa_practices'] += 1
        extra = {'vedic_principle': 'ahimsa', 'ahimsa_event': True}
        extra.update(kwargs)
        self.logger.info(f"[AHIMSA] {message}", extra=extra)
    
    def satya_log(self, message: str, **kwargs):
        """Log with satya (truthfulness) context."""
        self.vedic_events['satya_adherence'] += 1
        extra = {'vedic_principle': 'satya', 'satya_event': True}
        extra.update(kwargs)
        self.logger.info(f"[SATYA] {message}", extra=extra)
    
    def training_step_log(
        self,
        step: int,
        loss: float,
        lr: float,
        phase: str = "training",
        **metrics
    ):
        """Log training step with comprehensive metrics."""
        extra = {
            'training_step': step,
            'loss': loss,
            'learning_rate': lr,
            'training_phase': phase,
            'metrics': metrics
        }
        self.logger.info(f"Training Step {step}: loss={loss:.4f}, lr={lr:.2e}", extra=extra)
    
    def model_performance_log(
        self,
        metric_name: str,
        metric_value: float,
        benchmark: str = None,
        **context
    ):
        """Log model performance metrics."""
        extra = {
            'metric_name': metric_name,
            'metric_value': metric_value,
            'performance_log': True
        }
        
        if benchmark:
            extra['benchmark'] = benchmark
        
        extra.update(context)
        
        self.logger.info(
            f"Performance: {metric_name}={metric_value:.4f}" + 
            (f" on {benchmark}" if benchmark else ""),
            extra=extra
        )
    
    def vedic_alignment_log(
        self,
        text: str,
        alignment_scores: Dict[str, float],
        **context
    ):
        """Log Vedic alignment analysis."""
        extra = {
            'vedic_alignment': alignment_scores,
            'analyzed_text_length': len(text),
            'overall_alignment': alignment_scores.get('overall_alignment', 0.0),
            'alignment_analysis': True
        }
        extra.update(context)
        
        overall = alignment_scores.get('overall_alignment', 0.0)
        self.logger.info(f"Vedic Alignment Analysis: {overall:.3f}", extra=extra)
    
    def error_with_context(
        self,
        message: str,
        error: Exception,
        context: Dict[str, Any] = None,
        recovery_action: str = None
    ):
        """Log error with rich context information."""
        extra = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {},
            'error_log': True
        }
        
        if recovery_action:
            extra['recovery_action'] = recovery_action
        
        self.logger.error(f"{message}: {error}", extra=extra, exc_info=True)
    
    def get_vedic_stats(self) -> Dict[str, int]:
        """Get statistics of Vedic principle usage in logs."""
        return self.vedic_events.copy()
    
    def __getattr__(self, name: str):
        """Delegate unknown attributes to underlying logger."""
        return getattr(self.logger, name)

class TrainingLogger:
    """Specialized logger for training with metrics tracking."""
    
    def __init__(self, name: str, output_dir: str, config: Optional[LogConfig] = None):
        """
        Initialize training logger.
        
        Args:
            name: Logger name
            output_dir: Output directory for logs
            config: Logging configuration
        """
        if config is None:
            config = LogConfig()
        
        config.log_dir = os.path.join(output_dir, "logs")
        config.file_logging = True
        config.json_logging = True
        
        self.vedic_logger = VedicLogger(name, config)
        self.metrics_history: Dict[str, List[Any]] = {}
        
        # Create separate loggers for different components
        self.model_logger = VedicLogger(f"{name}.model", config)
        self.data_logger = VedicLogger(f"{name}.data", config)
        self.eval_logger = VedicLogger(f"{name}.eval", config)
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.vedic_logger.info(
            f"Starting epoch {epoch}/{total_epochs}",
            extra={'epoch': epoch, 'total_epochs': total_epochs, 'event': 'epoch_start'}
        )
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch end with metrics."""
        self.vedic_logger.info(
            f"Completed epoch {epoch}",
            extra={'epoch': epoch, 'epoch_metrics': metrics, 'event': 'epoch_end'}
        )
        
        # Store metrics
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append(value)
    
    def log_batch(
        self,
        step: int,
        loss: float,
        batch_size: int,
        lr: float = None,
        **additional_metrics
    ):
        """Log batch processing."""
        extra = {
            'step': step,
            'loss': loss,
            'batch_size': batch_size,
            'event': 'batch_processed'
        }
        
        if lr is not None:
            extra['learning_rate'] = lr
        
        extra.update(additional_metrics)
        
        self.vedic_logger.info(f"Step {step}: loss={loss:.4f}", extra=extra)
    
    def log_validation(self, step: int, val_metrics: Dict[str, float]):
        """Log validation results."""
        self.eval_logger.info(
            f"Validation at step {step}",
            extra={'step': step, 'validation_metrics': val_metrics, 'event': 'validation'}
        )
    
    def log_checkpoint_save(self, step: int, checkpoint_path: str, file_size_mb: float):
        """Log checkpoint saving."""
        self.model_logger.info(
            f"Checkpoint saved at step {step}",
            extra={
                'step': step,
                'checkpoint_path': checkpoint_path,
                'file_size_mb': file_size_mb,
                'event': 'checkpoint_saved'
            }
        )
    
    def log_model_info(self, model_params: int, model_config: Dict[str, Any]):
        """Log model information."""
        self.model_logger.info(
            f"Model initialized with {model_params:,} parameters",
            extra={
                'model_parameters': model_params,
                'model_config': model_config,
                'event': 'model_info'
            }
        )
    
    def log_data_loading(self, dataset_size: int, batch_size: int, num_batches: int):
        """Log data loading information."""
        self.data_logger.info(
            f"Data loaded: {dataset_size} samples, {num_batches} batches",
            extra={
                'dataset_size': dataset_size,
                'batch_size': batch_size,
                'num_batches': num_batches,
                'event': 'data_loaded'
            }
        )
    
    def log_vedic_training_phase(self, phase: str, phase_description: str):
        """Log Vedic training phase transitions."""
        self.vedic_logger.dharma_log(
            f"Training phase: {phase} - {phase_description}",
            training_phase=phase,
            phase_description=phase_description
        )

def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    colored_logs: bool = True,
    file_logging: bool = True,
    json_logging: bool = False,
    max_file_size: int = 10 * 1024 * 1024,
    backup_count: int = 5
) -> LogConfig:
    """
    Setup logging configuration for INDRA LLM.
    
    Args:
        log_level: Logging level
        log_dir: Directory for log files
        colored_logs: Whether to use colored console output
        file_logging: Whether to log to files
        json_logging: Whether to include JSON structured logging
        max_file_size: Maximum file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        LogConfig object
    """
    config = LogConfig(
        log_level=log_level,
        log_dir=log_dir,
        colored_logs=colored_logs,
        file_logging=file_logging,
        json_logging=json_logging,
        max_file_size=max_file_size,
        backup_count=backup_count
    )
    
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    return config

def get_logger(
    name: str,
    config: Optional[LogConfig] = None,
    vedic_enabled: bool = True
) -> Union[VedicLogger, logging.Logger]:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        config: Logging configuration
        vedic_enabled: Whether to use Vedic logger features
        
    Returns:
        Logger instance
    """
    if vedic_enabled:
        return VedicLogger(name, config)
    else:
        return logging.getLogger(name)

def create_training_logger(
    experiment_name: str,
    output_dir: str,
    config: Optional[LogConfig] = None
) -> TrainingLogger:
    """
    Create a training logger for experiment tracking.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Output directory for logs
        config: Logging configuration
        
    Returns:
        TrainingLogger instance
    """
    return TrainingLogger(experiment_name, output_dir, config)

# Utility functions for common logging patterns

def log_system_info(logger: logging.Logger):
    """Log system information."""
    import platform
    import psutil
    import torch
    
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2)
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            system_info[f'gpu_{i}_name'] = gpu_name
            system_info[f'gpu_{i}_memory_gb'] = round(gpu_memory, 2)
    
    logger.info("System Information", extra={'system_info': system_info})

def log_training_completion(
    logger: logging.Logger,
    total_steps: int,
    total_time: float,
    final_loss: float,
    model_path: str
):
    """Log training completion summary."""
    completion_info = {
        'total_steps': total_steps,
        'total_time_hours': round(total_time / 3600, 2),
        'final_loss': final_loss,
        'model_path': model_path,
        'tokens_per_second': None,  # Could be calculated if available
        'event': 'training_completed'
    }
    
    logger.info(
        f"Training completed: {total_steps} steps in {total_time/3600:.2f} hours",
        extra=completion_info
    )

# Context managers for scoped logging

class LoggingContext:
    """Context manager for scoped logging with automatic timing."""
    
    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        level: int = logging.INFO,
        **context
    ):
        """
        Initialize logging context.
        
        Args:
            logger: Logger instance
            operation: Operation being performed
            level: Log level
            **context: Additional context information
        """
        self.logger = logger
        self.operation = operation
        self.level = level
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        """Enter context - log operation start."""
        self.start_time = datetime.now()
        self.logger.log(
            self.level,
            f"Starting {self.operation}",
            extra={
                'operation': self.operation,
                'event': 'operation_start',
                'start_time': self.start_time.isoformat(),
                **self.context
            }
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - log operation completion."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            # Success
            self.logger.log(
                self.level,
                f"Completed {self.operation} in {duration:.2f}s",
                extra={
                    'operation': self.operation,
                    'event': 'operation_completed',
                    'duration_seconds': duration,
                    'success': True,
                    **self.context
                }
            )
        else:
            # Error occurred
            self.logger.error(
                f"Failed {self.operation} after {duration:.2f}s: {exc_val}",
                extra={
                    'operation': self.operation,
                    'event': 'operation_failed',
                    'duration_seconds': duration,
                    'success': False,
                    'error_type': exc_type.__name__ if exc_type else None,
                    'error_message': str(exc_val) if exc_val else None,
                    **self.context
                },
                exc_info=True
            )

# Example usage functions

def demo_vedic_logging():
    """Demonstrate Vedic logging capabilities."""
    config = setup_logging(colored_logs=True, json_logging=True)
    logger = get_logger("indra.demo", config)
    
    # Regular logging
    logger.info("Starting INDRA LLM training")
    
    # Vedic principle logging
    logger.dharma_log("Model architecture follows dharmic principles of balance")
    logger.karma_log(
        "Training step completed",
        action="gradient_update",
        consequence="loss_reduction"
    )
    logger.ahimsa_log("Using non-harmful training practices")
    logger.satya_log("Reporting accurate training metrics")
    
    # Training metrics
    logger.training_step_log(
        step=1000,
        loss=2.45,
        lr=1e-4,
        phase="pretrain",
        vedic_alignment=0.85
    )
    
    # Get Vedic statistics
    stats = logger.get_vedic_stats()
    logger.info(f"Vedic logging stats: {stats}")

if __name__ == "__main__":
    demo_vedic_logging()
