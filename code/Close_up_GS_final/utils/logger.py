"""
Logging utilities for Close-up-GS
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(output_dir: str, 
                log_level: str = 'INFO',
                log_to_file: bool = True,
                log_to_console: bool = True) -> logging.Logger:
    """
    Setup logger for training and evaluation
    
    Args:
        output_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('closeup_gs')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    if log_to_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(output_dir) / f'closeup_gs_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


class MetricsLogger:
    """Logger for training metrics and statistics"""
    
    def __init__(self, output_dir: str, use_tensorboard: bool = True, use_wandb: bool = False):
        """
        Initialize metrics logger
        
        Args:
            output_dir: Directory to save logs
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tensorboard_writer = None
        self.wandb_run = None
        
        # Setup TensorBoard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tensorboard_writer = SummaryWriter(self.output_dir / 'tensorboard')
                print(f"TensorBoard logging to: {self.output_dir / 'tensorboard'}")
            except ImportError:
                print("TensorBoard not available, skipping...")
        
        # Setup Weights & Biases
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project="close-up-gs",
                    dir=str(self.output_dir),
                    name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                print("Weights & Biases logging initialized")
            except ImportError:
                print("Weights & Biases not available, skipping...")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value"""
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(tag, value, step)
        
        if self.wandb_run:
            self.wandb_run.log({tag: value}, step=step)
    
    def log_image(self, tag: str, image, step: int):
        """Log image"""
        if self.tensorboard_writer:
            self.tensorboard_writer.add_image(tag, image, step)
        
        if self.wandb_run:
            import wandb
            self.wandb_run.log({tag: wandb.Image(image)}, step=step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram"""
        if self.tensorboard_writer:
            self.tensorboard_writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text"""
        if self.tensorboard_writer:
            self.tensorboard_writer.add_text(tag, text, step)
        
        if self.wandb_run:
            self.wandb_run.log({tag: text}, step=step)
    
    def log_hyperparameters(self, hparams: dict, metrics: dict):
        """Log hyperparameters"""
        if self.tensorboard_writer:
            self.tensorboard_writer.add_hparams(hparams, metrics)
        
        if self.wandb_run:
            self.wandb_run.config.update(hparams)
    
    def close(self):
        """Close loggers"""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.wandb_run:
            self.wandb_run.finish()


class ProgressLogger:
    """Simple progress logger with ETA estimation"""
    
    def __init__(self, total_steps: int, log_interval: int = 100):
        """
        Initialize progress logger
        
        Args:
            total_steps: Total number of steps
            log_interval: Interval for logging progress
        """
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.start_time = datetime.now()
        self.current_step = 0
    
    def update(self, step: int, metrics: Optional[dict] = None):
        """Update progress"""
        self.current_step = step
        
        if step % self.log_interval == 0 or step == self.total_steps:
            self._log_progress(metrics)
    
    def _log_progress(self, metrics: Optional[dict] = None):
        """Log current progress"""
        elapsed_time = datetime.now() - self.start_time
        progress = self.current_step / self.total_steps
        
        if progress > 0:
            eta = elapsed_time / progress - elapsed_time
            eta_str = str(eta).split('.')[0]  # Remove microseconds
        else:
            eta_str = "Unknown"
        
        elapsed_str = str(elapsed_time).split('.')[0]
        
        log_msg = (
            f"Step {self.current_step}/{self.total_steps} "
            f"({progress:.1%}) | "
            f"Elapsed: {elapsed_str} | "
            f"ETA: {eta_str}"
        )
        
        if metrics:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            log_msg += f" | {metrics_str}"
        
        print(log_msg)


if __name__ == '__main__':
    # Test logger setup
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = setup_logger(temp_dir)
        logger.info("Test log message")
        
        metrics_logger = MetricsLogger(temp_dir, use_tensorboard=False, use_wandb=False)
        metrics_logger.log_scalar("test/loss", 0.5, 1)
        metrics_logger.close()
        
        progress_logger = ProgressLogger(100, log_interval=25)
        for i in range(0, 101, 25):
            progress_logger.update(i, {"loss": 1.0 - i/100})

