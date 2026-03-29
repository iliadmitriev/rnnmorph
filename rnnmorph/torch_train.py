# -*- coding: utf-8 -*-
"""
PyTorch Training Module for RNNMorph.

Pure PyTorch implementation without TensorFlow/Keras dependencies.
Includes:
- Training loop with gradient accumulation
- Metric evaluation (word/sentence accuracy)
- TensorBoard logging
- Checkpoint saving (local + Google Drive)
- Progress bars with time estimates
"""

import os
import json
import time
import random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from rnnmorph.torch_inference import RNNMorphNN, ReversedLSTM, CharEmbeddingNetwork
from rnnmorph.data_preparation.grammeme_vectorizer import GrammemeVectorizer
from rnnmorph.data_preparation.word_vocabulary import WordVocabulary
from rnnmorph.batch_generator import BatchGenerator
from rnnmorph.config import BuildModelConfig, TrainConfig


@dataclass
class TorchTrainConfig:
    """Configuration for PyTorch training."""
    
    # Training hyperparameters
    epochs_num: int = 50
    batch_size: int = 64
    external_batch_size: int = 5000
    val_part: float = 0.05
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 5.0
    
    # Learning rate
    learning_rate: float = 0.001
    lr_patience: int = 3
    lr_factor: float = 0.5
    lr_min: float = 1e-6
    
    # Regularization
    weight_decay: float = 0.0
    dropout: float = 0.3
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_freq: int = 1  # Save every N epochs
    keep_last_n: int = 3  # Keep last N checkpoints
    
    # Logging
    log_dir: str = "logs"
    log_freq: int = 10  # Log every N batches
    print_freq: int = 50  # Print every N batches
    
    # Random seed
    random_seed: int = 42
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", or "cuda:N"
    
    # Mixed precision
    use_amp: bool = False  # Automatic Mixed Precision
    
    # Google Drive (optional)
    use_gdrive: bool = False
    gdrive_path: str = "/content/drive/MyDrive/rnnmorph_checkpoints"
    
    def save(self, filename: str) -> None:
        """Save config to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, filename: str) -> 'TorchTrainConfig':
        """Load config from JSON file."""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


class MetricsTracker:
    """Tracks training and validation metrics."""
    
    def __init__(self):
        self.reset_epoch()
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.history = {
            'train_loss': [],
            'train_word_acc': [],
            'train_sentence_acc': [],
            'val_loss': [],
            'val_word_acc': [],
            'val_sentence_acc': [],
            'learning_rate': [],
        }
    
    def reset_epoch(self):
        """Reset epoch-level counters."""
        self.epoch_loss = 0.0
        self.epoch_word_correct = 0
        self.epoch_word_total = 0
        self.epoch_sentence_correct = 0
        self.epoch_sentence_total = 0
        self.num_batches = 0
    
    def update_batch(self, loss: float, word_correct: int, word_total: int,
                     sentence_correct: int, sentence_total: int):
        """Update metrics after a batch."""
        self.epoch_loss += loss
        self.epoch_word_correct += word_correct
        self.epoch_word_total += word_total
        self.epoch_sentence_correct += sentence_correct
        self.epoch_sentence_total += sentence_total
        self.num_batches += 1
    
    def get_epoch_metrics(self) -> Dict[str, float]:
        """Get averaged epoch metrics."""
        return {
            'loss': self.epoch_loss / max(self.num_batches, 1),
            'word_accuracy': self.epoch_word_correct / max(self.epoch_word_total, 1),
            'sentence_accuracy': self.epoch_sentence_correct / max(self.epoch_sentence_total, 1),
        }
    
    def update_history(self, train_metrics: Dict[str, float], 
                       val_metrics: Dict[str, float], lr: float):
        """Update epoch history."""
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_word_acc'].append(train_metrics['word_accuracy'])
        self.history['train_sentence_acc'].append(train_metrics['sentence_accuracy'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_word_acc'].append(val_metrics['word_accuracy'])
        self.history['val_sentence_acc'].append(val_metrics['sentence_accuracy'])
        self.history['learning_rate'].append(lr)
        
        # Track best model
        if val_metrics['word_accuracy'] > self.best_val_accuracy:
            self.best_val_accuracy = val_metrics['word_accuracy']
            self.best_epoch = len(self.history['train_loss']) - 1


class TorchRNNMorphTrainer:
    """
    Trainer for RNNMorph model in PyTorch.
    
    Features:
    - Training loop with gradient accumulation
    - Metric evaluation (word/sentence accuracy)
    - TensorBoard logging
    - Checkpoint saving
    - Progress bars with time estimates
    """
    
    def __init__(
        self,
        model: RNNMorphNN,
        train_config: TorchTrainConfig,
        build_config: BuildModelConfig,
        gram_vectorizer_input: GrammemeVectorizer,
        gram_vectorizer_output: GrammemeVectorizer,
        word_vocabulary: WordVocabulary,
        char_set: str,
        output_dir: str = "output",
    ):
        self.model = model
        self.train_config = train_config
        self.build_config = build_config
        self.gram_vectorizer_input = gram_vectorizer_input
        self.gram_vectorizer_output = gram_vectorizer_output
        self.word_vocabulary = word_vocabulary
        self.char_set = char_set
        self.output_dir = Path(output_dir)
        
        # Setup device
        self.device = self._setup_device(train_config.device)
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=train_config.lr_factor,
            patience=train_config.lr_patience,
            min_lr=train_config.lr_min,
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Metrics tracker
        self.metrics = MetricsTracker()
        
        # TensorBoard writer
        self.writer = None
        
        # Scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if train_config.use_amp else None
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / train_config.save_dir).mkdir(exist_ok=True)
        
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Output directory: {self.output_dir}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computing device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        elif device.startswith("cuda") and torch.cuda.is_available():
            return torch.device(device)
        return torch.device("cpu")
    
    def _setup_tensorboard(self):
        """Setup TensorBoard writer."""
        log_path = self.output_dir / self.train_config.log_dir
        self.writer = SummaryWriter(log_dir=str(log_path))
        print(f"[INFO] TensorBoard logs: {log_path}")
    
    def _prepare_batch(
        self,
        inputs: List[np.ndarray],
        targets: List[np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare batch tensors for training.
        
        Returns:
            grammemes: (batch, seq_len, gram_input_size)
            chars: (batch, seq_len, max_word_length)
            labels: (batch, seq_len)
        """
        # Extract features based on config
        if self.build_config.use_gram:
            grammemes = torch.from_numpy(inputs[-2 if self.build_config.use_chars else -1]).float()
        else:
            grammemes = None
        
        if self.build_config.use_chars:
            chars = torch.from_numpy(inputs[-1]).long()
        else:
            chars = None
        
        # Targets
        labels = torch.from_numpy(targets[0]).long().squeeze(-1)  # (batch, seq_len)
        
        return grammemes, chars, labels
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[int, int, int, int]:
        """
        Compute word and sentence accuracy metrics.
        
        Returns:
            word_correct, word_total, sentence_correct, sentence_total
        """
        # Get predicted classes (skip padding class 0)
        pred_classes = predictions.argmax(dim=-1)  # (batch, seq_len)
        
        batch_size, seq_len = labels.shape
        word_correct = 0
        word_total = 0
        sentence_correct = 0
        sentence_total = 0
        
        for i in range(batch_size):
            # Find non-padding positions
            mask = labels[i] != 0
            if not mask.any():
                continue
            
            sentence_labels = labels[i, mask]
            sentence_preds = pred_classes[i, mask]
            
            # Word-level accuracy
            correct = (sentence_labels == sentence_preds).sum().item()
            total = len(sentence_labels)
            word_correct += correct
            word_total += total
            
            # Sentence-level accuracy (all words correct)
            if correct == total:
                sentence_correct += 1
            sentence_total += 1
        
        return word_correct, word_total, sentence_correct, sentence_total
    
    def train_epoch(
        self,
        batch_generator: BatchGenerator,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.metrics.reset_epoch()
        
        # Create progress bar
        pbar = tqdm(
            total=0,  # Unknown total, will update dynamically
            desc=f"Epoch {epoch + 1}/{self.train_config.epochs_num} [Train]",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        
        batch_num = 0
        accumulated_loss = 0.0
        
        for inputs, targets in batch_generator:
            # Prepare tensors
            grammemes, chars, labels = self._prepare_batch(inputs, targets)
            grammemes = grammemes.to(self.device)
            chars = chars.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            if self.train_config.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(grammemes, chars)
                    
                    # Reshape for loss: (batch * seq_len, num_classes)
                    batch_size, seq_len, num_classes = logits.shape
                    logits_flat = logits.view(-1, num_classes)
                    labels_flat = labels.view(-1)
                    
                    loss = self.criterion(logits_flat, labels_flat)
                    loss = loss / self.train_config.gradient_accumulation_steps
            else:
                logits = self.model(grammemes, chars)
                
                batch_size, seq_len, num_classes = logits.shape
                logits_flat = logits.view(-1, num_classes)
                labels_flat = labels.view(-1)
                
                loss = self.criterion(logits_flat, labels_flat)
                loss = loss / self.train_config.gradient_accumulation_steps
            
            # Backward pass
            if self.train_config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulated_loss += loss.item() * self.train_config.gradient_accumulation_steps
            
            # Gradient accumulation step
            if (batch_num + 1) % self.train_config.gradient_accumulation_steps == 0:
                if self.train_config.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.max_grad_norm
                )
                
                if self.train_config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Compute metrics
            with torch.no_grad():
                predictions = logits.detach()
                word_correct, word_total, sentence_correct, sentence_total = \
                    self._compute_metrics(predictions, labels)
            
            self.metrics.update_batch(
                loss=accumulated_loss,
                word_correct=word_correct,
                word_total=word_total,
                sentence_correct=sentence_correct,
                sentence_total=sentence_total,
            )
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{accumulated_loss:.4f}",
                'word_acc': f"{self.metrics.epoch_word_correct / max(self.metrics.epoch_word_total, 1):.4f}",
            })
            
            # TensorBoard logging
            if self.writer and batch_num % self.train_config.log_freq == 0:
                step = epoch * 1000 + batch_num
                self.writer.add_scalar('train/batch_loss', accumulated_loss, step)
                self.writer.add_scalar(
                    'train/batch_word_acc',
                    word_correct / max(word_total, 1),
                    step
                )
            
            batch_num += 1
        
        pbar.close()
        
        return self.metrics.get_epoch_metrics()
    
    @torch.no_grad()
    def evaluate(
        self,
        batch_generator: BatchGenerator,
        epoch: int,
    ) -> Dict[str, float]:
        """Evaluate on validation data."""
        self.model.eval()
        self.metrics.reset_epoch()
        
        # Create progress bar
        pbar = tqdm(
            total=0,
            desc=f"Epoch {epoch + 1}/{self.train_config.epochs_num} [Val]  ",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        
        batch_num = 0
        
        for inputs, targets in batch_generator:
            # Prepare tensors
            grammemes, chars, labels = self._prepare_batch(inputs, targets)
            grammemes = grammemes.to(self.device)
            chars = chars.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(grammemes, chars)
            
            # Compute loss
            batch_size, seq_len, num_classes = logits.shape
            logits_flat = logits.view(-1, num_classes)
            labels_flat = labels.view(-1)
            
            loss = self.criterion(logits_flat, labels_flat)
            
            # Compute metrics
            predictions = logits
            word_correct, word_total, sentence_correct, sentence_total = \
                self._compute_metrics(predictions, labels)
            
            self.metrics.update_batch(
                loss=loss.item(),
                word_correct=word_correct,
                word_total=word_total,
                sentence_correct=sentence_correct,
                sentence_total=sentence_total,
            )
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'word_acc': f"{self.metrics.epoch_word_correct / max(self.metrics.epoch_word_total, 1):.4f}",
            })
            
            batch_num += 1
        
        pbar.close()
        
        return self.metrics.get_epoch_metrics()
    
    def save_checkpoint(
        self,
        epoch: int,
        val_accuracy: float,
        filename: Optional[str] = None,
    ) -> str:
        """Save training checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch{epoch + 1}_acc{val_accuracy:.4f}.pt"
        
        save_path = self.output_dir / self.train_config.save_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_config': asdict(self.train_config),
            'build_config': self.build_config.__dict__,
            'gram_vectorizer_input': self.gram_vectorizer_input,
            'gram_vectorizer_output': self.gram_vectorizer_output,
            'word_vocabulary': self.word_vocabulary,
            'char_set': self.char_set,
            'val_accuracy': val_accuracy,
        }
        
        if self.train_config.use_amp and self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, str(save_path))
        
        # Google Drive backup
        if self.train_config.use_gdrive:
            gdrive_path = Path(self.train_config.gdrive_path)
            gdrive_path.mkdir(parents=True, exist_ok=True)
            gdrive_file = gdrive_path / filename
            torch.save(checkpoint, str(gdrive_file))
            print(f"[INFO] Checkpoint saved to Google Drive: {gdrive_file}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(save_path)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only last N."""
        checkpoint_dir = self.output_dir / self.train_config.save_dir
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for checkpoint in checkpoints[self.train_config.keep_last_n:]:
            checkpoint.unlink()
            print(f"[INFO] Removed old checkpoint: {checkpoint}")
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
        epoch_time: float,
    ):
        """Log epoch metrics to TensorBoard and console."""
        # TensorBoard
        if self.writer:
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/train_word_acc', train_metrics['word_accuracy'], epoch)
            self.writer.add_scalar('epoch/train_sentence_acc', train_metrics['sentence_accuracy'], epoch)
            self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/val_word_acc', val_metrics['word_accuracy'], epoch)
            self.writer.add_scalar('epoch/val_sentence_acc', val_metrics['sentence_accuracy'], epoch)
            self.writer.add_scalar('epoch/learning_rate', lr, epoch)
            self.writer.add_scalar('epoch/time_per_epoch', epoch_time, epoch)
        
        # Console output
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{self.train_config.epochs_num} completed in {epoch_time:.1f}s")
        print(f"{'='*70}")
        print(f"  Train Loss:     {train_metrics['loss']:.4f}")
        print(f"  Train Word Acc: {train_metrics['word_accuracy']:.4f} ({self.metrics.epoch_word_correct}/{self.metrics.epoch_word_total})")
        print(f"  Train Sent Acc: {train_metrics['sentence_accuracy']:.4f}")
        print(f"  Val Loss:       {val_metrics['loss']:.4f}")
        print(f"  Val Word Acc:   {val_metrics['word_accuracy']:.4f}")
        print(f"  Val Sent Acc:   {val_metrics['sentence_accuracy']:.4f}")
        print(f"  Learning Rate:  {lr:.6f}")
        print(f"{'='*70}\n")
    
    def train(
        self,
        file_names: List[str],
        resume_from: Optional[str] = None,
    ):
        """
        Full training loop.
        
        Args:
            file_names: List of training file paths
            resume_from: Optional checkpoint path to resume from
        """
        # Setup TensorBoard
        self._setup_tensorboard()
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            start_epoch = self._load_checkpoint(resume_from)
        
        # Setup data splitting
        np.random.seed(self.train_config.random_seed)
        torch.manual_seed(self.train_config.random_seed)
        random.seed(self.train_config.random_seed)
        
        sample_counter = self._count_samples(file_names)
        train_idx, val_idx = self._get_split(sample_counter, self.train_config.val_part)
        
        print(f"[INFO] Total samples: {sample_counter:,}")
        print(f"[INFO] Train samples: {len(train_idx):,} ({100 * len(train_idx) / sample_counter:.1f}%)")
        print(f"[INFO] Val samples: {len(val_idx):,} ({100 * len(val_idx) / sample_counter:.1f}%)")
        
        # Training loop
        best_val_accuracy = self.metrics.best_val_accuracy
        start_time = time.time()
        
        for epoch in range(start_epoch, self.train_config.epochs_num):
            epoch_start = time.time()
            
            # Create batch generators
            train_generator = BatchGenerator(
                language='ru',
                file_names=file_names,
                config=self._create_train_config(),
                grammeme_vectorizer_input=self.gram_vectorizer_input,
                grammeme_vectorizer_output=self.gram_vectorizer_output,
                indices=train_idx,
                word_vocabulary=self.word_vocabulary,
                char_set=self.char_set,
                build_config=self.build_config,
            )
            
            val_generator = BatchGenerator(
                language='ru',
                file_names=file_names,
                config=self._create_train_config(),
                grammeme_vectorizer_input=self.gram_vectorizer_input,
                grammeme_vectorizer_output=self.gram_vectorizer_output,
                indices=val_idx,
                word_vocabulary=self.word_vocabulary,
                char_set=self.char_set,
                build_config=self.build_config,
            )
            
            # Train
            train_metrics = self.train_epoch(train_generator, epoch)
            
            # Evaluate
            val_metrics = self.evaluate(val_generator, epoch)
            
            # Update scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics['word_accuracy'])
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.metrics.update_history(train_metrics, val_metrics, current_lr)
            
            # Epoch timing
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            epochs_remaining = self.train_config.epochs_num - epoch - 1
            eta = epoch_time * epochs_remaining if epochs_remaining > 0 else 0
            
            # Log
            self.log_epoch(epoch, train_metrics, val_metrics, new_lr, epoch_time)
            
            print(f"Total time: {total_time / 3600:.2f}h | ETA: {eta / 60:.1f}min ({epochs_remaining} epochs remaining)")
            
            # Save checkpoint
            if (epoch + 1) % self.train_config.save_freq == 0:
                checkpoint_path = self.save_checkpoint(epoch, val_metrics['word_accuracy'])
                print(f"[INFO] Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if val_metrics['word_accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['word_accuracy']
                best_path = self.save_checkpoint(
                    epoch,
                    val_metrics['word_accuracy'],
                    filename="checkpoint_best.pt"
                )
                print(f"[INFO] New best model saved: {best_path}")
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print("TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Total training time: {total_time / 3600:.2f} hours")
        print(f"Best validation word accuracy: {best_val_accuracy:.4f} (epoch {self.metrics.best_epoch + 1})")
        print(f"Final validation word accuracy: {self.metrics.history['val_word_acc'][-1]:.4f}")
        print(f"{'='*70}\n")
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
    
    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and return epoch to resume from."""
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore vectorizers
        self.gram_vectorizer_input.__dict__.update(
            checkpoint['gram_vectorizer_input'].__dict__
        )
        self.gram_vectorizer_output.__dict__.update(
            checkpoint['gram_vectorizer_output'].__dict__
        )
        self.word_vocabulary.__dict__.update(
            checkpoint['word_vocabulary'].__dict__
        )
        self.char_set = checkpoint['char_set']
        
        epoch = checkpoint['epoch'] + 1
        self.metrics.best_val_accuracy = checkpoint.get('val_accuracy', 0.0)
        
        print(f"[INFO] Resuming from epoch {epoch}")
        return epoch
    
    def _count_samples(self, file_names: List[str]) -> int:
        """Count number of sentences in files."""
        sample_counter = 0
        for filename in file_names:
            with open(filename, "r", encoding='utf-8') as f:
                for line in f:
                    if len(line.strip()) == 0:
                        sample_counter += 1
        return sample_counter
    
    def _get_split(self, sample_counter: int, val_part: float) -> Tuple[np.array, np.array]:
        """Split indices into train/val."""
        perm = np.random.permutation(sample_counter)
        border = int(sample_counter * (1 - val_part))
        return perm[:border], perm[border:]
    
    def _create_train_config(self) -> TrainConfig:
        """Create TrainConfig for BatchGenerator."""
        config = TrainConfig()
        config.batch_size = self.train_config.batch_size
        config.external_batch_size = self.train_config.external_batch_size
        return config


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "auto"
) -> Tuple[RNNMorphNN, TorchTrainConfig, BuildModelConfig, GrammemeVectorizer, GrammemeVectorizer, WordVocabulary, str]:
    """
    Load model and all training state from checkpoint.
    
    Returns:
        model, train_config, build_config, gram_vectorizer_input,
        gram_vectorizer_output, word_vocabulary, char_set
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Setup device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    device = torch.device(device)
    
    # Restore configs
    train_config = TorchTrainConfig(**checkpoint['train_config'])
    build_config = BuildModelConfig()
    build_config.__dict__.update(checkpoint['build_config'])
    
    # Restore vectorizers
    gram_vectorizer_input = GrammemeVectorizer()
    gram_vectorizer_input.__dict__.update(
        checkpoint['gram_vectorizer_input'].__dict__
    )
    
    gram_vectorizer_output = GrammemeVectorizer()
    gram_vectorizer_output.__dict__.update(
        checkpoint['gram_vectorizer_output'].__dict__
    )
    
    word_vocabulary = WordVocabulary()
    word_vocabulary.__dict__.update(
        checkpoint['word_vocabulary'].__dict__
    )
    
    char_set = checkpoint['char_set']
    
    # Recreate model
    gram_input_size = gram_vectorizer_input.grammemes_count()
    num_classes = gram_vectorizer_output.size() + 1
    char_vocab_size = len(char_set) + 1
    
    model = RNNMorphNN(
        config=checkpoint['build_config'],
        gram_input_size=gram_input_size,
        num_output_classes=num_classes,
        char_vocab_size=char_vocab_size
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return (model, train_config, build_config, gram_vectorizer_input,
            gram_vectorizer_output, word_vocabulary, char_set)
