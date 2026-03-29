#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch Training Script for RNNMorph.

Usage:
    # Train from scratch
    python train_torch.py --train-file rnnmorph/datasets/prepared/training_combined.txt

    # Train with custom config
    python train_torch.py --train-file data.txt --epochs 100 --batch-size 128

    # Resume from checkpoint
    python train_torch.py --train-file data.txt --resume checkpoints/checkpoint_epoch10.pt

    # Use Google Drive (Colab)
    python train_torch.py --train-file data.txt --use-gdrive

    # Monitor with TensorBoard
    tensorboard --logdir output/logs
"""

import argparse
import os
import sys
from pathlib import Path

import torch

from rnnmorph.torch_train import (
    TorchTrainConfig,
    TorchRNNMorphTrainer,
    load_model_from_checkpoint,
)
from rnnmorph.config import BuildModelConfig
from rnnmorph.data_preparation.loader import Loader
from rnnmorph.data_preparation.grammeme_vectorizer import GrammemeVectorizer
from rnnmorph.data_preparation.word_vocabulary import WordVocabulary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RNNMorph POS tagger in PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train_torch.py --train-file training.txt

  # Custom hyperparameters
  python train_torch.py --train-file training.txt --epochs 100 --batch-size 128 --lr 0.002

  # Resume training
  python train_torch.py --train-file training.txt --resume checkpoints/checkpoint_epoch10.pt

  # Google Colab with Drive
  python train_torch.py --train-file training.txt --use-gdrive --gdrive-path /content/drive/MyDrive/rnnmorph

  # Mixed precision training (faster on GPU)
  python train_torch.py --train-file training.txt --use-amp

  # Gradient accumulation for large effective batch size
  python train_torch.py --train-file training.txt --batch-size 32 --grad-accum 4
        """
    )
    
    # Data
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Path to training file (tab-separated format)"
    )
    parser.add_argument(
        "--build-config",
        type=str,
        default=None,
        help="Path to build config JSON (default: create from defaults)"
    )
    parser.add_argument(
        "--vectorizers-dir",
        type=str,
        default=None,
        help="Path to existing vectorizers (gram_input.json, gram_output.json, etc.)"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)"
    )
    parser.add_argument(
        "--external-batch-size",
        type=int,
        default=5000,
        help="External batch size for bucketing (default: 5000)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay / L2 regularization (default: 0.0)"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=5.0,
        help="Maximum gradient norm for clipping (default: 5.0)"
    )
    parser.add_argument(
        "--val-part",
        type=float,
        default=0.05,
        help="Validation set proportion (default: 0.05)"
    )
    
    # Learning rate scheduler
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=3,
        help="LR scheduler patience (default: 3)"
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="LR scheduler factor (default: 0.5)"
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-6,
        help="Minimum learning rate (default: 1e-6)"
    )
    
    # Checkpointing
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint subdirectory (default: checkpoints)"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (default: 1)"
    )
    parser.add_argument(
        "--keep-last-n",
        type=int,
        default=3,
        help="Keep last N checkpoints (default: 3)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    # Logging
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="TensorBoard log subdirectory (default: logs)"
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=10,
        help="Log to TensorBoard every N batches (default: 10)"
    )
    parser.add_argument(
        "--print-freq",
        type=int,
        default=50,
        help="Print to console every N batches (default: 50)"
    )
    
    # Device and performance
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
        help="Device for training (default: auto)"
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use Automatic Mixed Precision (faster on modern GPUs)"
    )
    
    # Google Drive
    parser.add_argument(
        "--use-gdrive",
        action="store_true",
        help="Backup checkpoints to Google Drive"
    )
    parser.add_argument(
        "--gdrive-path",
        type=str,
        default="/content/drive/MyDrive/rnnmorph_checkpoints",
        help="Google Drive path for checkpoints (default: /content/drive/MyDrive/rnnmorph_checkpoints)"
    )
    
    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    return parser.parse_args()


def build_model_and_vectorizers(args, vectorizers_dir: Path):
    """Build model and vectorizers from scratch or load existing."""
    
    print("="*70)
    print("Building Model and Vectorizers")
    print("="*70)
    
    # Check if vectorizers exist
    gram_input_path = vectorizers_dir / "gram_input.json"
    gram_output_path = vectorizers_dir / "gram_output.json"
    word_vocab_path = vectorizers_dir / "word_vocabulary.pickle"
    char_set_path = vectorizers_dir / "char_set.txt"
    
    if args.build_config:
        build_config_path = Path(args.build_config)
    else:
        build_config_path = vectorizers_dir / "build_config.json"
    
    # Load or build vectorizers
    if gram_input_path.exists() and args.vectorizers_dir:
        print(f"[INFO] Loading existing vectorizers from: {args.vectorizers_dir}")
        
        gram_vectorizer_input = GrammemeVectorizer()
        gram_vectorizer_input.load(str(gram_input_path))
        
        gram_vectorizer_output = GrammemeVectorizer()
        gram_vectorizer_output.load(str(gram_output_path))
        
        word_vocabulary = WordVocabulary()
        word_vocabulary.load(str(word_vocab_path))
        
        with open(char_set_path, 'r', encoding='utf-8') as f:
            char_set = f.read().rstrip()
        
        print(f"  ✓ Input grammemes: {gram_vectorizer_input.size():,}")
        print(f"  ✓ Output grammemes: {gram_vectorizer_output.size():,}")
        print(f"  ✓ Word vocabulary: {word_vocabulary.size():,} words")
        print(f"  ✓ Character set: {len(char_set)} characters")
    else:
        print(f"[INFO] Building vectorizers from training data...")
        print(f"       This may take 5-15 minutes for large datasets.")
        
        loader = Loader(language='ru')
        loader.parse_corpora([args.train_file])
        
        gram_vectorizer_input = loader.grammeme_vectorizer_input
        gram_vectorizer_output = loader.grammeme_vectorizer_output
        word_vocabulary = loader.word_vocabulary
        char_set = loader.char_set
        
        # Save vectorizers
        vectorizers_dir.mkdir(parents=True, exist_ok=True)
        gram_vectorizer_input.save(str(gram_input_path))
        gram_vectorizer_output.save(str(gram_output_path))
        word_vocabulary.save(str(word_vocab_path))
        with open(char_set_path, 'w', encoding='utf-8') as f:
            f.write(char_set)
        
        print(f"  ✓ Vectorizers saved to: {vectorizers_dir}")
    
    # Load or create build config
    if build_config_path.exists():
        print(f"[INFO] Loading build config: {build_config_path}")
        build_config = BuildModelConfig()
        build_config.load(str(build_config_path))
    else:
        print(f"[INFO] Creating default build config")
        build_config = BuildModelConfig()
        build_config.save(str(build_config_path))
    
    # Build model
    gram_input_size = gram_vectorizer_input.grammemes_count()
    num_classes = gram_vectorizer_output.size() + 1  # +1 for padding
    char_vocab_size = len(char_set) + 1  # +1 for unknown
    
    print(f"\n[INFO] Building PyTorch model:")
    print(f"  • Gram input size: {gram_input_size}")
    print(f"  • Output classes: {num_classes}")
    print(f"  • Char vocab size: {char_vocab_size}")
    
    from rnnmorph.torch_inference import RNNMorphNN
    
    model = RNNMorphNN(
        config=build_config.__dict__,
        gram_input_size=gram_input_size,
        num_output_classes=num_classes,
        char_vocab_size=char_vocab_size
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return (
        model,
        build_config,
        gram_vectorizer_input,
        gram_vectorizer_output,
        word_vocabulary,
        char_set,
    )


def main():
    args = parse_args()
    
    # Print configuration
    print("\n" + "="*70)
    print("RNNMorph PyTorch Training")
    print("="*70)
    print(f"[INFO] Training file: {args.train_file}")
    print(f"[INFO] Output directory: {args.output_dir}")
    print(f"[INFO] Device: {args.device}")
    
    if args.use_amp:
        print(f"[INFO] Mixed Precision (AMP): Enabled")
    
    if args.use_gdrive:
        print(f"[INFO] Google Drive backup: {args.gdrive_path}")
    
    # Verify training file exists
    if not os.path.exists(args.train_file):
        print(f"[ERROR] Training file not found: {args.train_file}")
        sys.exit(1)
    
    # Count words in training file
    print(f"\n[INFO] Analyzing training data...")
    with open(args.train_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        words = sum(1 for l in lines if l.strip())
        sentences = sum(1 for l in lines if not l.strip())
    print(f"  • Words: {words:,}")
    print(f"  • Sentences: {sentences:,}")
    
    # Setup vectorizers directory
    vectorizers_dir = Path(args.vectorizers_dir) if args.vectorizers_dir else Path(args.output_dir) / "vectorizers"
    
    # Build or load model
    if args.resume:
        print(f"\n[INFO] Resuming from checkpoint: {args.resume}")
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        (model, train_config, build_config, gram_vectorizer_input,
         gram_vectorizer_output, word_vocabulary, char_set) = load_model_from_checkpoint(
            args.resume, device=device
        )
        # Override some config values from args
        train_config.epochs_num = args.epochs
        train_config.save_dir = args.save_dir
        train_config.log_dir = args.log_dir
        train_config.use_gdrive = args.use_gdrive
        train_config.gdrive_path = args.gdrive_path
    else:
        (model, build_config, gram_vectorizer_input, gram_vectorizer_output,
         word_vocabulary, char_set) = build_model_and_vectorizers(args, vectorizers_dir)
        
        # Create training config
        train_config = TorchTrainConfig(
            epochs_num=args.epochs,
            batch_size=args.batch_size,
            external_batch_size=args.external_batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            gradient_accumulation_steps=args.grad_accum,
            max_grad_norm=args.max_grad_norm,
            val_part=args.val_part,
            lr_patience=args.lr_patience,
            lr_factor=args.lr_factor,
            lr_min=args.lr_min,
            save_dir=args.save_dir,
            save_freq=args.save_freq,
            keep_last_n=args.keep_last_n,
            log_dir=args.log_dir,
            log_freq=args.log_freq,
            print_freq=args.print_freq,
            random_seed=args.seed,
            device=args.device,
            use_amp=args.use_amp,
            use_gdrive=args.use_gdrive,
            gdrive_path=args.gdrive_path,
        )
    
    # Create trainer
    trainer = TorchRNNMorphTrainer(
        model=model,
        train_config=train_config,
        build_config=build_config,
        gram_vectorizer_input=gram_vectorizer_input,
        gram_vectorizer_output=gram_vectorizer_output,
        word_vocabulary=word_vocabulary,
        char_set=char_set,
        output_dir=args.output_dir,
    )
    
    # Save training config
    config_path = Path(args.output_dir) / "train_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    train_config.save(str(config_path))
    print(f"\n[INFO] Training config saved: {config_path}")
    
    # Print training summary
    print("\n" + "="*70)
    print("Training Configuration Summary")
    print("="*70)
    print(f"  Epochs:              {train_config.epochs_num}")
    print(f"  Batch size:          {train_config.batch_size}")
    print(f"  External batch:      {train_config.external_batch_size}")
    print(f"  Learning rate:       {train_config.learning_rate}")
    print(f"  Weight decay:        {train_config.weight_decay}")
    print(f"  Gradient accumulation: {train_config.gradient_accumulation_steps}")
    print(f"  Validation split:    {train_config.val_part * 100:.1f}%")
    print(f"  Random seed:         {train_config.random_seed}")
    print("="*70)
    
    # Start training
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70 + "\n")
    
    trainer.train(
        file_names=[args.train_file],
        resume_from=args.resume,
    )
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nTo monitor with TensorBoard:")
    print(f"  tensorboard --logdir {args.output_dir}/{args.log_dir}")
    print(f"\nCheckpoint directory:")
    print(f"  {args.output_dir}/{args.save_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
