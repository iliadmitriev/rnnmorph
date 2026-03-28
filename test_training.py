#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick training test script - runs 1 epoch on small dataset to verify setup.
"""

import os
import json
from rnnmorph.train import train
from rnnmorph.config import BuildModelConfig, TrainConfig

# Create test configs for quick validation
def create_test_configs():
    # Build config - smaller model for testing
    build_config = BuildModelConfig()
    build_config.use_gram = True
    build_config.gram_hidden_size = 30
    build_config.gram_dropout = 0.3
    
    build_config.use_chars = True
    build_config.char_max_word_length = 30
    build_config.char_embedding_dim = 10
    build_config.char_function_hidden_size = 128
    build_config.char_dropout = 0.3
    build_config.char_function_output_size = 64
    
    build_config.use_word_embeddings = False
    build_config.word_max_count = 10000
    build_config.use_trained_char_embeddings = False  # Disable for simple test
    
    build_config.rnn_input_size = 200
    build_config.rnn_hidden_size = 128
    build_config.rnn_n_layers = 2
    build_config.rnn_dropout = 0.3
    
    build_config.dense_size = 128
    build_config.dense_dropout = 0.3
    
    build_config.use_crf = False
    build_config.use_pos_lm = True
    build_config.use_word_lm = False
    
    build_config.save('rnnmorph/datasets/prepared/test_build_config.json')
    
    # Train config - 1 epoch, small batches
    train_config = TrainConfig()
    train_config.batch_size = 64  # Small batch for testing
    train_config.external_batch_size = 500  # Small external batch
    train_config.epochs_num = 1  # Just 1 epoch for testing
    train_config.val_part = 0.1  # 10% validation
    train_config.random_seed = 42
    train_config.dump_model_freq = 1
    train_config.rewrite_model = True
    
    # Set output paths
    train_config.train_model_config_path = 'rnnmorph/datasets/prepared/test_train_model.json'
    train_config.train_model_weights_path = 'rnnmorph/datasets/prepared/test_train_model.h5'
    train_config.eval_model_config_path = 'rnnmorph/datasets/prepared/test_eval_model.json'
    train_config.eval_model_weights_path = 'rnnmorph/datasets/prepared/test_eval_model.h5'
    # Let the model build new vectorizers from test data
    train_config.gram_dict_input = 'rnnmorph/datasets/prepared/test_gram_input.json'
    train_config.gram_dict_output = 'rnnmorph/datasets/prepared/test_gram_output.json'
    train_config.word_vocabulary = 'rnnmorph/datasets/prepared/test_word_vocabulary.pickle'
    train_config.char_set_path = 'rnnmorph/datasets/prepared/test_char_set.txt'
    train_config.rewrite_model = True  # Force rebuild
    
    train_config.save('rnnmorph/datasets/prepared/test_train_config.json')
    
    return 'rnnmorph/datasets/prepared/test_train_config.json', \
           'rnnmorph/datasets/prepared/test_build_config.json'

if __name__ == '__main__':
    print("="*70)
    print("RNNMorph Training Test - 1 Epoch on Small Dataset")
    print("="*70)
    
    # Create configs
    train_config_path, build_config_path = create_test_configs()
    print(f"\nCreated test configs:")
    print(f"  Train config: {train_config_path}")
    print(f"  Build config: {build_config_path}")
    
    # Training data
    train_file = 'rnnmorph/datasets/prepared/training_test_sentences.txt'
    print(f"\nTraining data: {train_file}")
    
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")
    
    # Run training
    train(
        file_names=[train_file],
        train_config_path=train_config_path,
        build_config_path=build_config_path,
        language='ru'
    )
    
    print("\n" + "="*70)
    print("✅ Training test completed successfully!")
    print("="*70)
