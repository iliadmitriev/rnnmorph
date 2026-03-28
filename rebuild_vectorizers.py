#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rebuild vectorizers from cleaned training data.

This script builds new vocabulary and grammeme vectorizers from the
cleaned training data (with PronType, Poss, Animacy removed).

Usage:
    python rebuild_vectorizers.py
"""

from rnnmorph.data_preparation.loader import Loader
import os

def rebuild_vectorizers():
    print("="*70)
    print("Rebuilding Vectorizers from Cleaned Training Data")
    print("="*70)
    
    # Create loader for Russian
    loader = Loader(language='ru')
    
    # Training data file
    training_file = 'rnnmorph/datasets/prepared/training_combined.txt'
    print(f"\nLoading training data: {training_file}")
    
    if not os.path.exists(training_file):
        print(f"ERROR: Training file not found: {training_file}")
        print("Run: python download_training_data.py --all")
        return None
    
    # Parse corpus - this builds all vectorizers
    print("Parsing corpus (this may take 5-15 minutes for large datasets)...")
    loader.parse_corpora([training_file])
    
    # Save outputs
    output_dir = 'rnnmorph/datasets/prepared'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving vectorizers to: {output_dir}")
    
    # Word vocabulary
    vocab_path = f'{output_dir}/new_word_vocabulary.pickle'
    loader.word_vocabulary.save(vocab_path)
    print(f"  ✓ Word vocabulary: {loader.word_vocabulary.size():,} words")
    print(f"    Saved: {vocab_path}")
    
    # Character set
    char_path = f'{output_dir}/new_char_set.txt'
    with open(char_path, 'w', encoding='utf-8') as f:
        f.write(loader.char_set)
    print(f"  ✓ Character set: {len(loader.char_set)} characters")
    print(f"    Saved: {char_path}")
    
    # Grammeme input vectorizer (from pymorphy2)
    gram_input_path = f'{output_dir}/new_gram_input.json'
    loader.grammeme_vectorizer_input.save(gram_input_path)
    print(f"  ✓ Input grammeme vectors: {loader.grammeme_vectorizer_input.size():,}")
    print(f"    Saved: {gram_input_path}")
    
    # Grammeme output vectorizer (from training data)
    gram_output_path = f'{output_dir}/new_gram_output.json'
    loader.grammeme_vectorizer_output.save(gram_output_path)
    print(f"  ✓ Output grammeme vectors: {loader.grammeme_vectorizer_output.size():,}")
    print(f"    Saved: {gram_output_path}")
    
    # Show statistics
    print(f"\n" + "="*70)
    print("Vectorizer Statistics")
    print("="*70)
    print(f"Word vocabulary:     {loader.word_vocabulary.size():>10,} words")
    print(f"Character set:       {len(loader.char_set):>10} characters")
    print(f"Input vectors:       {loader.grammeme_vectorizer_input.size():>10,} tags")
    print(f"Output vectors:      {loader.grammeme_vectorizer_output.size():>10,} tags")
    print(f"Input categories:    {loader.grammeme_vectorizer_input.grammemes_count():>10}")
    print(f"Output categories:   {loader.grammeme_vectorizer_output.grammemes_count():>10}")
    
    # Show sample output tags
    print(f"\nSample output tags (first 15):")
    sorted_tags = sorted(
        loader.grammeme_vectorizer_output.name_to_index.items(), 
        key=lambda x: x[1]
    )
    for i, (tag_name, idx) in enumerate(sorted_tags[:15]):
        print(f"  {idx:3d}: {tag_name}")
    
    # Verify cleaned tags
    print(f"\n" + "="*70)
    print("Verifying Tag Cleanup")
    print("="*70)
    
    prontype_tags = [t for t in loader.grammeme_vectorizer_output.name_to_index.keys() if 'PronType' in t]
    poss_tags = [t for t in loader.grammeme_vectorizer_output.name_to_index.keys() if 'Poss=' in t]
    animacy_tags = [t for t in loader.grammeme_vectorizer_output.name_to_index.keys() if 'Animacy=' in t]
    
    print(f"Tags with PronType: {len(prontype_tags):>10} (should be 0)")
    print(f"Tags with Poss:     {len(poss_tags):>10} (should be 0)")
    print(f"Tags with Animacy:  {len(animacy_tags):>10} (should be 0)")
    
    if len(prontype_tags) == 0 and len(poss_tags) == 0 and len(animacy_tags) == 0:
        print("\n✅ Vectorizers are CLEAN - ready for training!")
    else:
        print("\n⚠️  WARNING: Vectorizers still have problematic tags!")
        if prontype_tags:
            print(f"   PronType examples: {prontype_tags[:5]}")
        if poss_tags:
            print(f"   Poss examples: {poss_tags[:5]}")
        if animacy_tags:
            print(f"   Animacy examples: {animacy_tags[:5]}")
    
    # Create training config using new vectorizers
    print(f"\n" + "="*70)
    print("Creating Training Configuration")
    print("="*70)
    
    from rnnmorph.config import BuildModelConfig, TrainConfig
    
    # Build config
    build_config = BuildModelConfig()
    build_config_path = f'{output_dir}/build_config.json'
    build_config.save(build_config_path)
    print(f"  ✓ Build config: {build_config_path}")
    
    # Train config
    train_config = TrainConfig()
    train_config.epochs_num = 50
    train_config.batch_size = 256
    train_config.external_batch_size = 5000
    train_config.val_part = 0.05
    train_config.random_seed = 42
    train_config.dump_model_freq = 1
    train_config.rewrite_model = True
    
    # Point to NEW vectorizers
    train_config.gram_dict_input = gram_input_path
    train_config.gram_dict_output = gram_output_path
    train_config.word_vocabulary = vocab_path
    train_config.char_set_path = char_path
    
    # Model output paths
    train_config.train_model_config_path = f'{output_dir}/train_model.json'
    train_config.train_model_weights_path = f'{output_dir}/train_model.h5'
    train_config.eval_model_config_path = f'{output_dir}/eval_model.json'
    train_config.eval_model_weights_path = f'{output_dir}/eval_model.h5'
    
    train_config_path = f'{output_dir}/train_config.json'
    train_config.save(train_config_path)
    print(f"  ✓ Train config: {train_config_path}")
    
    print(f"\n" + "="*70)
    print("Next Steps")
    print("="*70)
    print("""
To train the model with the new vectorizers:

    python -c "
    from rnnmorph.train import train
    train(
        file_names=['rnnmorph/datasets/prepared/training_combined.txt'],
        train_config_path='rnnmorph/datasets/prepared/train_config.json',
        build_config_path='rnnmorph/datasets/prepared/build_config.json',
        language='ru'
    )
    "

Or use the test script for a quick validation:
    python test_training_quick.py

Expected training time:
  - 1 epoch on 7.8M words: ~15-30 minutes on GPU
  - 50 epochs (full): ~12-24 hours on GPU
""")
    
    print("="*70)
    print("✅ Vectorizer rebuild complete!")
    print("="*70)
    
    return loader

if __name__ == '__main__':
    rebuild_vectorizers()
