# How to Rebuild Vectorizers for RNNMorph Training

## Overview

Vectorizers are the vocabulary and tag mappings that the model needs to convert text into numerical representations. When you change the tag format (like we did by removing PronType, Poss, Animacy), you need to rebuild them.

## What Gets Built

1. **Word Vocabulary** (`word_vocabulary.pickle`) - Maps words to indices
2. **Character Set** (`char_set.txt`) - All unique characters in the corpus
3. **Grammeme Input Vectorizer** (`gram_input.json`) - Maps pymorphy2 tags to vectors
4. **Grammeme Output Vectorizer** (`gram_output.json`) - Maps training tags to vectors

## Method 1: Using the Training Pipeline (Recommended)

The easiest way is to let the training pipeline build them automatically:

```python
from rnnmorph.train import train
from rnnmorph.config import BuildModelConfig, TrainConfig

# Configure training
build_config = BuildModelConfig()
build_config.save('build_config.json')

train_config = TrainConfig()
train_config.epochs_num = 50
train_config.batch_size = 256
train_config.val_part = 0.05

# Set paths for NEW vectorizers (will be created)
train_config.gram_dict_input = 'my_new_gram_input.json'
train_config.gram_dict_output = 'my_new_gram_output.json'
train_config.word_vocabulary = 'my_new_word_vocabulary.pickle'
train_config.char_set_path = 'my_new_char_set.txt'
train_config.save('train_config.json')

# Train - vectorizers will be built automatically from your data!
train(
    file_names=['rnnmorph/datasets/prepared/training_combined.txt'],
    train_config_path='train_config.json',
    build_config_path='build_config.json',
    language='ru'
)
```

The `model.prepare()` method in `rnnmorph/model.py` will:
1. Check if vectorizer files exist
2. If NOT, build them from your training data
3. Save them for future use

## Method 2: Manual Vectorizer Building

If you want to build vectorizers separately (for inspection or debugging):

```python
from rnnmorph.data_preparation.loader import Loader
from rnnmorph.data_preparation.grammeme_vectorizer import GrammemeVectorizer
from rnnmorph.data_preparation.word_vocabulary import WordVocabulary

# Create loader for Russian
loader = Loader(language='ru')

# Parse your training corpus
# This builds vocabularies and vectorizers from the data
loader.parse_corpora(['rnnmorph/datasets/prepared/training_combined.txt'])

# Save the built components
loader.word_vocabulary.save('my_word_vocabulary.pickle')

with open('my_char_set.txt', 'w', encoding='utf-8') as f:
    f.write(loader.char_set)

loader.grammeme_vectorizer_input.save('my_gram_input.json')
loader.grammeme_vectorizer_output.save('my_gram_output.json')

# Print statistics
print(f"Word vocabulary size: {loader.word_vocabulary.size():,}")
print(f"Character set size: {len(loader.char_set)}")
print(f"Input grammeme vectors: {loader.grammeme_vectorizer_input.size():,}")
print(f"Output grammeme vectors: {loader.grammeme_vectorizer_output.size():,}")
print(f"Input grammeme categories: {loader.grammeme_vectorizer_input.grammemes_count()}")
print(f"Output grammeme categories: {loader.grammeme_vectorizer_output.grammemes_count()}")
```

## Method 3: Quick Script for Your Cleaned Data

Here's a complete script to rebuild vectorizers from your cleaned training data:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rebuild vectorizers from cleaned training data.
"""

from rnnmorph.data_preparation.loader import Loader
import json

def rebuild_vectorizers():
    print("="*70)
    print("Rebuilding Vectorizers from Cleaned Training Data")
    print("="*70)
    
    # Create loader
    loader = Loader(language='ru')
    
    # Training data file
    training_file = 'rnnmorph/datasets/prepared/training_combined.txt'
    print(f"\nLoading training data: {training_file}")
    
    # Parse corpus - this builds all vectorizers
    loader.parse_corpora([training_file])
    
    # Save outputs
    output_dir = 'rnnmorph/datasets/prepared'
    
    print(f"\nSaving vectorizers to: {output_dir}")
    
    loader.word_vocabulary.save(f'{output_dir}/new_word_vocabulary.pickle')
    print(f"  ✓ Word vocabulary: {loader.word_vocabulary.size():,} words")
    
    with open(f'{output_dir}/new_char_set.txt', 'w', encoding='utf-8') as f:
        f.write(loader.char_set)
    print(f"  ✓ Character set: {len(loader.char_set)} characters")
    
    loader.grammeme_vectorizer_input.save(f'{output_dir}/new_gram_input.json')
    print(f"  ✓ Input grammeme vectors: {loader.grammeme_vectorizer_input.size():,}")
    
    loader.grammeme_vectorizer_output.save(f'{output_dir}/new_gram_output.json')
    print(f"  ✓ Output grammeme vectors: {loader.grammeme_vectorizer_output.size():,}")
    
    # Show sample tags
    print(f"\nSample output tags (first 10):")
    for i, (tag_name, idx) in enumerate(sorted(
        loader.grammeme_vectorizer_output.name_to_index.items(), 
        key=lambda x: x[1]
    )[:10]):
        print(f"  {idx:3d}: {tag_name}")
    
    print("\n" + "="*70)
    print("✅ Vectorizers rebuilt successfully!")
    print("="*70)
    
    return loader

if __name__ == '__main__':
    rebuild_vectorizers()
```

## Verifying the New Vectorizers

After building, verify they match your cleaned data format:

```python
from rnnmorph.data_preparation.grammeme_vectorizer import GrammemeVectorizer

# Load new output vectorizer
gv = GrammemeVectorizer()
gv.load('rnnmorph/datasets/prepared/new_gram_output.json')

# Check for problematic tags (should all be 0)
prontype_tags = [t for t in gv.name_to_index.keys() if 'PronType' in t]
poss_tags = [t for t in gv.name_to_index.keys() if 'Poss=' in t]
animacy_tags = [t for t in gv.name_to_index.keys() if 'Animacy=' in t]

print(f"Tags with PronType: {len(prontype_tags)}")
print(f"Tags with Poss: {len(poss_tags)}")
print(f"Tags with Animacy: {len(animacy_tags)}")

if len(prontype_tags) == 0 and len(poss_tags) == 0 and len(animacy_tags) == 0:
    print("✅ Vectorizers are clean!")
else:
    print("⚠️  Vectorizers still have problematic tags!")
```

## Using New Vectorizers for Training

Once built, update your training config to use them:

```python
from rnnmorph.config import TrainConfig

train_config = TrainConfig()
train_config.gram_dict_input = 'rnnmorph/datasets/prepared/new_gram_input.json'
train_config.gram_dict_output = 'rnnmorph/datasets/prepared/new_gram_output.json'
train_config.word_vocabulary = 'rnnmorph/datasets/prepared/new_word_vocabulary.pickle'
train_config.char_set_path = 'rnnmorph/datasets/prepared/new_char_set.txt'
train_config.save('train_config.json')
```

## Expected Output Statistics

For your cleaned training data (7.8M words), expect:

| Component | Expected Size |
|-----------|--------------|
| Word Vocabulary | ~100,000-200,000 words |
| Character Set | ~100-150 characters |
| Output Grammeme Vectors | ~200-300 tags |
| Output Grammeme Categories | ~50-60 categories |

**Note**: The output vectorizer will be SMALLER than the original (253 tags) because we removed PronType, Poss, Animacy, etc.

## Complete Training Pipeline

Here's the complete flow:

```bash
# Step 1: Rebuild vectorizers
python rebuild_vectorizers.py

# Step 2: Update training config to use new vectorizers
# (edit train_config.json or use the code above)

# Step 3: Train the model
python -c "
from rnnmorph.train import train
train(
    file_names=['rnnmorph/datasets/prepared/training_combined.txt'],
    train_config_path='train_config.json',
    build_config_path='build_config.json',
    language='ru'
)
"
```

## Troubleshooting

### KeyError during training
**Problem**: `KeyError: 'VERB#SomeTag'`
**Cause**: Training data has tags not in vectorizer
**Solution**: Make sure vectorizers were built from the SAME data you're training on

### Vectorizer size mismatch
**Problem**: Model output layer has wrong size
**Cause**: Using old vectorizer with new data
**Solution**: Always rebuild vectorizers when changing tag format

### Slow vectorizer building
**Problem**: Takes too long to build
**Solution**: This is normal for large corpora (7.8M words = 5-15 minutes)
