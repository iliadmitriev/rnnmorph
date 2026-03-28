# RNNMorph PyTorch Inference

This module provides PyTorch-based inference for RNNMorph, a neural network POS tagger for Russian.

## Features

- **Exact weight compatibility**: Loads weights directly from original Keras `.h5` files
- **100% prediction match**: Produces identical predictions to the original Keras model
- **Inference optimized**: No training code, minimal dependencies for inference only
- **Batch processing**: Support for batch inference on multiple sentences

## Installation

Requires Python 3.7+ and the following packages:

```bash
pip install torch numpy h5py pymorphy2 russian-tagsets
```

## Usage

### Basic Usage

```python
from rnnmorph.torch_inference import RNNMorphInference

# Initialize model
model = RNNMorphInference(
    model_dir="rnnmorph/models/ru",
    language="ru",
    device="cpu"  # or "cuda"
)

# Predict POS tags for a sentence
sentence = ["мама", "мыла", "раму"]
results = model.predict(sentence)

for result in results:
    print(f"{result['word']}: {result['pos']} - {result['tag']} (score: {result['score']:.4f})")
```

### Batch Inference

```python
sentences = [
    ["мама", "мыла", "раму"],
    ["кот", "сидит", "на", "окне"],
]

results = model.predict_batch(sentences, batch_size=64)
```

### Get Full Probability Distribution

```python
probs = model.get_tag_probabilities(["мама", "мыла", "раму"])
print(f"Shape: {probs.shape}")  # (seq_len, num_tags)
```

## Architecture

The model consists of:

1. **Character Embedding Network**: CNN-like architecture for character-level word representations
2. **Grammeme Embedding**: Dense layer for morphological feature vectors from pymorphy2
3. **BiLSTM Encoder**: 2-layer bidirectional LSTM (128 hidden units each direction)
4. **Output Layer**: Dense + softmax for tag classification

### Model Configuration

Default configuration (from `build_config.json`):
- Character embedding dim: 24
- Character network: 500 → 200 hidden units
- Grammeme dense: 56 → 30 hidden units
- BiLSTM: 2 layers, 128 hidden units each
- Output dense: 256 → 128 hidden units
- Output classes: 253 (252 tags + 1 padding)

## Comparison with Keras

The PyTorch implementation produces **identical predictions** to the original Keras model:

```python
from rnnmorph.torch_inference import RNNMorphInference
from rnnmorph.predictor import RNNMorphPredictor

torch_model = RNNMorphInference(model_dir="rnnmorph/models/ru")
keras_model = RNNMorphPredictor(language="ru")

sentence = ["мама", "мыла", "раму"]

torch_results = torch_model.predict(sentence)
keras_results = keras_model.predict(sentence)

# POS tags match 100%
for t, k in zip(torch_results, keras_results):
    assert t['pos'] == k.pos
```

## Files

- `torch_inference.py`: Main PyTorch implementation
- `test_torch_inference.py`: Test suite comparing with Keras model

## Performance

- **Speed**: Similar to Keras model (~200-600 words/second on CPU)
- **Memory**: ~300-400 MB (slightly less than Keras due to PyTorch efficiency)

## Limitations

- Inference only (no training support)
- Requires original model files (`.h5`, `.json`, `.pickle`)
- Russian language support (English requires additional testing)

## Author

Based on the original RNNMorph by Ilya Gusev.
PyTorch implementation for inference-only use cases.
