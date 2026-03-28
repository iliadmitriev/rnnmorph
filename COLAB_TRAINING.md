# RNNMorph Training on Google Colab

## Setup (5 minutes)

```python
# Mount Google Drive for data storage
from google.colab import drive
drive.mount('/content/drive')

# Install compatible dependencies
!pip install keras==2.8.0 tensorflow==2.8.0 h5py==3.6.0
!pip install pymorphy2 russian-tagsets tqdm jsonpickle
!pip install rnnmorph  # Install the package

# Download training data (example - replace with actual data)
# Option 1: From Drive
!cp /content/drive/MyDrive/training_data.txt /content/

# Option 2: From morphoRuEval repository
!git clone https://github.com/dialogue-evaluation/morphoRuEval-2017.git
```

## Training Configuration

```python
import json
from rnnmorph.config import BuildModelConfig, TrainConfig

# Create build config (model architecture)
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

build_config.use_word_embeddings = False  # Set True if you have word2vec
build_config.word_max_count = 10000

build_config.rnn_input_size = 200
build_config.rnn_hidden_size = 128
build_config.rnn_n_layers = 2
build_config.rnn_dropout = 0.3

build_config.dense_size = 128
build_config.dense_dropout = 0.3

build_config.use_crf = False
build_config.use_pos_lm = True  # Auxiliary task improves accuracy
build_config.use_word_lm = False

build_config.save('/content/build_config.json')

# Create train config
train_config = TrainConfig()
train_config.batch_size = 256
train_config.external_batch_size = 5000
train_config.epochs_num = 10  # Reduced for Colab (use 50 for full training)
train_config.val_part = 0.05
train_config.random_seed = 42
train_config.dump_model_freq = 1  # Save every batch

# Adjust paths for Colab
train_config.train_model_config_path = '/content/train_model.json'
train_config.train_model_weights_path = '/content/train_model.h5'
train_config.eval_model_config_path = '/content/eval_model.json'
train_config.eval_model_weights_path = '/content/eval_model.h5'
train_config.gram_dict_input = '/content/gram_input.json'
train_config.gram_dict_output = '/content/gram_output.json'
train_config.word_vocabulary = '/content/word_vocabulary.pickle'
train_config.char_set_path = '/content/char_set.txt'

train_config.save('/content/train_config.json')
```

## Training (6-12 hours for 10 epochs)

```python
from rnnmorph.train import train

# Train the model
train(
    file_names=['/content/training_data.txt'],
    train_config_path='/content/train_config.json',
    build_config_path='/content/build_config.json',
    language='ru',
    embeddings_path=None  # Add path to word2vec if available
)

# Save trained model to Drive
!cp /content/eval_model.h5 /content/drive/MyDrive/rnnmorph_model.h5
!cp /content/eval_model.json /content/drive/MyDrive/rnnmorph_model.json
!cp /content/*.json /content/drive/MyDrive/rnnmorph_configs/
!cp /content/*.pickle /content/drive/MyDrive/rnnmorph_configs/
!cp /content/char_set.txt /content/drive/MyDrive/rnnmorph_configs/
```

## Quick Test After Training

```python
from rnnmorph.predictor import RNNMorphPredictor

# Note: You need to copy all config files to use the model
predictor = RNNMorphPredictor(language='ru')
results = predictor.predict(['мама', 'мыла', 'раму'])

for r in results:
    print(f'{r.word}: {r.pos} - {r.tag}')
```

---

## Realistic Expectations for Colab

### Training Time Estimates

| Dataset Size | Epochs | Time on Colab T4 | Sessions Needed |
|-------------|--------|------------------|-----------------|
| 1M words | 10 | 3-4 hours | 1 |
| 1M words | 50 | 15-20 hours | 2 |
| 5M words | 10 | 8-10 hours | 1 |
| 5M words | 50 | 40-50 hours | 4-5 |

### Recommendations

1. **Start small**: Train on 100K-1M words first to verify setup
2. **Use fewer epochs**: 10 epochs gives ~90% of final accuracy
3. **Save checkpoints**: Copy to Drive every few hours
4. **Colab Pro**: Worth it for longer sessions (24 hours) and better GPUs

### Workarounds for Session Limits

```python
# Save checkpoint callback
import os

def save_checkpoint(epoch):
    !cp /content/eval_model.h5 /content/drive/MyDrive/checkpoint_epoch_{epoch}.h5
    print(f"Checkpoint saved at epoch {epoch}")

# Resume training from checkpoint
# (Requires modifying rnnmorph/train.py to load existing weights)
```

---

## Alternative: Use PyTorch Version

The PyTorch implementation I created might work better on Colab:

```python
# PyTorch is pre-installed on Colab
import torch
print(f"PyTorch {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# You would need to port the training code to PyTorch
# (currently only inference is implemented)
```

---

## Bottom Line

**Yes, it's feasible on Colab with these conditions:**

✅ **For experimentation/small models**: Free tier works great
✅ **For partial training** (10-20 epochs): Free tier sufficient  
⚠️ **For full training** (50 epochs, large dataset): Colab Pro recommended
⚠️ **For production models**: Use a dedicated GPU server

**Estimated cost for full training:**
- **Free Colab**: 2-5 sessions (need to restart)
- **Colab Pro** ($10/month): 1-2 sessions, faster GPU
- **Colab Pro+** ($50/month): 1 session, best GPU
