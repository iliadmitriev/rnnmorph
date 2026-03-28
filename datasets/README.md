# RNNMorph Training Data - Complete Guide

## ✅ All Datasets Downloaded & Prepared!

| Dataset | Words | Size | Status |
|---------|-------|------|--------|
| **UD SynTagRus** | 1,517,763 | 18 MB | ✅ Ready |
| **UD Taiga** | 1,758,937 | 19 MB | ✅ Ready |
| **UD GSD** | 97,994 | 656 KB | ✅ Ready |
| **UD PUD** | 19,355 | 1.1 MB | ✅ Ready |
| **OpenCorpora** | 1,989,538 | 74 MB | ✅ Ready |
| **RNC Open** | 1,345,157 | 58 MB | ✅ Ready |
| **GIKRYA** | 1,086,148 | 54 MB | ✅ Ready |
| **TOTAL** | **7,814,892** | **225 MB** | ✅ **READY FOR TRAINING** |

## Quick Start

```bash
# Download and prepare all available data
python download_training_data.py --all

# This will download:
# - UD Russian corpora (SynTagRus, Taiga, GSD, PUD)
# - OpenCorpora annotated corpus
# - RNC Open (Russian National Corpus)
# - GIKRYA (Internet corpus)
# - Sample dataset for testing
```

## What You Get

After running the script, you'll have:

```
rnnmorph/datasets/prepared/
├── training_combined.txt     (225 MB, 7.8M words)  ← Use this for training
├── ud_combined.txt           (39 MB, 3.4M words)
├── opencorpora_annotated.txt (74 MB, 2.0M words)
├── rnc_texts.txt             (58 MB, 1.3M words)
├── gikrya_texts.txt          (54 MB, 1.1M words)
└── sample_training.txt       (4.7 KB, 82 words)
```

## Training

```python
from rnnmorph.train import train

# Train with all available data (7.8M words)
train(
    file_names=['rnnmorph/datasets/prepared/training_combined.txt'],
    train_config_path='rnnmorph/models/ru/train_config.json',
    build_config_path='rnnmorph/models/ru/build_config.json',
    language='ru'
)
```

**Expected Results:**
- **POS Accuracy**: ~97-98%
- **Full Tag Accuracy**: ~94-96%
- **Training Time**: 12-18 hours on GPU (RTX 3090/A100)

## Data Format

All prepared data uses RNNMorph format (tab-separated):
```
word<TAB>lemma<TAB>POS<TAB>grammemes
```

Example:
```
мама	мама	NOUN	Case=Nom|Gender=Fem|Number=Sing
мыла	мыть	VERB	Mood=Ind|Number=Sing|Person=3|Tense=Notpast
раму	рама	NOUN	Case=Acc|Gender=Fem|Number=Sing
```

Sentences are separated by blank lines.

## macOS Specific Notes

The script automatically uses **`unar`** (The Unarchiver) for RAR files on macOS:

```bash
# Install if needed
brew install unar

# The script will use it automatically for RNC extraction
```

## Expected Model Performance

| Training Data | Words | POS Accuracy | Full Tag | Training Time (GPU) |
|--------------|-------|--------------|----------|---------------------|
| Sample only | 82 | - | - | <1 min |
| UD corpora | 3.4M | ~94-96% | ~90-93% | 4-6 hours |
| UD + RNC + GIKRYA | 5.8M | ~96-97% | ~93-95% | 8-12 hours |
| **Current (ALL)** | **7.8M** | **~97-98%** | **~94-96%** | **12-18 hours** |
| Published model | ~12M+ | **98.26%** | **95.81%** | 18-24 hours |

## Troubleshooting

### OpenCorpora download fails
- Server may block automated downloads
- **Solution**: Download manually from https://opencorpora.org/?page=downloads
- Select: "Whole corpus with resolved homonymy" (ZIP format)

### RNC extraction fails
- **macOS**: Install `unar`: `brew install unar`
- **Linux**: Install `unrar`: `sudo apt-get install unrar`

### Out of memory during preparation
- **Solution**: Process corpora individually:
  ```bash
  python download_training_data.py --ud --no-prepare
  python download_training_data.py --opencorpora --no-prepare
  python download_training_data.py --rnc --no-prepare
  ```

### Training too slow on CPU
- CPU training: ~3-5 days for full dataset
- **Solution**: Use GPU (Colab, local GPU, or cloud)

## License Summary

| Dataset | License | Commercial Use |
|---------|---------|----------------|
| UD Corpora | CC BY-SA | ✅ Yes (attribution) |
| OpenCorpora | CC BY | ✅ Yes (attribution) |
| RNC Open | Academic | ✅ Research |
| GIKRYA | Academic | ✅ Research |

## Download Script Options

```bash
# All available data
python download_training_data.py --all

# Specific datasets
python download_training_data.py --ud         # UD corpora only
python download_training_data.py --opencorpora  # OpenCorpora only
python download_training_data.py --rnc        # RNC only
python download_training_data.py --gikrya     # GIKRYA only
python download_training_data.py --sample     # Sample for testing

# Skip preparation (faster re-runs)
python download_training_data.py --no-prepare

# Custom output directory
python download_training_data.py --output-dir /path/to/data
```

## Contact & Resources

- **morphoRuEval-2017**: https://github.com/dialogue-evaluation/morphoRuEval-2017
- **UD Russian**: https://universaldependencies.org/russian.html
- **OpenCorpora**: https://opencorpora.org/
- **RNC**: https://ruscorpora.ru/
- **RNNMorph**: https://github.com/IlyaGusev/rnnmorph
