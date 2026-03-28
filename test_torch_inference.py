#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for PyTorch RNNMorph inference.
Compares outputs between Keras and PyTorch models.
"""

import os
import sys

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_torch_model():
    """Test the PyTorch inference model."""
    from rnnmorph.torch_inference import RNNMorphInference

    print("Loading PyTorch model...")
    model = RNNMorphInference(
        model_dir="rnnmorph/models/ru", language="ru", device="cpu"
    )

    # Test sentences
    test_sentences = [
        ["мама", "мыла", "раму"],
        ["кот", "сидит", "на", "окне"],
        ["привет", ",", "как", "дела", "?"],
    ]

    print("\n" + "=" * 60)
    print("PyTorch Model Predictions")
    print("=" * 60)

    for sentence in test_sentences:
        print(f"\nSentence: {' '.join(sentence)}")
        print("-" * 50)
        results = model.predict(sentence)
        for result in results:
            print(
                f"  {result['word']:15} -> {result['pos']:6} | {result['tag']:30} (score: {result['score']:.4f})"
            )

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


def compare_with_keras():
    """Compare PyTorch outputs with original Keras model."""
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    print("\n" + "=" * 60)
    print("Comparing PyTorch vs Keras outputs")
    print("=" * 60)

    # Load PyTorch model
    from rnnmorph.torch_inference import RNNMorphInference

    torch_model = RNNMorphInference(
        model_dir="rnnmorph/models/ru", language="ru", device="cpu"
    )

    # Load Keras model
    from rnnmorph.predictor import RNNMorphPredictor

    keras_model = RNNMorphPredictor(language="ru")

    test_sentences = [
        ["мама", "мыла", "раму"],
        ["кот", "сидит", "на", "окне"],
        ["привет", ",", "как", "дела", "?"],
    ]

    total_matches = 0
    total_words = 0

    for sentence in test_sentences:
        print(f"\nSentence: {' '.join(sentence)}")
        print("-" * 50)

        keras_results = keras_model.predict(sentence)
        torch_results = torch_model.predict(sentence)

        for k, t in zip(keras_results, torch_results):
            match = k.pos == t["pos"]
            if match:
                total_matches += 1
            total_words += 1
            status = "✓" if match else "✗"
            print(f"  {k.word:<10} Keras: {k.pos:<8} Torch: {t['pos']:<8} {status}")

    print("\n" + "=" * 60)
    print(
        f"POS Match rate: {total_matches}/{total_words} ({100 * total_matches / total_words:.1f}%)"
    )
    print("=" * 60)


def test_batch_inference():
    """Test batch inference."""
    from rnnmorph.torch_inference import RNNMorphInference

    model = RNNMorphInference(
        model_dir="rnnmorph/models/ru", language="ru", device="cpu"
    )

    sentences = [
        ["мама", "мыла", "раму"],
        ["кот", "сидит", "на", "окне"],
        ["привет", ",", "как", "дела", "?"],
    ]

    print("\n" + "=" * 60)
    print("Batch Inference Test")
    print("=" * 60)

    results = model.predict_batch(sentences, batch_size=2)

    for i, (sentence, result) in enumerate(zip(sentences, results)):
        print(f"\nSentence {i + 1}: {' '.join(sentence)}")
        for r in result:
            print(f"  {r['word']:15} -> {r['pos']:6} | {r['tag']:<30}")


def test_probabilities():
    """Test probability output."""
    import numpy as np

    from rnnmorph.torch_inference import RNNMorphInference

    model = RNNMorphInference(
        model_dir="rnnmorph/models/ru", language="ru", device="cpu"
    )

    sentence = ["мама", "мыла", "раму"]
    probs = model.get_tag_probabilities(sentence)

    print("\n" + "=" * 60)
    print("Probability Distribution Test")
    print("=" * 60)
    print(f"\nSentence: {' '.join(sentence)}")
    print(f"Probability matrix shape: {probs.shape}")

    for i, word in enumerate(sentence):
        top5_idx = np.argsort(probs[i])[-5:][::-1]
        print(f"\n{word}:")
        for idx in top5_idx:
            print(f"  Tag {idx}: {probs[i][idx]:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test PyTorch RNNMorph inference")
    parser.add_argument(
        "--compare", action="store_true", help="Compare with Keras model"
    )
    parser.add_argument("--batch", action="store_true", help="Test batch inference")
    parser.add_argument("--probs", action="store_true", help="Test probability output")
    args = parser.parse_args()

    test_torch_model()

    if args.compare:
        compare_with_keras()

    if args.batch:
        test_batch_inference()

    if args.probs:
        test_probabilities()
