# -*- coding: utf-8 -*-
"""
Minimal PyTorch inference module for RNNMorph.
Contains only the neural network and weight loading - no external NLP dependencies.

Usage:
    from rnnmorph.torch_inference import RNNMorphInference
    
    model = RNNMorphInference(model_dir="rnnmorph/models/ru")
    results = model.predict(["мама", "мыла", "раму"])
"""

import json
import pickle
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

# Import original feature preparation modules
from rnnmorph.data_preparation.grammeme_vectorizer import GrammemeVectorizer
from rnnmorph.data_preparation.word_vocabulary import WordVocabulary
from rnnmorph.batch_generator import BatchGenerator
from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters


class ReversedLSTM(nn.Module):
    """LSTM that processes sequence in reverse order (right-to-left)."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reverse sequence, run LSTM, reverse output
        x_reversed = torch.flip(x, dims=[1])
        output, _ = self.lstm(x_reversed)
        return torch.flip(output, dims=[1])


class CharEmbeddingNetwork(nn.Module):
    """Character-level embedding network for word representation."""
    
    def __init__(self, char_vocab_size: int, char_emb_dim: int,
                 max_word_length: int, hidden_size: int, 
                 output_size: int, dropout: float = 0.0):
        super().__init__()
        self.max_word_length = max_word_length
        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim)
        self.dense1 = nn.Linear(char_emb_dim * max_word_length, hidden_size)
        self.dense2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, word_len = chars.shape
        
        # Embed characters: (batch, seq_len, word_len, char_emb_dim)
        char_emb = self.char_embedding(chars)
        
        # Flatten word dimension: (batch, seq_len, word_len * char_emb_dim)
        char_emb = char_emb.reshape(batch_size, seq_len, -1)
        
        if self.dropout:
            char_emb = self.dropout(char_emb)
        
        # Dense layers with ReLU
        char_emb = F.relu(self.dense1(char_emb))
        if self.dropout:
            char_emb = self.dropout(char_emb)
        char_emb = self.dense2(char_emb)
        if self.dropout:
            char_emb = self.dropout(char_emb)
            
        return char_emb


class RNNMorphNN(nn.Module):
    """
    Pure neural network for RNNMorph POS tagging.
    This is the inference-only PyTorch implementation.
    """
    
    def __init__(self, config: Dict[str, Any], gram_input_size: int, 
                 num_output_classes: int, char_vocab_size: int):
        super().__init__()
        self.config = config
        
        # Get config values with defaults from original model
        gram_hidden_size = config.get('gram_hidden_size', 30)
        gram_dropout = config.get('gram_dropout', 0.2)
        
        char_emb_dim = config.get('char_embedding_dim', 24)
        char_max_word_length = config.get('char_max_word_length', 32)
        char_hidden_size = config.get('char_function_hidden_size', 500)
        char_output_size = config.get('char_function_output_size', 200)
        char_dropout = config.get('char_dropout', 0.2)
        
        rnn_input_size = config.get('rnn_input_size', 200)
        rnn_hidden_size = config.get('rnn_hidden_size', 128)
        rnn_n_layers = config.get('rnn_n_layers', 2)
        rnn_dropout = config.get('rnn_dropout', 0.3)
        
        dense_size = config.get('dense_size', 128)
        dense_dropout = config.get('dense_dropout', 0.2)
        
        # Grammeme embedding branch
        self.gram_dropout = nn.Dropout(gram_dropout)
        self.gram_dense = nn.Linear(gram_input_size, gram_hidden_size)
        
        # Character embedding branch
        self.char_network = CharEmbeddingNetwork(
            char_vocab_size=char_vocab_size,
            char_emb_dim=char_emb_dim,
            max_word_length=char_max_word_length,
            hidden_size=char_hidden_size,
            output_size=char_output_size,
            dropout=char_dropout
        )
        
        # Input projection
        combined_input_size = gram_hidden_size + char_output_size
        self.input_dense = nn.Linear(combined_input_size, rnn_input_size)
        
        # First BiLSTM layer (separate forward and backward)
        self.lstm_forward_1 = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.lstm_backward_1 = ReversedLSTM(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size
        )
        
        # Additional BiLSTM layers
        self.rnn_layers = nn.ModuleList()
        for i in range(rnn_n_layers - 1):
            self.rnn_layers.append(nn.LSTM(
                input_size=rnn_hidden_size * 2,
                hidden_size=rnn_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            ))
        
        # Output projection with BatchNorm
        self.output_dense = nn.Linear(rnn_hidden_size * 2, dense_size)
        self.output_dropout = nn.Dropout(dense_dropout)
        self.output_bn = nn.BatchNorm1d(dense_size, eps=0.001, momentum=0.99)
        
        # Final classification
        self.classifier = nn.Linear(dense_size, num_output_classes)
        
    def forward(self, grammemes: torch.Tensor, chars: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            grammemes: (batch, seq_len, gram_input_size) - grammeme vectors
            chars: (batch, seq_len, max_word_length) - character indices
            
        Returns:
            (batch, seq_len, num_classes) - logits for each tag
        """
        batch_size, seq_len = grammemes.shape[:2]
        
        # Grammeme branch
        gram_emb = self.gram_dropout(grammemes)
        gram_emb = F.relu(self.gram_dense(gram_emb))
        
        # Character branch
        char_emb = self.char_network(chars)
        
        # Combine and project
        combined = torch.cat([gram_emb, char_emb], dim=-1)
        x = F.relu(self.input_dense(combined))
        
        # First BiLSTM layer
        lstm1_out_fwd, _ = self.lstm_forward_1(x)
        lstm1_out_bwd = self.lstm_backward_1(x)
        x = torch.cat([lstm1_out_fwd, lstm1_out_bwd], dim=-1)
        
        # Additional BiLSTM layers
        for rnn_layer in self.rnn_layers:
            x, _ = rnn_layer(x)
        
        # Output projection
        x = self.output_dense(x)
        x = self.output_dropout(x)
        
        # BatchNorm (applied per timestep)
        x = x.reshape(-1, x.shape[-1])
        x = self.output_bn(x)
        x = x.reshape(batch_size, seq_len, -1)
        x = F.relu(x)
        
        # Classification
        output = self.classifier(x)
        
        return output


def load_keras_weights(model: RNNMorphNN, keras_weights_path: str) -> None:
    """
    Load weights from Keras .h5 file into PyTorch model.
    
    The weight matrices are transposed because Keras and PyTorch
    use different conventions for weight storage.
    """
    with h5py.File(keras_weights_path, 'r') as f:
        # Grammeme dense layer (dense_1)
        model.gram_dense.weight.data = torch.from_numpy(
            f['dense_1/dense_1/kernel:0'][()].T)
        model.gram_dense.bias.data = torch.from_numpy(
            f['dense_1/dense_1/bias:0'][()])
        
        # Character embeddings (inside time_distributed_1)
        model.char_network.char_embedding.weight.data = torch.from_numpy(
            f['time_distributed_1/chars_embeddings/embeddings:0'][()])
        
        # Char dense1 (dense_2 inside time_distributed_1)
        model.char_network.dense1.weight.data = torch.from_numpy(
            f['time_distributed_1/dense_2/kernel:0'][()].T)
        model.char_network.dense1.bias.data = torch.from_numpy(
            f['time_distributed_1/dense_2/bias:0'][()])
        
        # Char dense2 (dense_3 inside time_distributed_1)
        model.char_network.dense2.weight.data = torch.from_numpy(
            f['time_distributed_1/dense_3/kernel:0'][()].T)
        model.char_network.dense2.bias.data = torch.from_numpy(
            f['time_distributed_1/dense_3/bias:0'][()])
        
        # Input dense (dense_4)
        model.input_dense.weight.data = torch.from_numpy(
            f['dense_4/dense_4/kernel:0'][()].T)
        model.input_dense.bias.data = torch.from_numpy(
            f['dense_4/dense_4/bias:0'][()])
        
        # LSTM layer 1 forward
        _load_lstm_weights(model.lstm_forward_1, f, 'LSTM_1_forward')
        
        # LSTM layer 1 backward
        _load_lstm_weights(model.lstm_backward_1.lstm, f, 'LSTM_1_backward')
        
        # Additional BiLSTM layers (bidirectional_1 contains LSTM_0)
        for i, rnn_layer in enumerate(model.rnn_layers):
            _load_bidirectional_lstm_weights(rnn_layer, f, 'bidirectional_1')
        
        # Output dense (time_distributed_2)
        model.output_dense.weight.data = torch.from_numpy(
            f['time_distributed_2/time_distributed_2/kernel:0'][()].T)
        model.output_dense.bias.data = torch.from_numpy(
            f['time_distributed_2/time_distributed_2/bias:0'][()])
        
        # BatchNorm (time_distributed_4)
        bn = f['time_distributed_4/time_distributed_4']
        model.output_bn.weight.data = torch.from_numpy(bn['gamma:0'][()])
        model.output_bn.bias.data = torch.from_numpy(bn['beta:0'][()])
        model.output_bn.running_mean.data = torch.from_numpy(bn['moving_mean:0'][()])
        model.output_bn.running_var.data = torch.from_numpy(bn['moving_variance:0'][()])
        
        # Classifier (main_pred)
        model.classifier.weight.data = torch.from_numpy(
            f['main_pred/main_pred/kernel:0'][()].T)
        model.classifier.bias.data = torch.from_numpy(
            f['main_pred/main_pred/bias:0'][()])


def save_torch_model(model: RNNMorphNN, config: Dict[str, Any],
                     gram_input_data: Dict, gram_output_data: Dict,
                     word_vocab, char_set: str, save_path: str) -> None:
    """
    Save PyTorch model with all required data to a single .pt file.
    
    Args:
        model: PyTorch model to save
        config: Model configuration dict
        gram_input_data: Grammemes input vectorizer data
        gram_output_data: Grammemes output vectorizer data
        word_vocab: Word vocabulary object
        char_set: Character set string
        save_path: Path to save .pt file
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'gram_input': gram_input_data,
        'gram_output': gram_output_data,
        'word_vocabulary': word_vocab,
        'char_set': char_set,
    }
    torch.save(checkpoint, save_path)


def load_torch_model(load_path: str, device: str = 'cpu') -> Tuple[RNNMorphNN, Dict, Dict, Dict, Any, str]:
    """
    Load PyTorch model from .pt file.
    
    Args:
        load_path: Path to .pt file
        device: Device to load model to
        
    Returns:
        Tuple of (model, config, gram_input_data, gram_output_data, word_vocab, char_set)
    """
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    gram_input_data = checkpoint['gram_input']
    gram_output_data = checkpoint['gram_output']
    word_vocab = checkpoint['word_vocabulary']
    char_set = checkpoint['char_set']
    
    # Recreate model
    gram_input_size = 56  # Fixed for Russian
    num_classes = len(gram_output_data['name_to_index']) + 1
    char_vocab_size = len(char_set) + 1
    
    model = RNNMorphNN(
        config=config,
        gram_input_size=gram_input_size,
        num_output_classes=num_classes,
        char_vocab_size=char_vocab_size
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config, gram_input_data, gram_output_data, word_vocab, char_set


def _load_lstm_weights(lstm: nn.LSTM, h5: h5py.File, name: str) -> None:
    """Load weights for single LSTM from Keras format."""
    prefix = f'{name}/{name}'
    kernel = h5[f'{prefix}/kernel:0'][()]
    recurrent_kernel = h5[f'{prefix}/recurrent_kernel:0'][()]
    bias = h5[f'{prefix}/bias:0'][()]
    
    lstm.weight_ih_l0.data = torch.from_numpy(kernel.T)
    lstm.weight_hh_l0.data = torch.from_numpy(recurrent_kernel.T)
    lstm.bias_ih_l0.data = torch.from_numpy(bias)
    lstm.bias_hh_l0.data.zero_()


def _load_bidirectional_lstm_weights(bi_lstm: nn.LSTM, h5: h5py.File, 
                                      name: str) -> None:
    """Load weights for Bidirectional LSTM from Keras format."""
    # Forward
    fwd_prefix = f'{name}/{name}/forward_LSTM_0'
    bi_lstm.weight_ih_l0.data = torch.from_numpy(
        h5[f'{fwd_prefix}/kernel:0'][()].T)
    bi_lstm.weight_hh_l0.data = torch.from_numpy(
        h5[f'{fwd_prefix}/recurrent_kernel:0'][()].T)
    bi_lstm.bias_ih_l0.data = torch.from_numpy(
        h5[f'{fwd_prefix}/bias:0'][()])
    bi_lstm.bias_hh_l0.data.zero_()
    
    # Backward
    bwd_prefix = f'{name}/{name}/backward_LSTM_0'
    bi_lstm.weight_ih_l0_reverse.data = torch.from_numpy(
        h5[f'{bwd_prefix}/kernel:0'][()].T)
    bi_lstm.weight_hh_l0_reverse.data = torch.from_numpy(
        h5[f'{bwd_prefix}/recurrent_kernel:0'][()].T)
    bi_lstm.bias_ih_l0_reverse.data = torch.from_numpy(
        h5[f'{bwd_prefix}/bias:0'][()])
    bi_lstm.bias_hh_l0_reverse.data.zero_()


class RNNMorphInference:
    """
    Minimal inference-only class for RNNMorph.
    
    This class contains only what's needed for neural network inference:
    - The PyTorch model
    - Weight loading from Keras format
    - Feature preparation using original BatchGenerator
    - Prediction with softmax probabilities
    """
    
    def __init__(self, model_dir: str, language: str = 'ru', 
                 device: Optional[str] = None):
        """
        Initialize the inference model.
        
        Args:
            model_dir: Path to directory with model files
            language: Language code ('ru' or 'en')
            device: Device for inference ('cpu', 'cuda', or None for auto)
        """
        self.language = language
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir  # Store for save() method
        
        # Load config
        with open(f'{model_dir}/build_config.json', 'r') as f:
            self.config = json.load(f)
        
        # Load grammeme vectorizers
        self.gram_vectorizer_input = GrammemeVectorizer()
        self.gram_vectorizer_input.load(f'{model_dir}/gram_input.json')
        
        self.gram_vectorizer_output = GrammemeVectorizer()
        self.gram_vectorizer_output.load(f'{model_dir}/gram_output.json')
        
        with open(f'{model_dir}/gram_output.json', 'r') as f:
            gram_output = json.load(f)
        self.gram_output_vectors = gram_output['vectors']
        
        # Load word vocabulary
        self.word_vocab = WordVocabulary()
        self.word_vocab.load(f'{model_dir}/word_vocabulary.pickle')
        
        # Load char set
        with open(f'{model_dir}/char_set.txt', 'r', encoding='utf-8') as f:
            self.char_set = f.read().rstrip()
        
        # Initialize morphological analyzer
        if language == 'ru':
            self.morph = MorphAnalyzer()
            self.converter = converters.converter('opencorpora-int', 'ud14')
        else:
            self.morph = None
            self.converter = None
        
        # Build model
        gram_input_size = self.gram_vectorizer_input.grammemes_count()
        num_classes = self.gram_vectorizer_output.size() + 1  # +1 for padding
        
        self.model = RNNMorphNN(
            config=self.config,
            gram_input_size=gram_input_size,
            num_output_classes=num_classes,
            char_vocab_size=len(self.char_set) + 1
        )
        
        # Load weights from Keras
        load_keras_weights(self.model, f'{model_dir}/eval_model.h5')

        self.model.to(self.device)
        self.model.eval()

    def save(self, save_path: str) -> None:
        """
        Save model to PyTorch .pt format.
        
        Args:
            save_path: Path to save .pt file
        """
        # Load gram data from files
        with open(f'{self.model_dir}/gram_input.json', 'r') as f:
            gram_input_data = json.load(f)
        with open(f'{self.model_dir}/gram_output.json', 'r') as f:
            gram_output_data = json.load(f)
        
        # Save using standalone function
        save_torch_model(
            model=self.model,
            config=self.config,
            gram_input_data=gram_input_data,
            gram_output_data=gram_output_data,
            word_vocab=self.word_vocab,
            char_set=self.char_set,
            save_path=save_path
        )
        print(f'Model saved to {save_path}')

    @classmethod
    def from_torch_checkpoint(cls, checkpoint_path: str, 
                               device: Optional[str] = None) -> 'RNNMorphInference':
        """
        Load model from PyTorch .pt checkpoint.
        
        Args:
            checkpoint_path: Path to .pt file
            device: Device for inference ('cpu', 'cuda', or None for auto)
            
        Returns:
            RNNMorphInference instance
        """
        # Create a minimal instance that loads directly from checkpoint
        instance = cls.__new__(cls)
        
        instance.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=instance.device, weights_only=False)
        
        instance.config = checkpoint['config']
        instance.char_set = checkpoint['char_set']
        instance.word_vocab = checkpoint['word_vocabulary']
        instance.gram_output_vectors = checkpoint['gram_output']['vectors']
        
        # Initialize morphological analyzer
        instance.language = 'ru'
        instance.morph = MorphAnalyzer()
        instance.converter = converters.converter('opencorpora-int', 'ud14')
        
        # Build and load model
        num_classes = len(checkpoint['gram_output']['name_to_index']) + 1
        
        instance.model = RNNMorphNN(
            config=instance.config,
            gram_input_size=56,  # Fixed size for Russian
            num_output_classes=num_classes,
            char_vocab_size=len(instance.char_set) + 1
        )
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.model.to(instance.device)
        instance.model.eval()
        
        # Create proper vectorizers for feature preparation
        # Filter out jsonpickle artifacts from all_grammemes
        gram_input_data = checkpoint['gram_input']
        gram_output_data = checkpoint['gram_output']
        
        # Clean all_grammemes by removing jsonpickle artifacts
        def clean_grammemes(raw_grammemes):
            cleaned = {}
            for key, value in raw_grammemes.items():
                if key in ('default_factory', 'py/object'):
                    continue
                if isinstance(value, dict) and 'py/set' in value:
                    cleaned[key] = set(value['py/set'])
                elif isinstance(value, set):
                    cleaned[key] = value
            return cleaned
        
        instance.gram_vectorizer_input = GrammemeVectorizer()
        instance.gram_vectorizer_input.name_to_index = gram_input_data['name_to_index']
        instance.gram_vectorizer_input.all_grammemes = clean_grammemes(gram_input_data['all_grammemes'])
        instance.gram_vectorizer_input.vectors = gram_input_data['vectors']
        
        instance.gram_vectorizer_output = GrammemeVectorizer()
        instance.gram_vectorizer_output.name_to_index = gram_output_data['name_to_index']
        instance.gram_vectorizer_output.all_grammemes = clean_grammemes(gram_output_data['all_grammemes'])
        instance.gram_vectorizer_output.vectors = gram_output_data['vectors']
        
        return instance

    def _prepare_input(self, sentence: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare model inputs for a sentence using original BatchGenerator."""
        max_word_len = self.config.get('char_max_word_length', 32)
        word_count = self.config.get('word_max_count', 100000)

        # Use original BatchGenerator.get_sample for feature preparation
        word_indices, gram_vectors, char_vectors = BatchGenerator.get_sample(
            sentence=sentence,
            language=self.language,
            converter=self.converter,
            morph=self.morph,
            grammeme_vectorizer=self.gram_vectorizer_input,
            max_word_len=max_word_len,
            word_vocabulary=self.word_vocab,
            word_count=word_count,
            char_set=self.char_set
        )

        # Convert to tensors (add batch dimension)
        grammemes = torch.from_numpy(np.array(gram_vectors, dtype=np.float32)).unsqueeze(0)
        chars = torch.from_numpy(np.array(char_vectors, dtype=np.int64)).unsqueeze(0)

        return grammemes.to(self.device), chars.to(self.device)

    @torch.no_grad()
    def predict(self, sentence: List[str]) -> List[Dict[str, Any]]:
        """
        Predict POS tags for a sentence.

        Args:
            sentence: List of word tokens

        Returns:
            List of dicts with keys: word, pos, tag, score, vector
        """
        if not sentence:
            return []

        grammemes, chars = self._prepare_input(sentence)
        logits = self.model(grammemes, chars)
        # Skip first class (padding) like the original code does
        probs = F.softmax(logits[0, :, 1:], dim=-1)

        results = []
        for i, word in enumerate(sentence):
            word_probs = probs[i].cpu().numpy()
            tag_idx = int(np.argmax(word_probs))
            score = float(word_probs[tag_idx])

            # Get tag name from vectorizer (same as original code)
            full_tag = self.gram_vectorizer_output.get_name_by_index(tag_idx)
            pos, tag = full_tag.split('#')

            results.append({
                'word': word,
                'pos': pos,
                'tag': tag,
                'score': score,
                'vector': self.gram_output_vectors[tag_idx] if tag_idx < len(self.gram_output_vectors) else []
            })

        return results
    
    @torch.no_grad()
    def predict_batch(self, sentences: List[List[str]], 
                      batch_size: int = 64) -> List[List[Dict[str, Any]]]:
        """Predict POS tags for multiple sentences."""
        all_results = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            for sent in batch:
                all_results.append(self.predict(sent))
        return all_results
    
    def get_tag_probabilities(self, sentence: List[str]) -> np.ndarray:
        """
        Get full probability distribution over all tags for each word.

        Args:
            sentence: List of word tokens

        Returns:
            Array of shape (seq_len, num_tags) with probabilities
        """
        if not sentence:
            return np.array([])

        grammemes, chars = self._prepare_input(sentence)
        logits = self.model(grammemes, chars)
        # Skip first class (padding) like the original code does
        probs = F.softmax(logits[0, :, 1:], dim=-1)

        return probs.detach().cpu().numpy()
