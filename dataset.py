#!/usr/bin/env python3
"""
Dataset utilities for PKDOT
Handles loading and processing of multimodal emotion datasets
"""

import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
import pickle
import numpy as np
from pathlib import Path


class MultimodalEmotionDataset(Dataset):
    """
    Multimodal Emotion Dataset supporting audio (emotion2vec) and text features
    """

    def __init__(self, dataset_name, split="train", cache_dir="/home/rml/Documents/audio/emotion2vec_base_finetuned"):
        """
        Args:
            dataset_name: Name of dataset (IEMO, MSPI, MSPP, CMUMOSEI, SAMSEMO)
            split: "train" or "test"
            cache_dir: Directory containing cached features
        """
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = Path(cache_dir)

        # Load the appropriate pickle file
        if dataset_name == "IEMO":
            filename = "iemocap_features.pkl"
        elif dataset_name == "MSPI":
            filename = "msp-improv_features.pkl"
        elif dataset_name == "MSPP":
            filename = "msp-podcast_features.pkl"
        elif dataset_name == "CMUMOSEI":
            filename = "cmu-mosei_features.pkl"
        elif dataset_name == "SAMSEMO":
            filename = "samse-mo_features.pkl"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        filepath = self.cache_dir / filename
        print(f"Loading {dataset_name} from {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Extract samples for this split
        self.data = [item for item in data if item['split'] == split]

        # Map emotion labels to 0-3 (neutral, happy, sad, anger)
        self.label_mapping = {
            'neu': 0, 'neutral': 0,
            'hap': 1, 'happy': 1, 'exc': 1,
            'sad': 2,
            'ang': 3, 'anger': 3
        }

        print(f"Loaded {len(self.data)} {split} samples from {dataset_name}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Audio features (emotion2vec)
        audio_features = item['features']  # [seq_len, 768]
        if isinstance(audio_features, np.ndarray):
            audio_features = torch.from_numpy(audio_features).float()

        # Text (for BERT encoding in model)
        text = item.get('text', '')

        # Label
        label_str = item['label']
        label = self.label_mapping.get(label_str.lower(), 0)

        # VAD values
        valence = item.get('valence', 3.0)
        arousal = item.get('arousal', 3.0)
        domination = item.get('domination', 3.0)

        # Speaker ID
        speaker_id = item.get('speaker_id', 'unknown')

        return {
            'audio_features': audio_features,
            'text': text,
            'label': label,
            'valence': valence,
            'arousal': arousal,
            'domination': domination,
            'speaker_id': speaker_id,
            'filename': item.get('filename', '')
        }


def create_difficulty_subsets(dataset, proto_scores, split_indices):
    """
    Create dataset subsets based on prototypicality splits

    Args:
        dataset: Full MultimodalEmotionDataset
        proto_scores: Array of prototypicality scores
        split_indices: Dict with 'easy', 'medium', 'hard', 'full' keys

    Returns:
        dict: Subset datasets for each difficulty level
    """
    subsets = {}

    for difficulty, indices in split_indices.items():
        subsets[difficulty] = Subset(dataset, indices)
        print(f"  {difficulty.capitalize()} subset: {len(indices)} samples")

    return subsets


def collate_multimodal_batch(batch):
    """
    Custom collate function for batching multimodal data
    Handles variable-length audio sequences
    """
    # Extract components
    audio_features = [item['audio_features'] for item in batch]
    texts = [item['text'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    valences = torch.tensor([item['valence'] for item in batch], dtype=torch.float32)
    arousals = torch.tensor([item['arousal'] for item in batch], dtype=torch.float32)
    dominations = torch.tensor([item['domination'] for item in batch], dtype=torch.float32)
    speaker_ids = [item['speaker_id'] for item in batch]
    filenames = [item['filename'] for item in batch]

    # Pad audio features to max length in batch
    max_len = max(feat.shape[0] for feat in audio_features)
    audio_dim = audio_features[0].shape[1]

    padded_audio = torch.zeros(len(batch), max_len, audio_dim)
    audio_masks = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, feat in enumerate(audio_features):
        length = feat.shape[0]
        padded_audio[i, :length, :] = feat
        audio_masks[i, :length] = True

    return {
        'audio_features': padded_audio,
        'audio_masks': audio_masks,
        'texts': texts,
        'labels': labels,
        'valences': valences,
        'arousals': arousals,
        'dominations': dominations,
        'speaker_ids': speaker_ids,
        'filenames': filenames
    }
