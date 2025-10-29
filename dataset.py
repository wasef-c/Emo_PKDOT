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
import math
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from functions import calculate_difficulty


class MultimodalEmotionDataset(Dataset):
    """
    Dataset class for multimodal emotion recognition
    Supports audio-only, text-only, and multimodal (audio + text) modes
    """

    def __init__(self, dataset_name, split="train", config=None, Train=False):
        self.dataset_name = dataset_name
        self.split = split
        self.config = config
        self.modality = getattr(config, "modality", "both")

        # Load HuggingFace dataset
        if dataset_name == "IEMO":
            self.hf_dataset = load_dataset(
                "cairocode/IEMO_Emotion2Vec_Text", split=split
            )
        elif dataset_name == "MSPI":
            self.hf_dataset = load_dataset(
                "cairocode/MSPI_Emotion2Vec_Text", split=split
            )
        elif dataset_name == "MSPP":
            self.hf_dataset = load_dataset(
                "cairocode/MSPP_Emotion2Vec_Text", split=split
            )
        elif dataset_name == "CMUMOSEI":
            self.hf_dataset = load_dataset(
                "cairocode/CMU_MOSEI_EMOTION2VEC_4class_2", split=split
            )
        elif dataset_name == "SAMSEMO":
            self.hf_dataset = load_dataset(
                "cairocode/samsemo_emotion2vec_4_V2", split=split
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Process data
        self.data = []

        for item in self.hf_dataset:
            # Extract audio features if needed for audio or multimodal mode
            if self.modality in ["audio", "both"]:
                features = torch.tensor(
                    item["emotion2vec_features"][0]["feats"], dtype=torch.float32
                )

                # Calculate sequence length for curriculum learning
                if len(features.shape) == 2:
                    sequence_length = features.shape[0]  # [seq_len, feature_dim]
                else:
                    sequence_length = 1  # Already pooled to [feature_dim]
            else:
                # Text-only mode: no audio features needed
                features = None
                sequence_length = 1

            # Extract transcript for text or multimodal mode
            if self.modality in ["text", "both"]:
                # 1. Try to get the 'transcript'
                transcript = item.get("transcript")

                # 2. If 'transcript' is missing or None, try 'text' as a fallback
                if transcript is None or transcript == "":
                    transcript = item.get("text")

                # 3. Handle cases where both 'transcript' and 'text' are missing/empty
                if transcript is None or transcript == "":
                    transcript = "[EMPTY]"  # Placeholder for missing transcripts
            else:
                transcript = None

            # Get speaker and session information
            if Train == True:
                # Get speaker ID and calculate session directly
                if self.dataset_name == "IEMO":
                    speaker_id = item["speaker_id"]
                    session = (speaker_id - 1) // 2 + 1
                elif self.dataset_name == "MSPI":
                    speaker_id = item["speakerID"]
                    session = (speaker_id - 947) // 2 + 1
                elif self.dataset_name == "MSPP":
                    speaker_id = item["SpkrID"]
                    session = (speaker_id - 1) // 500 + 1
                elif self.dataset_name == "CMUMOSEI":
                    # CMU-MOSEI has video_id field
                    speaker_id = hash(item.get("video_id", "unknown")) % 10000
                    session = (speaker_id - 1) // 100 + 1
                elif self.dataset_name == "SAMSEMO":
                    # SAMSEMO may have speaker_id or file_name
                    speaker_id = item.get(
                        "speaker_id", hash(item.get("file_name", "unknown")) % 10000
                    )
                    session = (speaker_id - 1) // 100 + 1
                else:
                    # Fallback for other datasets
                    try:
                        speaker_id = item["speaker_id"]
                    except:
                        speaker_id = item.get("speakerID", item.get("SpkrID", 1))
                    session = (speaker_id - 1) // 2 + 1
            else:
                speaker_id = -1  # Use -1 instead of None for test datasets
                session = -1  # Use -1 instead of None for test datasets

            label = item["label"]

            # Get curriculum order from dataset
            curriculum_order = item.get(
                "curriculum_order", 0.5
            )  # Default to middle if missing

            # Get VAD values for difficulty calculation
            valence = item.get("valence", item.get("EmoVal", None))
            arousal = item.get("arousal", item.get("EmoAct", None))
            domination = item.get(
                "domination", item.get("consensus_dominance", item.get("EmoDom", None))
            )

            # Replace NaN or None with 3
            def fix_vad(value):
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    return 3
                return value

            valence = fix_vad(valence)
            arousal = fix_vad(arousal)
            domination = fix_vad(domination)
            item_with_vad = {
                "label": label,
                "valence": valence,
                "arousal": arousal,
                "domination": domination,
            }
            difficulty = calculate_difficulty(
                item_with_vad,
                config.expected_vad,
                config.difficulty_method,
                dataset=dataset_name,
            )

            self.data.append(
                {
                    "features": features,
                    "transcript": transcript,
                    "label": label,
                    "speaker_id": speaker_id,
                    "session": session,
                    "dataset": dataset_name,
                    "difficulty": difficulty,
                    "curriculum_order": curriculum_order,
                    "sequence_length": sequence_length,
                    "valence": valence,
                    "arousal": arousal,
                    "domination": domination,
                }
            )

        print(f"✅ Loaded {len(self.data)} samples from {dataset_name}")
        print(f"   Modality: {self.modality}")

        # Print session distribution for debugging
        session_counts = defaultdict(int)
        for item in self.data:
            session_counts[item["session"]] += 1

        print(f"📊 {dataset_name} Sessions:")
        for session_id in sorted(session_counts.keys()):
            print(f"   Session {session_id}: {session_counts[session_id]} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        result = {
            "label": torch.tensor(item["label"], dtype=torch.long),
            "speaker_id": item["speaker_id"],
            "session": item["session"],
            "dataset": item["dataset"],
            "difficulty": item["difficulty"],
            "curriculum_order": item["curriculum_order"],
            "sequence_length": item["sequence_length"],
            "valence": item["valence"],
            "arousal": item["arousal"],
            "domination": item["domination"],
        }
        result["features"] = item["features"]
        result["transcript"] = item["transcript"]

        return result

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
    audio_features = [item['features'] for item in batch]
    texts = [item['transcript'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    valences = torch.tensor([item['valence'] for item in batch], dtype=torch.float32)
    arousals = torch.tensor([item['arousal'] for item in batch], dtype=torch.float32)
    dominations = torch.tensor([item['domination'] for item in batch], dtype=torch.float32)
    curriculum_orders = torch.tensor([item['curriculum_order'] for item in batch], dtype=torch.float32)
    speaker_ids = [item['speaker_id'] for item in batch]
    # filenames = [item['filename'] for item in batch]

    # Stack audio features - they are already 768-dim vectors
    padded_audio = torch.stack(audio_features)  # [batch_size, 768]
    audio_masks = torch.ones(len(batch), 1, dtype=torch.bool)  # All features are valid

    return {
        'audio_features': padded_audio,
        'audio_masks': audio_masks,
        'texts': texts,
        'labels': labels,
        'valences': valences,
        'arousals': arousals,
        'dominations': dominations,
        'curriculum_orders': curriculum_orders,
        'speaker_ids': speaker_ids,
        # 'filenames': filenames
    }
