#!/usr/bin/env python3
"""
Prototypicality calculation and data splitting utilities
Supports multiple methods for computing how "typical" a sample is for its class
"""

import numpy as np
import torch
import math
from collections import defaultdict


def calculate_vad_difficulty(sample, expected_vad, method="euclidean_distance"):
    """
    Calculate difficulty based on VAD distance from expected class prototype
    (Reuses logic from Emotion2Vec_Text)

    Args:
        sample: Dict with 'label', 'valence', 'arousal', 'domination'
        expected_vad: Dict mapping label -> [expected_v, expected_a, expected_d]
        method: Distance metric to use

    Returns:
        float: Difficulty score (0 = prototypical, higher = more difficult)
    """
    label = sample["label"]
    expected_v, expected_a, expected_d = expected_vad[label]

    actual_v = sample.get("valence", 3.0)
    actual_a = sample.get("arousal", 3.0)
    actual_d = sample.get("domination", 3.0)

    if method == "euclidean_distance":
        distance = math.sqrt(
            (actual_v - expected_v) ** 2 +
            (actual_a - expected_a) ** 2 +
            (actual_d - expected_d) ** 2
        )
        return distance

    elif method == "manhattan_distance":
        distance = (
            abs(actual_v - expected_v) +
            abs(actual_a - expected_a) +
            abs(actual_d - expected_d)
        )
        return distance

    else:
        raise ValueError(f"Unknown difficulty method: {method}")


def compute_prototypicality_scores(dataset, config):
    """
    Compute prototypicality scores for all samples in dataset

    Args:
        dataset: Dataset object with .data attribute
        config: PKDOTConfig with prototypicality settings

    Returns:
        np.array: Prototypicality scores (lower = more prototypical)
    """
    method = config.prototypicality_method

    if method == "vad_distance":
        # Use VAD-based difficulty
        scores = []
        for item in dataset.data:
            score = calculate_vad_difficulty(
                item,
                config.expected_vad,
                config.difficulty_method
            )
            scores.append(score)
        scores = np.array(scores)

    elif method == "feature_centroid":
        # Compute distance to class centroids in feature space
        scores = compute_feature_centroid_distance(dataset, config)

    elif method == "learned":
        # Use learned prototypicality estimator (future work)
        raise NotImplementedError("Learned prototypicality not yet implemented")

    else:
        raise ValueError(f"Unknown prototypicality method: {method}")

    # Normalize scores to [0, 1] if requested
    if config.normalize_prototypicality:
        scores = normalize_scores(scores)

    return scores


def compute_feature_centroid_distance(dataset, config):
    """
    Compute prototypicality as distance to class centroids in feature space

    Args:
        dataset: Dataset with audio and/or text features
        config: PKDOTConfig

    Returns:
        np.array: Distance to nearest class centroid for each sample
    """
    # Extract features for all samples
    features_list = []
    labels_list = []

    for item in dataset.data:
        # Combine audio and text features (if available)
        features = []

        if config.teachers["easy"]["modality"] in ["audio", "both"]:
            audio_feat = item["features"]
            if len(audio_feat.shape) == 2:  # [seq_len, dim]
                audio_feat = audio_feat.mean(dim=0)  # Pool to [dim]
            features.append(audio_feat.numpy())

        # Note: Text features would need pre-extraction with BERT
        # For now, just use audio features

        if features:
            combined = np.concatenate(features)
            features_list.append(combined)
            labels_list.append(item["label"])

    features = np.array(features_list)
    labels = np.array(labels_list)

    # Compute class centroids
    num_classes = config.num_classes
    centroids = np.zeros((num_classes, features.shape[1]))

    for class_idx in range(num_classes):
        class_mask = (labels == class_idx)
        if class_mask.sum() > 0:
            centroids[class_idx] = features[class_mask].mean(axis=0)

    # Compute distance to own class centroid
    distances = []
    for i, (feat, label) in enumerate(zip(features, labels)):
        centroid = centroids[label]
        dist = np.linalg.norm(feat - centroid)
        distances.append(dist)

    return np.array(distances)


def normalize_scores(scores):
    """Normalize scores to [0, 1] range"""
    min_score = scores.min()
    max_score = scores.max()

    if max_score - min_score < 1e-8:  # Avoid division by zero
        return np.zeros_like(scores)

    normalized = (scores - min_score) / (max_score - min_score)
    return normalized


def split_by_quantiles(dataset, proto_scores, quantile_thresholds):
    """
    Split dataset into subsets based on quantile thresholds

    Args:
        dataset: Full dataset
        proto_scores: Prototypicality scores (np.array)
        quantile_thresholds: List of quantiles, e.g., [0.33, 0.67]

    Returns:
        dict: {
            "easy": indices for samples in [0, threshold[0]),
            "medium": indices for samples in [threshold[0], threshold[1]),
            "hard": indices for samples in [threshold[1], 1.0]
        }
    """
    # Compute quantile values
    threshold_values = np.quantile(proto_scores, quantile_thresholds)

    # Split into easy/medium/hard
    easy_indices = np.where(proto_scores < threshold_values[0])[0]
    medium_indices = np.where(
        (proto_scores >= threshold_values[0]) &
        (proto_scores < threshold_values[1])
    )[0]
    hard_indices = np.where(proto_scores >= threshold_values[1])[0]

    splits = {
        "easy": easy_indices.tolist(),
        "medium": medium_indices.tolist(),
        "hard": hard_indices.tolist(),
        "full": list(range(len(proto_scores)))  # All samples
    }

    print(f"\n📊 Data Split by Prototypicality:")
    print(f"   Easy (bottom {quantile_thresholds[0]*100:.0f}%): {len(splits['easy'])} samples")
    print(f"   Medium ({quantile_thresholds[0]*100:.0f}%-{quantile_thresholds[1]*100:.0f}%): {len(splits['medium'])} samples")
    print(f"   Hard (top {(1-quantile_thresholds[1])*100:.0f}%): {len(splits['hard'])} samples")
    print(f"   Full (all): {len(splits['full'])} samples")

    # Print class distribution for each split
    for split_name, indices in splits.items():
        if split_name == "full":
            continue
        class_counts = defaultdict(int)
        for idx in indices:
            label = dataset.data[idx]["label"]
            class_counts[label] += 1
        print(f"   {split_name.capitalize()} class distribution: {dict(class_counts)}")

    return splits, threshold_values


def route_to_teacher(proto_score, threshold_values):
    """
    Determine which teacher should handle a sample based on its prototypicality

    Args:
        proto_score: float, prototypicality score for sample
        threshold_values: np.array, quantile threshold values

    Returns:
        str: "easy", "medium", or "full" (for hard samples)
    """
    if proto_score < threshold_values[0]:
        return "easy"
    elif proto_score < threshold_values[1]:
        return "medium"
    else:
        return "full"  # Hard samples go to full teacher


def compute_prototypicality_batch(batch, proto_scores_full, indices):
    """
    Get prototypicality scores for a batch of samples

    Args:
        batch: Current batch
        proto_scores_full: Pre-computed scores for full dataset
        indices: Indices of samples in this batch

    Returns:
        torch.Tensor: Prototypicality scores for batch
    """
    batch_scores = proto_scores_full[indices]
    return torch.tensor(batch_scores, dtype=torch.float32)
