#!/usr/bin/env python3
"""
Multimodal emotion recognition models supporting audio, text, and fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from text_encoder import FrozenBERTEncoder
from fusion import get_fusion_module


class MultimodalEmotionClassifier(nn.Module):
    """
    Multimodal emotion classifier supporting:
    - Audio-only mode
    - Text-only mode
    - Multimodal mode (audio + text with fusion)
    """

    def __init__(
        self,
        audio_dim=768,
        text_model_name="bert-base-uncased",
        hidden_dim=1024,
        num_classes=4,
        dropout=0.1,
        modality="both",  # "audio", "text", or "both"
        fusion_type="cross_attention",  # "cross_attention", "concat", "gated", "adaptive"
        fusion_hidden_dim=512,
        num_attention_heads=8,
        freeze_text_encoder=True
    ):
        super().__init__()

        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.modality = modality
        self.fusion_type = fusion_type
        self.freeze_text_encoder = freeze_text_encoder

        # Initialize text encoder if needed
        if modality in ["text", "both"]:
            self.text_encoder = FrozenBERTEncoder(model_name=text_model_name)
            self.text_dim = self.text_encoder.get_output_dim()

            if not freeze_text_encoder:
                print("⚠️  Warning: Text encoder will be fine-tuned (not frozen)")
                for param in self.text_encoder.parameters():
                    param.requires_grad = True
        else:
            self.text_encoder = None
            self.text_dim = None

        # Build the model based on modality
        if modality == "audio":
            self._build_audio_only_model()
        elif modality == "text":
            self._build_text_only_model()
        elif modality == "both":
            self._build_multimodal_model(fusion_type, fusion_hidden_dim, num_attention_heads, dropout)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        print(f"✅ MultimodalEmotionClassifier initialized:")
        print(f"   Modality: {modality}")
        print(f"   Audio dim: {audio_dim}")
        if modality in ["text", "both"]:
            print(f"   Text dim: {self.text_dim}")
        if modality == "both":
            print(f"   Fusion type: {fusion_type}")
            print(f"   Fusion hidden dim: {fusion_hidden_dim}")

    def _build_audio_only_model(self):
        """Build audio-only classification model"""
        self.classifier = nn.Sequential(
            nn.Linear(self.audio_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
        self.dropout = nn.Dropout(self.dropout_rate)

    def _build_text_only_model(self):
        """Build text-only classification model"""
        self.classifier = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
        self.dropout = nn.Dropout(self.dropout_rate)

    def _build_multimodal_model(self, fusion_type, fusion_hidden_dim, num_attention_heads, dropout):
        """Build multimodal classification model with fusion"""
        # Get fusion module
        self.fusion_module = get_fusion_module(
            fusion_type=fusion_type,
            audio_dim=self.audio_dim,
            text_dim=self.text_dim,
            hidden_dim=fusion_hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        # Classifier on top of fused features
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def set_dropout_rate(self, dropout_rate):
        """Dynamically change dropout rate"""
        old_rate = self.dropout.p
        self.dropout.p = dropout_rate

        # Update dropout in classifier if it exists
        for module in self.classifier.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate

        print(f"   🔧 Dropout changed from {old_rate:.3f} to {dropout_rate:.3f}")

    def forward(self, audio_features=None, text_input_ids=None, text_attention_mask=None):
        """
        Forward pass supporting different modality configurations

        Args:
            audio_features: [batch_size, audio_dim] or [batch_size, seq_len, audio_dim]
            text_input_ids: [batch_size, seq_len] - token IDs for text
            text_attention_mask: [batch_size, seq_len] - attention mask for text

        Returns:
            logits: [batch_size, num_classes]
        """
        if self.modality == "audio":
            return self._forward_audio_only(audio_features)
        elif self.modality == "text":
            return self._forward_text_only(text_input_ids, text_attention_mask)
        elif self.modality == "both":
            return self._forward_multimodal(audio_features, text_input_ids, text_attention_mask)

    def _forward_audio_only(self, audio_features):
        """Audio-only forward pass"""
        if audio_features is None:
            raise ValueError("audio_features required for audio-only mode")

        # Handle sequence inputs (pool if needed)
        if len(audio_features.shape) == 3:
            audio_features = audio_features.mean(dim=1)  # [batch_size, audio_dim]

        logits = self.classifier(audio_features)
        return logits

    def _forward_text_only(self, text_input_ids, text_attention_mask):
        """Text-only forward pass"""
        if text_input_ids is None or text_attention_mask is None:
            raise ValueError("text_input_ids and text_attention_mask required for text-only mode")

        # Extract text features
        text_features = self.text_encoder(text_input_ids, text_attention_mask)

        # Classify
        logits = self.classifier(text_features)
        return logits

    def _forward_multimodal(self, audio_features, text_input_ids, text_attention_mask):
        """Multimodal forward pass with fusion"""
        if audio_features is None:
            raise ValueError("audio_features required for multimodal mode")
        if text_input_ids is None or text_attention_mask is None:
            raise ValueError("text inputs required for multimodal mode")

        # Handle sequence audio inputs (pool if needed)
        if len(audio_features.shape) == 3:
            audio_features = audio_features.mean(dim=1)  # [batch_size, audio_dim]

        # Extract text features
        text_features = self.text_encoder(text_input_ids, text_attention_mask)

        # Fuse modalities
        if self.fusion_type == "adaptive":
            fused_features = self.fusion_module(audio_features, text_features)
        else:
            fused_features = self.fusion_module(audio_features, text_features)

        # Classify
        logits = self.classifier(fused_features)
        return logits


class SimpleEmotionClassifier(nn.Module):
    """
    Simple feedforward classifier for emotion recognition (audio-only)
    This is kept for backward compatibility with the original system
    """

    def __init__(self, input_dim=768, hidden_dim=1024, num_classes=4, dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def set_dropout_rate(self, dropout_rate):
        """Dynamically change dropout rate"""
        old_rate = self.dropout.p
        self.dropout.p = dropout_rate
        print(f"   🔧 Dropout changed from {old_rate:.3f} to {dropout_rate:.3f}")

    def forward(self, x):
        # Handle both 2D and 3D inputs
        if len(x.shape) == 3:
            # x shape: (batch_size, sequence_length, input_dim)
            # Global average pooling
            x = x.mean(dim=1)  # (batch_size, input_dim)

        x = self.linear1(x)
        x = self.layernorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.linear2(x)
        return logits


def create_model(config):
    """
    Factory function to create the appropriate model based on config

    Args:
        config: Configuration object with model parameters

    Returns:
        model: Initialized model
    """
    modality = getattr(config, 'modality', 'audio')

    if modality == "audio":
        # Use simple audio-only model
        model = SimpleEmotionClassifier(
            input_dim=getattr(config, 'audio_dim', 768),
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
    else:
        # Use multimodal model
        model = MultimodalEmotionClassifier(
            audio_dim=getattr(config, 'audio_dim', 768),
            text_model_name=getattr(config, 'text_model_name', 'bert-base-uncased'),
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
            modality=modality,
            fusion_type=getattr(config, 'fusion_type', 'cross_attention'),
            fusion_hidden_dim=getattr(config, 'fusion_hidden_dim', 512),
            num_attention_heads=getattr(config, 'num_attention_heads', 8),
            freeze_text_encoder=getattr(config, 'freeze_text_encoder', True)
        )

    return model


if __name__ == "__main__":
    # Test models
    print("Testing MultimodalEmotionClassifier...")

    batch_size = 8
    audio_dim = 768
    seq_len = 100
    text_seq_len = 50
    num_classes = 4

    # Create dummy inputs
    audio_features = torch.randn(batch_size, audio_dim)
    audio_seq_features = torch.randn(batch_size, seq_len, audio_dim)
    text_input_ids = torch.randint(0, 30000, (batch_size, text_seq_len))
    text_attention_mask = torch.ones(batch_size, text_seq_len)

    print(f"\nInput shapes:")
    print(f"  Audio (2D): {audio_features.shape}")
    print(f"  Audio (3D): {audio_seq_features.shape}")
    print(f"  Text IDs: {text_input_ids.shape}")
    print(f"  Text mask: {text_attention_mask.shape}")

    # Test audio-only model
    print(f"\n{'='*60}")
    print("Testing audio-only model...")
    model_audio = MultimodalEmotionClassifier(
        audio_dim=audio_dim,
        modality="audio",
        num_classes=num_classes
    )
    logits_audio = model_audio(audio_features=audio_features)
    print(f"Output logits shape: {logits_audio.shape}")

    # Test with sequence input
    logits_audio_seq = model_audio(audio_features=audio_seq_features)
    print(f"Output logits shape (seq input): {logits_audio_seq.shape}")

    # Test text-only model
    print(f"\n{'='*60}")
    print("Testing text-only model...")
    model_text = MultimodalEmotionClassifier(
        audio_dim=audio_dim,
        modality="text",
        num_classes=num_classes
    )
    logits_text = model_text(
        text_input_ids=text_input_ids,
        text_attention_mask=text_attention_mask
    )
    print(f"Output logits shape: {logits_text.shape}")

    # Test multimodal model with different fusion types
    fusion_types = ["cross_attention", "concat", "gated"]
    for fusion_type in fusion_types:
        print(f"\n{'='*60}")
        print(f"Testing multimodal model with {fusion_type} fusion...")
        model_multimodal = MultimodalEmotionClassifier(
            audio_dim=audio_dim,
            modality="both",
            fusion_type=fusion_type,
            num_classes=num_classes
        )
        logits_multimodal = model_multimodal(
            audio_features=audio_features,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask
        )
        print(f"Output logits shape: {logits_multimodal.shape}")

    # Test SimpleEmotionClassifier (backward compatibility)
    print(f"\n{'='*60}")
    print("Testing SimpleEmotionClassifier (backward compatible)...")
    simple_model = SimpleEmotionClassifier(input_dim=audio_dim, num_classes=num_classes)
    logits_simple = simple_model(audio_features)
    print(f"Output logits shape: {logits_simple.shape}")

    print("\n✅ All model tests passed!")
