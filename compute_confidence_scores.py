#!/usr/bin/env python3
"""
Pre-compute confidence scores for datasets using a simple baseline model
This allows us to use model confidence as a difficulty proxy in PKDOT
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from config import PKDOTConfig
from dataset import MultimodalEmotionDataset, collate_multimodal_batch
from model import MultimodalEmotionClassifier


def train_baseline_model(dataset_name, config, device, epochs=10):
    """
    Train a simple baseline model to compute confidence scores
    
    Args:
        dataset_name: Name of dataset
        config: PKDOTConfig
        device: torch device
        epochs: Number of training epochs
        
    Returns:
        trained model
    """
    print(f"\n🔧 Training baseline model for {dataset_name}...")
    
    # Load dataset
    train_dataset = MultimodalEmotionDataset(
        dataset_name=dataset_name,
        split="train", 
        config=config,
        Train=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_multimodal_batch,
        num_workers=4
    )
    
    # Create simple model
    model = MultimodalEmotionClassifier(
        num_classes=4,
        audio_dim=768,
        text_model_name="bert-base-uncased",
        modality="both",
        fusion_type="concat",
        fusion_hidden_dim=512,
        hidden_dim=1024,
        dropout=0.1
    ).to(device)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Simple training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            # Move to device
            audio_features = batch['audio_features'].to(device)
            audio_masks = batch['audio_masks'].to(device)
            texts = batch['texts']
            labels = batch['labels'].to(device)
            
            # Tokenize texts
            tokenized = model.text_encoder.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            text_input_ids = tokenized['input_ids'].to(device)
            text_attention_mask = tokenized['attention_mask'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(audio_features, text_input_ids, text_attention_mask)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        print(f"   Epoch {epoch+1}: Loss {avg_loss:.4f}, Acc {accuracy:.2f}%")
    
    print(f"✓ Baseline model training complete!")
    return model


def compute_confidence_scores(model, dataset_name, config, device):
    """
    Compute confidence scores for all samples in dataset
    
    Args:
        model: Trained baseline model
        dataset_name: Name of dataset
        config: PKDOTConfig
        device: torch device
        
    Returns:
        np.array: Confidence scores for each sample
    """
    print(f"\n📊 Computing confidence scores for {dataset_name}...")
    
    # Load dataset
    dataset = MultimodalEmotionDataset(
        dataset_name=dataset_name,
        split="train",
        config=config, 
        Train=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,  # Important: don't shuffle to maintain order
        collate_fn=collate_multimodal_batch,
        num_workers=4
    )
    
    model.eval()
    all_confidences = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing confidences"):
            # Move to device
            audio_features = batch['audio_features'].to(device)
            audio_masks = batch['audio_masks'].to(device)
            texts = batch['texts']
            
            # Tokenize texts
            tokenized = model.text_encoder.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            text_input_ids = tokenized['input_ids'].to(device)
            text_attention_mask = tokenized['attention_mask'].to(device)
            
            # Forward pass
            logits = model(audio_features, text_input_ids, text_attention_mask)
            
            # Convert to probabilities and get max confidence
            probs = torch.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            
            all_confidences.extend(max_probs.cpu().numpy())
    
    confidences = np.array(all_confidences)
    
    print(f"   Computed {len(confidences)} confidence scores")
    print(f"   Confidence range: [{confidences.min():.3f}, {confidences.max():.3f}]")
    print(f"   Mean confidence: {confidences.mean():.3f}")
    
    return confidences


def main():
    parser = argparse.ArgumentParser(description="Pre-compute confidence scores")
    parser.add_argument("--dataset", type=str, default="IEMO", 
                       help="Dataset name")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Training epochs for baseline model")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create config
    config = PKDOTConfig()
    config.train_dataset = args.dataset
    config.seed = args.seed
    
    # Train baseline model
    model = train_baseline_model(args.dataset, config, device, args.epochs)
    
    # Compute confidence scores
    confidences = compute_confidence_scores(model, args.dataset, config, device)
    
    # Save confidence scores
    output_file = f"confidence_scores_{args.dataset}.npy"
    np.save(output_file, confidences)
    
    print(f"\n💾 Saved confidence scores to {output_file}")
    print(f"   Use 'prototypicality_method: model_confidence' in configs to use these scores")


if __name__ == "__main__":
    main()