#!/usr/bin/env python3
"""
Phase 1: Train Multiple Specialized Teacher Models
Each teacher is trained on a specific difficulty subset of data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import wandb
from pathlib import Path
from tqdm import tqdm
import json

from config import PKDOTConfig
from dataset import MultimodalEmotionDataset, create_difficulty_subsets, collate_multimodal_batch
from prototypicality import compute_prototypicality_scores, split_by_quantiles
from model import MultimodalEmotionClassifier
from functions import compute_class_weights, evaluate_model
from early_stopping import EarlyStopping


def train_teacher(
    teacher_name,
    teacher_config,
    train_subset,
    val_loader,
    device,
    save_dir,
    use_wandb=True
):
    """
    Train a single teacher model on its assigned data subset

    Args:
        teacher_name: "easy", "medium", or "full"
        teacher_config: Dict with teacher configuration
        train_subset: Dataset subset for this teacher
        val_loader: Validation dataloader
        device: torch device
        save_dir: Directory to save checkpoints
        use_wandb: Whether to log to wandb

    Returns:
        dict: Training results and checkpoint path
    """
    print(f"\n{'='*60}")
    print(f"Training {teacher_name.upper()} Teacher")
    print(f"Difficulty range: {teacher_config['difficulty_range']}")
    print(f"Training samples: {len(train_subset)}")
    print(f"{'='*60}\n")

    # Create model
    model = MultimodalEmotionClassifier(
        num_classes=4,
        audio_dim=teacher_config['audio_dim'],
        text_model_name=teacher_config['text_model_name'],
        modality=teacher_config['modality'],
        fusion_type=teacher_config['fusion_type'],
        fusion_hidden_dim=teacher_config['fusion_hidden_dim'],
        num_attention_heads=teacher_config['num_attention_heads'],
        hidden_dim=teacher_config['hidden_dim'],
        dropout=teacher_config['dropout'],
        freeze_text_encoder=teacher_config['freeze_text_encoder']
    ).to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create dataloader
    train_loader = DataLoader(
        train_subset,
        batch_size=teacher_config['batch_size'],
        shuffle=True,
        collate_fn=collate_multimodal_batch,
        num_workers=4,
        pin_memory=True
    )

    # Compute class weights from subset
    labels = [train_subset.dataset.data[idx]['label'] for idx in train_subset.indices]
    class_weights_dict = compute_class_weights(labels)
    class_weights_tensor = torch.tensor(
        [class_weights_dict[i] for i in range(4)],
        dtype=torch.float32
    ).to(device)

    print(f"Class weights: {class_weights_tensor.cpu().numpy()}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(teacher_config['learning_rate']),
        weight_decay=float(teacher_config['weight_decay'])
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=teacher_config['num_epochs']
    )

    # Early stopping
    early_stopping = None
    if teacher_config.get('early_stopping_patience'):
        early_stopping = EarlyStopping(
            patience=teacher_config['early_stopping_patience'],
            mode='max'  # We want to maximize UAR
        )
        print(f"Early stopping enabled with patience: {teacher_config['early_stopping_patience']}")

    # Training loop
    best_val_acc = 0.0
    best_val_uar = 0.0
    best_checkpoint = None

    for epoch in range(teacher_config['num_epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{teacher_config['num_epochs']}")

        for batch in pbar:
            # Move to device
            audio_features = batch['audio_features'].to(device)
            audio_masks = batch['audio_masks'].to(device)
            texts = batch['texts']
            labels = batch['labels'].to(device)

            # Tokenize texts
            if model.modality in ['text', 'both']:
                tokenized = model.text_encoder.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=teacher_config['text_max_length'],
                    return_tensors='pt'
                )
                text_input_ids = tokenized['input_ids'].to(device)
                text_attention_mask = tokenized['attention_mask'].to(device)
            else:
                text_input_ids = None
                text_attention_mask = None

            # Forward pass
            optimizer.zero_grad()
            logits = model(audio_features, text_input_ids, text_attention_mask)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * train_correct / train_total:.2f}%'
            })

        # Epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total

        # Validation
        val_results = evaluate_model(model, val_loader, criterion, device)
        val_accuracy = val_results['accuracy']
        val_uar = val_results['uar']

        print(f"\nEpoch {epoch+1}/{teacher_config['num_epochs']}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Acc: {val_accuracy:.2f}% | Val UAR: {val_uar:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Log to wandb
        if use_wandb:
            wandb.log({
                f"{teacher_name}/train_loss": avg_train_loss,
                f"{teacher_name}/train_accuracy": train_accuracy,
                f"{teacher_name}/val_accuracy": val_accuracy,
                f"{teacher_name}/val_uar": val_uar,
                f"{teacher_name}/learning_rate": scheduler.get_last_lr()[0],
                f"{teacher_name}/epoch": epoch + 1
            })

        # Save best model
        if val_uar > best_val_uar:
            best_val_uar = val_uar
            best_val_acc = val_accuracy
            best_checkpoint = save_dir / f"teacher_{teacher_name}_best.pt"

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_uar': val_uar,
                'teacher_config': teacher_config
            }, best_checkpoint)

            print(f"  ✓ Saved best checkpoint (UAR: {val_uar:.2f}%)")

        # Check early stopping
        if early_stopping is not None:
            if early_stopping(val_uar, model):
                print(f"\n⏹️  Early stopping triggered after {epoch + 1} epochs")
                print(f"  Best UAR: {early_stopping.get_best_score():.2f}%")
                break

        scheduler.step()

    print(f"\n✓ {teacher_name.upper()} Teacher training complete!")
    print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"  Best Val UAR: {best_val_uar:.2f}%")
    print(f"  Checkpoint: {best_checkpoint}")

    return {
        'teacher_name': teacher_name,
        'best_val_accuracy': best_val_acc,
        'best_val_uar': best_val_uar,
        'checkpoint_path': str(best_checkpoint)
    }


def train_all_teachers(config, seed=42):
    """
    Train all teacher models (easy, medium, full)

    Args:
        config: PKDOTConfig instance
        seed: Random seed for reproducibility

    Returns:
        dict: Results for all teachers
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=f"{config.experiment_name}_teachers_seed{seed}",
        config=config.to_dict()
    )

    # Load dataset
    print("\n📁 Loading dataset...")
    train_dataset = MultimodalEmotionDataset(
        dataset_name=config.train_dataset,
        split="train",
        config=config,
        Train=True
    )
    val_dataset = MultimodalEmotionDataset(
        dataset_name=config.train_dataset,
        split="train",  # Use test as validation for now
        config=config,
        Train=False
    )

    # Compute prototypicality scores
    print("\n📊 Computing prototypicality scores...")
    proto_scores = compute_prototypicality_scores(train_dataset, config)

    # Split data by quantiles
    print("\n✂️ Splitting data by difficulty...")
    split_indices, threshold_values = split_by_quantiles(
        train_dataset,
        proto_scores,
        config.quantile_split
    )

    # Create subsets
    print("\n📦 Creating difficulty subsets...")
    train_subsets = create_difficulty_subsets(
        train_dataset,
        proto_scores,
        split_indices
    )

    # Create validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_multimodal_batch,
        num_workers=4,
        pin_memory=True
    )

    # Create save directory
    save_dir = Path(f"checkpoints/{config.experiment_name}/{seed}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save prototypicality scores and thresholds
    np.save(save_dir / "proto_scores.npy", proto_scores)
    np.save(save_dir / "threshold_values.npy", threshold_values)

    print(f"\n💾 Checkpoints will be saved to: {save_dir}")

    # Train each teacher
    results = {}

    for teacher_name in config.teachers.keys():
        teacher_config = config.teachers[teacher_name]
        train_subset = train_subsets[teacher_name]

        result = train_teacher(
            teacher_name=teacher_name,
            teacher_config=teacher_config,
            train_subset=train_subset,
            val_loader=val_loader,
            device=device,
            save_dir=save_dir,
            use_wandb=True
        )

        results[teacher_name] = result

    # Save results summary
    results_file = save_dir / "teacher_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("ALL TEACHERS TRAINED SUCCESSFULLY")
    print(f"{'='*60}")
    print("\nSummary:")
    for teacher_name, result in results.items():
        print(f"\n{teacher_name.upper()} Teacher:")
        print(f"  Val Accuracy: {result['best_val_accuracy']:.2f}%")
        print(f"  Val UAR: {result['best_val_uar']:.2f}%")
        print(f"  Checkpoint: {result['checkpoint_path']}")

    print(f"\nResults saved to: {results_file}")

    wandb.finish()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PKDOT teacher models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset", type=str, default="IEMO", help="Training dataset")
    parser.add_argument("--experiment", type=str, default="PKDOT_baseline", help="Experiment name")

    args = parser.parse_args()

    # Create config
    config = PKDOTConfig()
    config.seed = args.seed
    config.train_dataset = args.dataset
    config.experiment_name = args.experiment

    # Train teachers
    results = train_all_teachers(config, seed=args.seed)
