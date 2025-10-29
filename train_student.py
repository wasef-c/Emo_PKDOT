#!/usr/bin/env python3
"""
Phase 2: Train Student Model with Multi-Teacher Knowledge Distillation
Student learns from appropriate teacher based on sample prototypicality
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import wandb
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split

from config import PKDOTConfig
from dataset import MultimodalEmotionDataset, collate_multimodal_batch
from prototypicality import route_to_teacher
from model import MultimodalEmotionClassifier
from kd_loss import create_kd_loss
from functions import evaluate_model
from early_stopping import EarlyStopping


def load_teacher_models(teacher_checkpoints, device):
    """
    Load all trained teacher models

    Args:
        teacher_checkpoints: Dict mapping teacher_name -> checkpoint_path
        device: torch device

    Returns:
        dict: Loaded teacher models (frozen)
    """
    teachers = {}

    for teacher_name, checkpoint_path in teacher_checkpoints.items():
        print(f"Loading {teacher_name} teacher from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        teacher_config = checkpoint['teacher_config']

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

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to eval mode

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        teachers[teacher_name] = model

        print(f"  ✓ Loaded (Val UAR: {checkpoint['val_uar']:.2f}%)")

    return teachers


def get_teacher_logits_batch(teachers, proto_scores, threshold_values, batch, device):
    """
    Get teacher logits for each sample in batch based on routing

    Args:
        teachers: Dict of teacher models
        proto_scores: Prototypicality scores for batch samples
        threshold_values: Thresholds for routing
        batch: Input batch
        device: torch device

    Returns:
        torch.Tensor: [batch_size, num_classes] teacher logits
    """
    batch_size = len(batch['labels'])
    teacher_logits_batch = torch.zeros(batch_size, 4, device=device)

    # Route each sample to appropriate teacher
    audio_features = batch['audio_features'].to(device)
    audio_masks = batch['audio_masks'].to(device)
    texts = batch['texts']

    with torch.no_grad():
        for i, proto_score in enumerate(proto_scores):
            # Determine teacher
            teacher_name = route_to_teacher(proto_score, threshold_values)

            # Get teacher's prediction for this sample
            teacher = teachers[teacher_name]
            sample_audio = audio_features[i:i+1]
            sample_mask = audio_masks[i:i+1]
            sample_text = [texts[i]]

            # Tokenize text for teacher
            if teacher.modality in ['text', 'both']:
                tokenized = teacher.text_encoder.tokenizer(
                    sample_text,
                    padding=True,
                    truncation=True,
                    max_length=128,  # Default max length
                    return_tensors='pt'
                )
                text_input_ids = tokenized['input_ids'].to(device)
                text_attention_mask = tokenized['attention_mask'].to(device)
            else:
                text_input_ids = None
                text_attention_mask = None

            logits = teacher(sample_audio, text_input_ids, text_attention_mask)
            teacher_logits_batch[i] = logits[0]

    return teacher_logits_batch


def train_student_with_kd(config, teacher_checkpoints, proto_scores, threshold_values, seed=42):
    """
    Train student model with knowledge distillation from teachers

    Args:
        config: PKDOTConfig instance
        teacher_checkpoints: Dict of teacher checkpoint paths
        proto_scores: Pre-computed prototypicality scores
        threshold_values: Thresholds for routing
        seed: Random seed

    Returns:
        dict: Training results
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
        name=f"{config.experiment_name}_seed{seed}",
        config=config.to_dict()
    )

    # Load teachers
    print("\n👨‍🏫 Loading teacher models...")
    teachers = load_teacher_models(teacher_checkpoints, device)

    # Load dataset
    print("\n📁 Loading dataset...")
    full_dataset = MultimodalEmotionDataset(
        dataset_name=config.train_dataset,
        split="train",
        config=config,
        Train=True
    )
    
    # Create 80/20 train/validation split
    print(f"📊 Creating {int((1-config.val_split)*100)}/{int(config.val_split*100)} train/val split...")
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=config.val_split, 
        random_state=seed,
        stratify=[full_dataset[i]['label'].item() for i in indices]  # Stratify by class
    )
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # Create mapping from subset indices to original indices for prototypicality scores
    train_proto_scores = proto_scores[train_indices]
    
    print(f"  ✓ Train samples: {len(train_dataset)}")
    print(f"  ✓ Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.student['batch_size'],
        shuffle=True,
        collate_fn=collate_multimodal_batch,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_multimodal_batch,
        num_workers=4,
        pin_memory=True
    )

    # Create student model
    print("\n🎓 Creating student model...")
    student = MultimodalEmotionClassifier(
        num_classes=4,
        audio_dim=config.student['audio_dim'],
        text_model_name=config.student['text_model_name'],
        modality=config.student['modality'],
        fusion_type=config.student['fusion_type'],
        fusion_hidden_dim=config.student['fusion_hidden_dim'],
        num_attention_heads=config.student['num_attention_heads'],
        hidden_dim=config.student['hidden_dim'],
        dropout=config.student['dropout'],
        freeze_text_encoder=config.student['freeze_text_encoder']
    ).to(device)

    print(f"Student model: {sum(p.numel() for p in student.parameters())} parameters")

    # Create KD loss
    kd_loss_fn = create_kd_loss(config)
    print(f"KD Loss: {kd_loss_fn.__class__.__name__}")
    print(f"  Temperature: {config.student['kd_temperature']}")
    print(f"  Alpha: {config.student['kd_alpha']}")
    print(f"  Routing: {config.student['routing_strategy']}")
    
    # Create criterion for validation evaluation
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(
        student.parameters(),
        lr=float(config.student['learning_rate']),
        weight_decay=float(config.student['weight_decay'])
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.student['num_epochs']
    )

    # Save directory
    save_dir = Path(f"checkpoints/{config.experiment_name}/{seed}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Early stopping
    early_stopping = None
    if config.student.get('early_stopping_patience'):
        early_stopping = EarlyStopping(
            patience=config.student['early_stopping_patience'],
            mode='max'  # We want to maximize UAR
        )
        print(f"Early stopping enabled with patience: {config.student['early_stopping_patience']}")

    # Training loop
    best_val_acc = 0.0
    best_val_uar = 0.0
    best_checkpoint = None

    print(f"\n{'='*60}")
    print(f"Training Student with KD")
    print(f"{'='*60}\n")

    for epoch in range(config.student['num_epochs']):
        student.train()
        epoch_kd_loss = 0.0
        epoch_task_loss = 0.0
        epoch_total_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.student['num_epochs']}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            audio_features = batch['audio_features'].to(device)
            audio_masks = batch['audio_masks'].to(device)
            texts = batch['texts']
            labels = batch['labels'].to(device)

            # Get prototypicality scores for this batch
            batch_size = labels.size(0)
            start_idx = batch_idx * config.student['batch_size']
            end_idx = start_idx + batch_size
            batch_proto_scores = train_proto_scores[start_idx:end_idx]

            # Get teacher logits (routed per sample)
            teacher_logits = get_teacher_logits_batch(
                teachers,
                batch_proto_scores,
                threshold_values,
                batch,
                device
            )

            # Tokenize texts for student
            if student.modality in ['text', 'both']:
                tokenized = student.text_encoder.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=config.student['text_max_length'],
                    return_tensors='pt'
                )
                text_input_ids = tokenized['input_ids'].to(device)
                text_attention_mask = tokenized['attention_mask'].to(device)
            else:
                text_input_ids = None
                text_attention_mask = None

            # Student forward pass
            optimizer.zero_grad()
            student_logits = student(audio_features, text_input_ids, text_attention_mask)

            # Compute KD loss
            loss, loss_dict = kd_loss_fn(student_logits, teacher_logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics
            epoch_kd_loss += loss_dict['loss_kd']
            epoch_task_loss += loss_dict['loss_task']
            epoch_total_loss += loss_dict['loss_total']

            _, predicted = torch.max(student_logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'kd': f'{loss_dict["loss_kd"]:.4f}',
                'task': f'{loss_dict["loss_task"]:.4f}',
                'acc': f'{100.0 * train_correct / train_total:.2f}%'
            })

        # Epoch metrics
        avg_kd_loss = epoch_kd_loss / len(train_loader)
        avg_task_loss = epoch_task_loss / len(train_loader)
        avg_total_loss = epoch_total_loss / len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total

        # Validation
        val_results = evaluate_model(student, val_loader, criterion, device)
        val_accuracy = val_results['accuracy']
        val_uar = val_results['uar']

        print(f"\nEpoch {epoch+1}/{config.student['num_epochs']}:")
        print(f"  Train Loss: {avg_total_loss:.4f} (KD: {avg_kd_loss:.4f}, Task: {avg_task_loss:.4f})")
        print(f"  Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Acc: {val_accuracy:.2f}% | Val UAR: {val_uar:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Log to wandb
        wandb.log({
            'student/train_loss_total': avg_total_loss,
            'student/train_loss_kd': avg_kd_loss,
            'student/train_loss_task': avg_task_loss,
            'student/train_accuracy': train_accuracy,
            'student/val_accuracy': val_accuracy,
            'student/val_uar': val_uar,
            'student/learning_rate': scheduler.get_last_lr()[0],
            'student/epoch': epoch + 1
        })

        # Save best model
        if val_uar > best_val_uar:
            best_val_uar = val_uar
            best_val_acc = val_accuracy
            best_checkpoint = save_dir / f"student_best.pt"

            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_uar': val_uar,
                'student_config': config.student
            }, best_checkpoint)

            print(f"  ✓ Saved best checkpoint (UAR: {val_uar:.2f}%)")

        # Check early stopping
        if early_stopping is not None:
            if early_stopping(val_uar, student):
                print(f"\n⏹️  Early stopping triggered after {epoch + 1} epochs")
                print(f"  Best UAR: {early_stopping.get_best_score():.2f}%")
                break

        scheduler.step()

    print(f"\n✓ Student training complete!")
    print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"  Best Val UAR: {best_val_uar:.2f}%")
    print(f"  Checkpoint: {best_checkpoint}")

    # Save results
    results = {
        'best_val_accuracy': best_val_acc,
        'best_val_uar': best_val_uar,
        'checkpoint_path': str(best_checkpoint)
    }

    results_file = save_dir / "student_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Don't finish wandb here - let cross-corpus evaluation use the same run
    # wandb.finish()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PKDOT student model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset", type=str, default="IEMO", help="Training dataset")
    parser.add_argument("--experiment", type=str, default="PKDOT_baseline", help="Experiment name")
    parser.add_argument("--teacher_dir", type=str, required=True, help="Directory with teacher checkpoints")

    args = parser.parse_args()

    # Create config
    config = PKDOTConfig()
    config.seed = args.seed
    config.train_dataset = args.dataset
    config.experiment_name = args.experiment

    # Load teacher checkpoints
    teacher_dir = Path(args.teacher_dir)
    teacher_checkpoints = {
        'easy': teacher_dir / 'teacher_easy_best.pt',
        'medium': teacher_dir / 'teacher_medium_best.pt',
        'full': teacher_dir / 'teacher_full_best.pt'
    }

    # Load prototypicality scores
    proto_scores = np.load(teacher_dir / 'proto_scores.npy')
    threshold_values = np.load(teacher_dir / 'threshold_values.npy')

    # Train student
    results = train_student_with_kd(
        config,
        teacher_checkpoints,
        proto_scores,
        threshold_values,
        seed=args.seed
    )
