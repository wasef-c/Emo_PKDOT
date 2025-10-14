#!/usr/bin/env python3
"""
Main orchestrator for PKDOT (Prototypicality-Guided Multi-Teacher Knowledge Distillation)

Runs complete pipeline:
1. Phase 1: Train specialized teacher models
2. Phase 2: Train student with KD from teachers
3. Evaluation: Cross-corpus testing
"""

import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
import json
import wandb

from config import PKDOTConfig
from train_teachers import train_all_teachers
from train_student import train_student_with_kd
from dataset import MultimodalEmotionDataset, collate_multimodal_batch
from torch.utils.data import DataLoader
from functions import evaluate_model, compute_metrics


def evaluate_cross_corpus(student_checkpoint, config, seed):
    """
    Evaluate trained student on cross-corpus datasets

    Args:
        student_checkpoint: Path to student checkpoint
        config: PKDOTConfig instance
        seed: Random seed

    Returns:
        dict: Cross-corpus evaluation results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load student model
    from model import MultimodalEmotionClassifier

    checkpoint = torch.load(student_checkpoint, map_location=device)
    student_config = checkpoint['student_config']

    model = MultimodalEmotionClassifier(
        num_classes=4,
        audio_dim=student_config['audio_dim'],
        text_model_name=student_config['text_model_name'],
        modality=student_config['modality'],
        fusion_type=student_config['fusion_type'],
        fusion_hidden_dim=student_config['fusion_hidden_dim'],
        num_attention_heads=student_config['num_attention_heads'],
        hidden_dim=student_config['hidden_dim'],
        dropout=student_config['dropout'],
        freeze_text_encoder=student_config['freeze_text_encoder']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\n📊 Cross-Corpus Evaluation")
    print(f"Trained on: {config.train_dataset}")

    # Test datasets (exclude training dataset)
    test_datasets = ['IEMO', 'MSPI', 'MSPP', 'CMUMOSEI', 'SAMSEMO']
    if config.train_dataset in test_datasets:
        test_datasets.remove(config.train_dataset)

    results = {}

    for test_dataset_name in test_datasets:
        print(f"\nEvaluating on {test_dataset_name}...")

        try:
            test_dataset = MultimodalEmotionDataset(
                dataset_name=test_dataset_name,
                split="test"
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=64,
                shuffle=False,
                collate_fn=collate_multimodal_batch,
                num_workers=4
            )

            test_results = evaluate_model(model, test_loader, device)

            print(f"  Accuracy: {test_results['accuracy']:.2f}%")
            print(f"  UAR: {test_results['uar']:.2f}%")
            print(f"  F1: {test_results['f1']:.2f}%")

            results[test_dataset_name] = test_results

        except Exception as e:
            print(f"  ⚠️  Error evaluating {test_dataset_name}: {e}")
            results[test_dataset_name] = None

    return results


def run_full_pipeline(config, seed=42, skip_teachers=False, teacher_dir=None):
    """
    Run complete PKDOT pipeline

    Args:
        config: PKDOTConfig instance
        seed: Random seed
        skip_teachers: If True, skip teacher training (use existing)
        teacher_dir: Directory with existing teacher checkpoints

    Returns:
        dict: Complete results
    """
    print("\n" + "="*60)
    print("PKDOT: Prototypicality-Guided Multi-Teacher KD")
    print("="*60)
    print(f"\nExperiment: {config.experiment_name}")
    print(f"Dataset: {config.train_dataset}")
    print(f"Seed: {seed}")
    print(f"Prototypicality: {config.prototypicality_method}")
    print(f"Quantile Split: {config.quantile_split}")

    results = {
        'experiment_name': config.experiment_name,
        'seed': seed,
        'train_dataset': config.train_dataset,
        'config': config.to_dict()
    }

    # Phase 1: Train Teachers
    if not skip_teachers:
        print("\n" + "="*60)
        print("PHASE 1: Train Teacher Models")
        print("="*60)

        teacher_results = train_all_teachers(config, seed)
        results['teachers'] = teacher_results

        # Get checkpoint directory
        checkpoint_dir = Path(f"checkpoints/{config.experiment_name}_seed{seed}")

    else:
        print("\n⏭️  Skipping teacher training, using existing checkpoints")
        checkpoint_dir = Path(teacher_dir)
        results['teachers'] = 'skipped'

    # Phase 2: Train Student
    print("\n" + "="*60)
    print("PHASE 2: Train Student with KD")
    print("="*60)

    # Load teacher checkpoints
    teacher_checkpoints = {
        'easy': checkpoint_dir / 'teacher_easy_best.pt',
        'medium': checkpoint_dir / 'teacher_medium_best.pt',
        'full': checkpoint_dir / 'teacher_full_best.pt'
    }

    # Verify checkpoints exist
    for name, path in teacher_checkpoints.items():
        if not path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {path}")

    # Load prototypicality data
    proto_scores = np.load(checkpoint_dir / 'proto_scores.npy')
    threshold_values = np.load(checkpoint_dir / 'threshold_values.npy')

    student_results = train_student_with_kd(
        config,
        teacher_checkpoints,
        proto_scores,
        threshold_values,
        seed=seed
    )
    results['student'] = student_results

    # Phase 3: Cross-Corpus Evaluation
    if config.evaluation_mode in ['cross_corpus', 'both']:
        print("\n" + "="*60)
        print("PHASE 3: Cross-Corpus Evaluation")
        print("="*60)

        student_checkpoint = student_results['checkpoint_path']
        cross_corpus_results = evaluate_cross_corpus(
            student_checkpoint,
            config,
            seed
        )
        results['cross_corpus'] = cross_corpus_results

    # Save complete results
    results_file = checkpoint_dir / 'complete_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {results_file}")

    # Print summary
    print("\n📊 Final Summary:")
    if isinstance(results['teachers'], dict):
        print("\nTeachers:")
        for teacher_name, teacher_result in results['teachers'].items():
            print(f"  {teacher_name}: UAR {teacher_result['best_val_uar']:.2f}%")

    print(f"\nStudent:")
    print(f"  Val Accuracy: {student_results['best_val_accuracy']:.2f}%")
    print(f"  Val UAR: {student_results['best_val_uar']:.2f}%")

    if 'cross_corpus' in results:
        print(f"\nCross-Corpus Results:")
        for dataset_name, result in results['cross_corpus'].items():
            if result:
                print(f"  {config.train_dataset}→{dataset_name}: "
                      f"Acc {result['accuracy']:.2f}%, "
                      f"UAR {result['uar']:.2f}%, "
                      f"F1 {result['f1']:.2f}%")

    return results


def run_experiments_from_yaml(yaml_path):
    """
    Run multiple PKDOT experiments from YAML config

    Args:
        yaml_path: Path to YAML config file
    """
    with open(yaml_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    experiments = yaml_config.get('experiments', [])

    print(f"\n📋 Found {len(experiments)} experiments in {yaml_path}")

    all_results = []

    for exp_config in experiments:
        print(f"\n{'='*80}")
        print(f"Starting Experiment: {exp_config['name']}")
        print(f"{'='*80}")

        # Create PKDOTConfig from experiment config
        config = PKDOTConfig()

        # Update config with experiment parameters
        for key, value in exp_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Update teacher configs if specified
        if 'teacher_overrides' in exp_config:
            for teacher_name, overrides in exp_config['teacher_overrides'].items():
                if teacher_name in config.teachers:
                    config.teachers[teacher_name].update(overrides)

        # Update student config if specified
        if 'student_overrides' in exp_config:
            config.student.update(exp_config['student_overrides'])

        # Run with multiple seeds if specified
        seeds = exp_config.get('seeds', config.seeds)

        for seed in seeds:
            print(f"\n▶️  Running with seed {seed}")
            config.seed = seed

            try:
                results = run_full_pipeline(config, seed=seed)
                all_results.append(results)

            except Exception as e:
                print(f"\n❌ Experiment failed: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"✓ All experiments complete: {len(all_results)} successful")
    print(f"{'='*80}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PKDOT experiments")

    # Mode selection
    parser.add_argument("--mode", type=str, default="full",
                       choices=["full", "teachers_only", "student_only", "eval_only"],
                       help="Pipeline mode")

    # Config options
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--experiment", type=str, default="PKDOT_baseline",
                       help="Experiment name")
    parser.add_argument("--dataset", type=str, default="IEMO",
                       help="Training dataset")

    # Training options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--seeds", type=int, nargs='+', help="Multiple seeds")

    # For resuming
    parser.add_argument("--teacher_dir", type=str,
                       help="Directory with teacher checkpoints (for student_only mode)")
    parser.add_argument("--student_checkpoint", type=str,
                       help="Student checkpoint path (for eval_only mode)")

    # Run all experiments
    parser.add_argument("--all", action="store_true",
                       help="Run all experiments from config file")

    args = parser.parse_args()

    if args.all and args.config:
        # Run all experiments from YAML
        run_experiments_from_yaml(args.config)

    elif args.config:
        # Run experiments from YAML
        run_experiments_from_yaml(args.config)

    else:
        # Single experiment run
        config = PKDOTConfig()
        config.experiment_name = args.experiment
        config.train_dataset = args.dataset

        seeds = args.seeds if args.seeds else [args.seed]

        for seed in seeds:
            config.seed = seed

            if args.mode == "full":
                run_full_pipeline(config, seed=seed)

            elif args.mode == "teachers_only":
                train_all_teachers(config, seed=seed)

            elif args.mode == "student_only":
                if not args.teacher_dir:
                    raise ValueError("--teacher_dir required for student_only mode")

                checkpoint_dir = Path(args.teacher_dir)
                teacher_checkpoints = {
                    'easy': checkpoint_dir / 'teacher_easy_best.pt',
                    'medium': checkpoint_dir / 'teacher_medium_best.pt',
                    'full': checkpoint_dir / 'teacher_full_best.pt'
                }

                proto_scores = np.load(checkpoint_dir / 'proto_scores.npy')
                threshold_values = np.load(checkpoint_dir / 'threshold_values.npy')

                train_student_with_kd(
                    config,
                    teacher_checkpoints,
                    proto_scores,
                    threshold_values,
                    seed=seed
                )

            elif args.mode == "eval_only":
                if not args.student_checkpoint:
                    raise ValueError("--student_checkpoint required for eval_only mode")

                evaluate_cross_corpus(args.student_checkpoint, config, seed)
