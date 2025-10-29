#!/usr/bin/env python3
"""
Main orchestrator for PKDOT (Prototypicality-Guided Multi-Teacher Knowledge Distillation)

Runs complete pipeline:
1. Phase 1: Train specialized teacher models
2. Phase 2: Train student with KD from teachers
3. Evaluation: Cross-corpus testing
"""

import os
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import wandb
import pandas as pd
from collections import defaultdict

from config import PKDOTConfig
from train_teachers import train_all_teachers
from train_student import train_student_with_kd
from dataset import MultimodalEmotionDataset, collate_multimodal_batch
from torch.utils.data import DataLoader
from functions import evaluate_model


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
    
    # Cross-corpus evaluation should log to the same wandb run as student training
    # wandb should already be initialized by student training
    if not wandb.run:
        # Fallback initialization if wandb is not running
        wandb.init(
            project=config.wandb_project,
            name=f"{config.experiment_name}_seed{seed}",
            config=config.to_dict()
        )

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
                split="train",
                config=config,
                Train=False
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=64,
                shuffle=False,
                collate_fn=collate_multimodal_batch,
                num_workers=4
            )

            # Create criterion for evaluation
            criterion = torch.nn.CrossEntropyLoss()
            test_results = evaluate_model(model, test_loader, criterion, device)

            print(f"  Accuracy: {test_results['accuracy']*100:.2f}%")
            print(f"  UAR: {test_results['uar']*100:.2f}%")
            print(f"  F1: {test_results['f1']:.2f}%")

            results[test_dataset_name] = test_results
            
            # Log to wandb
            wandb.log({
                f"cross_corpus/{config.train_dataset}_to_{test_dataset_name}/accuracy": test_results['accuracy'] * 100,
                f"cross_corpus/{config.train_dataset}_to_{test_dataset_name}/uar": test_results['uar'] * 100,
                f"cross_corpus/{config.train_dataset}_to_{test_dataset_name}/f1": test_results['f1'],
                f"cross_corpus/{config.train_dataset}_to_{test_dataset_name}/loss": test_results['loss']
            })

        except Exception as e:
            print(f"  ⚠️  Error evaluating {test_dataset_name}: {e}")
            results[test_dataset_name] = None

    # Log summary metrics
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        avg_accuracy = sum(r['accuracy'] for r in valid_results.values()) / len(valid_results) * 100
        avg_uar = sum(r['uar'] for r in valid_results.values()) / len(valid_results) * 100
        avg_f1 = sum(r['f1'] for r in valid_results.values()) / len(valid_results)
        
        wandb.log({
            "cross_corpus_summary/avg_accuracy": avg_accuracy,
            "cross_corpus_summary/avg_uar": avg_uar,
            "cross_corpus_summary/avg_f1": avg_f1,
            "cross_corpus_summary/num_datasets_evaluated": len(valid_results)
        })
        
        print(f"\n📈 Cross-Corpus Summary:")
        print(f"  Average Accuracy: {avg_accuracy:.2f}%")
        print(f"  Average UAR: {avg_uar:.2f}%")
        print(f"  Average F1: {avg_f1:.2f}")
        print(f"  Datasets evaluated: {len(valid_results)}")

    # Log individual seed results to wandb
    if wandb.run:
        wandb.log({
            f"individual_seed_{seed}/cross_corpus_avg_accuracy": avg_accuracy if valid_results else 0,
            f"individual_seed_{seed}/cross_corpus_avg_uar": avg_uar if valid_results else 0,
            f"individual_seed_{seed}/cross_corpus_avg_f1": avg_f1 if valid_results else 0,
            f"individual_seed_{seed}/num_datasets_evaluated": len(valid_results)
        })

    wandb.finish()
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

    # Check for existing teacher checkpoints
    checkpoint_dir = Path(f"checkpoints/{config.experiment_name}/{seed}")
    teacher_names = list(config.teachers.keys())  # Get actual teacher names from config
    teacher_checkpoints_exist = all([
        (checkpoint_dir / f"teacher_{name}_best.pt").exists() 
        for name in teacher_names
    ])
    proto_data_exists = (
        (checkpoint_dir / 'proto_scores.npy').exists() and 
        (checkpoint_dir / 'threshold_values.npy').exists()
    )
    
    # Auto-detect if we should skip teacher training
    if teacher_checkpoints_exist and proto_data_exists and not skip_teachers:
        print(f"\n✅ Found existing teacher checkpoints in {checkpoint_dir}")
        print("   - teacher_easy_best.pt")
        print("   - teacher_medium_best.pt") 
        print("   - teacher_full_best.pt")
        print("   - proto_scores.npy")
        print("   - threshold_values.npy")
        print("⏭️  Skipping teacher training, using existing checkpoints")
        skip_teachers = True
        results['teachers'] = 'loaded_existing'
    elif skip_teachers and teacher_dir:
        print("\n⏭️  Manually skipping teacher training, using provided checkpoints")
        checkpoint_dir = Path(teacher_dir)
        results['teachers'] = 'manually_skipped'

    # Phase 1: Train Teachers (if needed)
    if not skip_teachers:
        print("\n" + "="*60)
        print("PHASE 1: Train Teacher Models")
        print("="*60)

        teacher_results = train_all_teachers(config, seed)
        results['teachers'] = teacher_results

    # Check for existing student checkpoint
    student_checkpoint_exists = (checkpoint_dir / "student_best.pt").exists()
    
    # Phase 2: Train Student
    if student_checkpoint_exists:
        print(f"\n✅ Found existing student checkpoint: {checkpoint_dir / 'student_best.pt'}")
        print("⏭️  Skipping student training, using existing checkpoint")
        
        # Load existing student results (you might want to implement this)
        results['student'] = 'loaded_existing'
    else:
        print("\n" + "="*60)
        print("PHASE 2: Train Student with KD")
        print("="*60)

    # Load teacher checkpoints
    teacher_checkpoints = {
        name: checkpoint_dir / f'teacher_{name}_best.pt'
        for name in config.teachers.keys()
    }

    # Verify checkpoints exist
    for name, path in teacher_checkpoints.items():
        if not path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {path}")

    # Load prototypicality data
    proto_scores = np.load(checkpoint_dir / 'proto_scores.npy')
    threshold_values = np.load(checkpoint_dir / 'threshold_values.npy')

    if not student_checkpoint_exists:
        student_results = train_student_with_kd(
            config,
            teacher_checkpoints,
            proto_scores,
            threshold_values,
            seed=seed
        )
        results['student'] = student_results
    else:
        # If student checkpoint exists, create a placeholder result
        results['student'] = {
            'status': 'loaded_existing',
            'checkpoint_path': str(checkpoint_dir / 'student_best.pt')
        }

    # Phase 3: Cross-Corpus Evaluation
    if config.evaluation_mode in ['cross_corpus', 'both']:
        print("\n" + "="*60)
        print("PHASE 3: Cross-Corpus Evaluation")
        print("="*60)

        # Check for existing cross-corpus results
        cross_corpus_results_file = checkpoint_dir / 'cross_corpus_results.json'
        
        if cross_corpus_results_file.exists():
            print(f"✅ Found existing cross-corpus results: {cross_corpus_results_file}")
            print("⏭️  Loading existing cross-corpus evaluation")
            
            with open(cross_corpus_results_file, 'r') as f:
                cross_corpus_results = json.load(f)
            results['cross_corpus'] = cross_corpus_results
        else:
            print("🔄 Running cross-corpus evaluation...")
            
            # Get student checkpoint path
            if student_checkpoint_exists:
                student_checkpoint = checkpoint_dir / 'student_best.pt'
            else:
                student_checkpoint = student_results['checkpoint_path']
            
            cross_corpus_results = evaluate_cross_corpus(
                student_checkpoint,
                config,
                seed
            )
            results['cross_corpus'] = cross_corpus_results
            
            # Save cross-corpus results
            with open(cross_corpus_results_file, 'w') as f:
                json.dump(cross_corpus_results, f, indent=2, default=str)
            print(f"💾 Saved cross-corpus results to {cross_corpus_results_file}")

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
    if results['student'].get('status') == 'loaded_existing':
        print(f"  Status: Loaded from existing checkpoint")
    else:
        print(f"  Val Accuracy: {results['student']['best_val_accuracy']:.2f}%")
        print(f"  Val UAR: {results['student']['best_val_uar']:.2f}%")

    if 'cross_corpus' in results:
        print(f"\nCross-Corpus Results:")
        for dataset_name, result in results['cross_corpus'].items():
            if result:
                print(f"  {config.train_dataset}→{dataset_name}: "
                      f"Acc {result['accuracy']:.2f}%, "
                      f"UAR {result['uar']:.2f}%, "
                      f"F1 {result['f1']:.2f}%")

    # Log individual seed summary to wandb (if cross-corpus hasn't been run)
    if wandb.run and 'cross_corpus' not in results:
        # Log teacher results
        if isinstance(results['teachers'], dict):
            for teacher_name, teacher_result in results['teachers'].items():
                wandb.log({
                    f"individual_seed_{seed}/teacher_{teacher_name}_val_accuracy": teacher_result['best_val_accuracy'],
                    f"individual_seed_{seed}/teacher_{teacher_name}_val_uar": teacher_result['best_val_uar']
                })
        
        # Log student results
        if results['student'].get('status') != 'loaded_existing':
            wandb.log({
                f"individual_seed_{seed}/student_val_accuracy": results['student']['best_val_accuracy'],
                f"individual_seed_{seed}/student_val_uar": results['student']['best_val_uar']
            })
        
        wandb.finish()

    return results


def aggregate_multi_seed_results(all_results, experiment_name, output_dir="results"):
    """
    Aggregate results across multiple seeds and generate summary statistics
    
    Args:
        all_results: List of result dicts from multiple seeds
        experiment_name: Name of the experiment
        output_dir: Directory to save results
    """
    if not all_results:
        print("No results to aggregate")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Group results by experiment name and collect metrics
    metrics_by_dataset = defaultdict(lambda: defaultdict(list))
    teacher_metrics = defaultdict(lambda: defaultdict(list))
    student_metrics = defaultdict(list)
    
    for result in all_results:
        seed = result['seed']
        exp_name = result['experiment_name']
        
        # Collect teacher metrics
        if isinstance(result.get('teachers'), dict):
            for teacher_name, teacher_result in result['teachers'].items():
                if isinstance(teacher_result, dict):
                    teacher_metrics[teacher_name]['val_accuracy'].append(teacher_result.get('best_val_accuracy', 0))
                    teacher_metrics[teacher_name]['val_uar'].append(teacher_result.get('best_val_uar', 0))
        
        # Collect student metrics
        if isinstance(result.get('student'), dict) and result['student'].get('status') != 'loaded_existing':
            student_metrics['val_accuracy'].append(result['student'].get('best_val_accuracy', 0))
            student_metrics['val_uar'].append(result['student'].get('best_val_uar', 0))
        
        # Collect cross-corpus metrics
        if 'cross_corpus' in result:
            for target_dataset, target_results in result['cross_corpus'].items():
                if target_results is not None:
                    metrics_by_dataset[target_dataset]['accuracy'].append(target_results['accuracy'] * 100)
                    metrics_by_dataset[target_dataset]['uar'].append(target_results['uar'] * 100)
                    metrics_by_dataset[target_dataset]['f1'].append(target_results['f1'])
                    metrics_by_dataset[target_dataset]['loss'].append(target_results['loss'])
    
    # Generate CSV summary
    csv_data = []
    
    # Add teacher results
    for teacher_name, metrics in teacher_metrics.items():
        for metric_name, values in metrics.items():
            if values:
                csv_data.append({
                    'Model': f'Teacher_{teacher_name}',
                    'Dataset': 'Validation',
                    'Metric': metric_name,
                    'Mean': np.mean(values),
                    'Std': np.std(values) if len(values) > 1 else 0.0,
                    'Min': np.min(values),
                    'Max': np.max(values),
                    'Count': len(values)
                })
    
    # Add student results
    for metric_name, values in student_metrics.items():
        if values:
            csv_data.append({
                'Model': 'Student',
                'Dataset': 'Validation',
                'Metric': metric_name,
                'Mean': np.mean(values),
                'Std': np.std(values) if len(values) > 1 else 0.0,
                'Min': np.min(values),
                'Max': np.max(values),
                'Count': len(values)
            })
    
    # Add cross-corpus results
    for dataset_name, metrics in metrics_by_dataset.items():
        for metric_name, values in metrics.items():
            if values:
                csv_data.append({
                    'Model': 'Student',
                    'Dataset': dataset_name,
                    'Metric': metric_name,
                    'Mean': np.mean(values),
                    'Std': np.std(values) if len(values) > 1 else 0.0,
                    'Min': np.min(values),
                    'Max': np.max(values),
                    'Count': len(values)
                })
    
    # Save CSV
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_file = output_path / f"{experiment_name}_multi_seed_summary.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n📊 Saved CSV summary to: {csv_file}")
    
    # Generate detailed text report
    txt_file = output_path / f"{experiment_name}_multi_seed_summary.txt"
    with open(txt_file, 'w') as f:
        f.write(f"PKDOT Multi-Seed Results Summary\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Seeds analyzed: {len(all_results)}\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write("="*60 + "\n\n")
        
        # Teacher Results
        if teacher_metrics:
            f.write("TEACHER MODELS\n")
            f.write("-" * 30 + "\n")
            for teacher_name, metrics in teacher_metrics.items():
                f.write(f"\n{teacher_name.upper()} Teacher:\n")
                for metric_name, values in metrics.items():
                    if values:
                        f.write(f"  {metric_name}: {np.mean(values):.2f} ± {np.std(values):.2f} "
                               f"(range: {np.min(values):.2f}-{np.max(values):.2f}, n={len(values)})\n")
        
        # Student Results
        if student_metrics:
            f.write(f"\nSTUDENT MODEL\n")
            f.write("-" * 30 + "\n")
            for metric_name, values in student_metrics.items():
                if values:
                    f.write(f"  {metric_name}: {np.mean(values):.2f} ± {np.std(values):.2f} "
                           f"(range: {np.min(values):.2f}-{np.max(values):.2f}, n={len(values)})\n")
        
        # Cross-corpus Results
        if metrics_by_dataset:
            f.write(f"\nCROSS-CORPUS EVALUATION\n")
            f.write("-" * 30 + "\n")
            for dataset_name, metrics in metrics_by_dataset.items():
                f.write(f"\n{dataset_name}:\n")
                for metric_name, values in metrics.items():
                    if values:
                        f.write(f"  {metric_name}: {np.mean(values):.2f} ± {np.std(values):.2f} "
                               f"(range: {np.min(values):.2f}-{np.max(values):.2f}, n={len(values)})\n")
        
        # Overall cross-corpus summary
        if metrics_by_dataset:
            f.write(f"\nOVERALL CROSS-CORPUS SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            all_accuracies = [val for metrics in metrics_by_dataset.values() 
                            for val in metrics.get('accuracy', [])]
            all_uars = [val for metrics in metrics_by_dataset.values() 
                       for val in metrics.get('uar', [])]
            all_f1s = [val for metrics in metrics_by_dataset.values() 
                      for val in metrics.get('f1', [])]
            
            if all_accuracies:
                f.write(f"  Average Accuracy: {np.mean(all_accuracies):.2f} ± {np.std(all_accuracies):.2f}%\n")
            if all_uars:
                f.write(f"  Average UAR: {np.mean(all_uars):.2f} ± {np.std(all_uars):.2f}%\n")
            if all_f1s:
                f.write(f"  Average F1: {np.mean(all_f1s):.2f} ± {np.std(all_f1s):.2f}\n")
    
    print(f"📄 Saved detailed summary to: {txt_file}")
    
    # Print summary to console
    print(f"\n📈 {experiment_name} Multi-Seed Summary:")
    if metrics_by_dataset:
        all_accuracies = [val for metrics in metrics_by_dataset.values() 
                        for val in metrics.get('accuracy', [])]
        all_uars = [val for metrics in metrics_by_dataset.values() 
                   for val in metrics.get('uar', [])]
        if all_accuracies and all_uars:
            print(f"  Cross-corpus Accuracy: {np.mean(all_accuracies):.2f} ± {np.std(all_accuracies):.2f}%")
            print(f"  Cross-corpus UAR: {np.mean(all_uars):.2f} ± {np.std(all_uars):.2f}%")
            print(f"  Seeds: {len(all_results)}")


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
            if key == 'name':  # Map 'name' to 'experiment_name'
                setattr(config, 'experiment_name', value)
            elif hasattr(config, key):
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

    # Group results by experiment name for multi-seed aggregation
    results_by_experiment = defaultdict(list)
    for result in all_results:
        # Extract base experiment name (without seed suffix)
        exp_name = result['experiment_name']
        if exp_name.endswith(f"_seed{result['seed']}"):
            base_name = exp_name[:-len(f"_seed{result['seed']}")]
        else:
            base_name = exp_name
        results_by_experiment[base_name].append(result)
    
    # Generate aggregated results for each experiment
    for base_exp_name, exp_results in results_by_experiment.items():
        if len(exp_results) >= 1:  # Generate summary for single or multiple seeds
            print(f"\n🔢 Generating results summary for {base_exp_name} ({len(exp_results)} seed{'s' if len(exp_results) > 1 else ''})")
            aggregate_multi_seed_results(exp_results, base_exp_name)

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
                    name: checkpoint_dir / f'teacher_{name}_best.pt'
                    for name in config.teachers.keys()
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
