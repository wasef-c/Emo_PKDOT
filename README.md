# PKDOT: Prototypicality-Guided Multi-Teacher Knowledge Distillation

A novel approach for multimodal emotion recognition using multiple specialized teacher models and knowledge distillation with prototypicality-based routing.

## Overview

PKDOT trains three specialized teacher models on different difficulty subsets of data:
- **Easy Teacher**: Trains on bottom 33% (most prototypical samples)
- **Medium Teacher**: Trains on middle 33% (moderately prototypical samples)
- **Full Teacher**: Trains on all data (handles hard/atypical samples)

A student model then learns from these teachers using Knowledge Distillation, where each sample is routed to the appropriate teacher based on its prototypicality score.

## Architecture

```
Phase 1: Train Teachers
├── Compute prototypicality scores (VAD distance)
├── Split data by quantiles [0.33, 0.67]
├── Train Easy Teacher (bottom 33%)
├── Train Medium Teacher (middle 33%)
└── Train Full Teacher (all data)

Phase 2: Train Student with KD
├── Load frozen teachers
├── For each sample:
│   ├── Compute prototypicality score
│   ├── Route to appropriate teacher
│   └── Use teacher logits for KD loss
└── Train student with KD + task loss

Phase 3: Evaluation
└── Cross-corpus testing on IEMO, MSPI, MSPP, CMUMOSEI, SAMSEMO
```

## Key Features

- **Prototypicality-Based Routing**: Samples routed to teachers based on difficulty
- **Multi-Teacher KD**: Learn from specialized experts
- **Multimodal Fusion**: Emotion2vec (audio) + BERT (text)
- **Multiple Fusion Mechanisms**: Concat, Cross-Attention, Gated, Adaptive
- **Curriculum Learning**: Optional for teacher training
- **Cross-Corpus Evaluation**: Generalization testing

## Installation

```bash
# Clone repository
cd /home/rml/Documents/pythontest/Emotion2VecTraining/Emotion2Vec_PKDOT

# Install dependencies
pip install torch transformers datasets wandb pyyaml tqdm scikit-learn
```

## Quick Start

### 1. Train Teachers and Student (Full Pipeline)

```bash
python main.py \
  --mode full \
  --dataset IEMO \
  --experiment PKDOT_baseline \
  --seed 42
```

### 2. Run Experiments from Config

```bash
# Run all experiments in config file
python main.py --config configs/pkd_baseline.yaml --all

# Run single experiment from config
python main.py --config configs/pkd_baseline.yaml
```

### 3. Train Only Teachers

```bash
python train_teachers.py \
  --dataset IEMO \
  --experiment PKDOT_test \
  --seed 42
```

### 4. Train Only Student (with existing teachers)

```bash
python train_student.py \
  --dataset IEMO \
  --experiment PKDOT_test \
  --teacher_dir checkpoints/PKDOT_test_seed42 \
  --seed 42
```

## Configuration

### Config File Structure

```yaml
template: &template
  wandb_project: "PKDOT_Multimodal_Emotion"
  train_dataset: "IEMO"
  seeds: [42, 87739, 1829]

  # Prototypicality
  prototypicality_method: "vad_distance"
  quantile_split: [0.33, 0.67]

  # Teacher configs
  teacher_overrides:
    easy:
      fusion_type: "concat"
      num_epochs: 30
      # ...

  # Student configs
  student_overrides:
    kd_temperature: 4.0
    kd_alpha: 0.7
    routing_strategy: "prototypicality"
```

### Key Parameters

**Prototypicality:**
- `prototypicality_method`: "vad_distance" or "feature_centroid"
- `quantile_split`: Thresholds for easy/medium/hard (default: [0.33, 0.67])
- `difficulty_method`: "euclidean_distance" or "manhattan_distance"

**Knowledge Distillation:**
- `kd_temperature`: Softening parameter (default: 4.0)
- `kd_alpha`: Weight for KD loss vs task loss (default: 0.7)
- `routing_strategy`: "prototypicality", "ensemble", or "confidence"
- `kd_loss_type`: "kl_div" or "mse"

**Model Architecture:**
- `fusion_type`: "concat", "cross_attention", "gated", "adaptive"
- `fusion_hidden_dim`: 512 (concat) or 1024 (cross-attention)
- `num_attention_heads`: 8 or 16 (for attention-based fusion)

## Datasets

Supported datasets:
- **IEMO** (IEMOCAP)
- **MSPI** (MSP-IMPROV)
- **MSPP** (MSP-PODCAST)
- **CMUMOSEI** (CMU-MOSEI)
- **SAMSEMO** (SAMSE-MO)

All datasets should be preprocessed with emotion2vec features and saved as pickle files in:
```
/home/rml/Documents/audio/emotion2vec_base_finetuned/
```

## Experiments

### Baseline Experiments (pkd_baseline.yaml)

- IEMO with Simple Concat (Full vs NoRouting)
- IEMO with Cross-Attention 1024
- MSPI with Simple Concat
- MSPI with Cross-Attention 1024

### Ablation Studies (pkd_ablation.yaml)

- **Temperature**: [2.0, 4.0, 6.0, 8.0]
- **Alpha**: [0.5, 0.7, 0.9]
- **Loss Type**: KL Divergence vs MSE
- **Routing**: Prototypicality vs Ensemble
- **Quantile Split**: [0.25, 0.75], [0.33, 0.67], [0.40, 0.60]

## Expected Results

Based on best-performing fusion methods from Emotion2Vec_Text:
- **Simple Concat**: ~65-70% cross-corpus accuracy
- **Cross-Attention 1024**: ~68-72% cross-corpus accuracy

**PKDOT Expected Improvement**: +2-5% over single-model baselines

## File Structure

```
Emotion2Vec_PKDOT/
├── config.py                 # PKDOTConfig class
├── dataset.py                # Dataset loading utilities
├── prototypicality.py        # Prototypicality calculation
├── kd_loss.py               # Knowledge distillation losses
├── train_teachers.py        # Phase 1: Train teachers
├── train_student.py         # Phase 2: Train student with KD
├── main.py                  # Main orchestrator
├── model.py                 # Model architectures (from Emotion2Vec_Text)
├── text_encoder.py          # BERT encoder (from Emotion2Vec_Text)
├── functions.py             # Utility functions (from Emotion2Vec_Text)
├── configs/
│   ├── pkd_baseline.yaml    # Baseline experiments
│   └── pkd_ablation.yaml    # Ablation studies
└── checkpoints/             # Saved models

```

## Checkpoints

Checkpoints are saved to:
```
checkpoints/{experiment_name}_seed{seed}/
├── teacher_easy_best.pt
├── teacher_medium_best.pt
├── teacher_full_best.pt
├── student_best.pt
├── proto_scores.npy
├── threshold_values.npy
├── teacher_results.json
├── student_results.json
└── complete_results.json
```

## WandB Logging

All experiments log to Weights & Biases:

**Teachers:**
- `{teacher_name}/train_loss`
- `{teacher_name}/train_accuracy`
- `{teacher_name}/val_accuracy`
- `{teacher_name}/val_uar`

**Student:**
- `student/train_loss_total`
- `student/train_loss_kd`
- `student/train_loss_task`
- `student/val_accuracy`
- `student/val_uar`

## Citation

If you use this code, please cite:

```bibtex
@article{pkdot2025,
  title={PKDOT: Prototypicality-Guided Multi-Teacher Knowledge Distillation for Multimodal Emotion Recognition},
  author={Your Name},
  year={2025}
}
```

## License

MIT License

## Acknowledgments

- Based on Emotion2Vec_Text multimodal emotion recognition framework
- Uses emotion2vec for audio features
- Uses BERT for text encoding
