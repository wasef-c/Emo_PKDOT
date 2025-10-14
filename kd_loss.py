#!/usr/bin/env python3
"""
Knowledge Distillation Loss Functions
Implements various KD losses for multi-teacher distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Standard Knowledge Distillation Loss using KL Divergence
    Based on Hinton et al. "Distilling the Knowledge in a Neural Network"
    """

    def __init__(self, temperature=4.0, alpha=0.7, reduction='batchmean'):
        """
        Args:
            temperature: Softening parameter for logits (higher = softer)
            alpha: Weight for KD loss (1-alpha for task loss)
            reduction: How to reduce the loss ('batchmean', 'mean', 'sum')
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, student_logits, teacher_logits, targets):
        """
        Compute combined KD + task loss

        Args:
            student_logits: [batch_size, num_classes] raw student outputs
            teacher_logits: [batch_size, num_classes] raw teacher outputs
            targets: [batch_size] ground truth labels

        Returns:
            loss: Combined distillation + task loss
            loss_dict: Dict with individual loss components
        """
        # KD loss: KL divergence between softened distributions
        loss_kd = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction=self.reduction
        ) * (self.temperature ** 2)  # Scale by T^2 to match gradient magnitudes

        # Task loss: Standard cross-entropy with ground truth
        loss_task = F.cross_entropy(student_logits, targets, reduction='mean')

        # Combined loss
        loss = self.alpha * loss_kd + (1 - self.alpha) * loss_task

        loss_dict = {
            'loss_kd': loss_kd.item(),
            'loss_task': loss_task.item(),
            'loss_total': loss.item()
        }

        return loss, loss_dict


class MultiTeacherKDLoss(nn.Module):
    """
    Multi-Teacher Knowledge Distillation Loss
    Each sample can learn from a different teacher based on routing
    """

    def __init__(self, temperature=4.0, alpha=0.7, reduction='batchmean'):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, student_logits, teacher_logits_batch, targets):
        """
        Compute KD loss where each sample has its own teacher logits

        Args:
            student_logits: [batch_size, num_classes]
            teacher_logits_batch: [batch_size, num_classes] - already routed per sample
            targets: [batch_size]

        Returns:
            loss: Combined loss
            loss_dict: Dict with loss components
        """
        # KD loss with per-sample teacher
        loss_kd = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits_batch / self.temperature, dim=1),
            reduction=self.reduction
        ) * (self.temperature ** 2)

        # Task loss
        loss_task = F.cross_entropy(student_logits, targets, reduction='mean')

        # Combined
        loss = self.alpha * loss_kd + (1 - self.alpha) * loss_task

        loss_dict = {
            'loss_kd': loss_kd.item(),
            'loss_task': loss_task.item(),
            'loss_total': loss.item()
        }

        return loss, loss_dict


class EnsembleKDLoss(nn.Module):
    """
    Ensemble Knowledge Distillation
    Student learns from averaged outputs of all teachers
    """

    def __init__(self, temperature=4.0, alpha=0.7, ensemble_weights=None):
        """
        Args:
            temperature: Softening parameter
            alpha: KD loss weight
            ensemble_weights: Optional weights for each teacher [w1, w2, w3]
                            If None, uses uniform weights
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ensemble_weights = ensemble_weights

    def forward(self, student_logits, teacher_logits_list, targets):
        """
        Compute KD loss from ensemble of teachers

        Args:
            student_logits: [batch_size, num_classes]
            teacher_logits_list: List of [batch_size, num_classes] from each teacher
            targets: [batch_size]

        Returns:
            loss: Combined loss
            loss_dict: Loss components
        """
        # Ensemble teacher logits
        if self.ensemble_weights is None:
            # Uniform weights
            ensemble_logits = torch.stack(teacher_logits_list).mean(dim=0)
        else:
            # Weighted ensemble
            weights = torch.tensor(self.ensemble_weights, device=teacher_logits_list[0].device)
            weights = weights / weights.sum()  # Normalize
            ensemble_logits = sum(w * logits for w, logits in zip(weights, teacher_logits_list))

        # KD loss
        loss_kd = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(ensemble_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Task loss
        loss_task = F.cross_entropy(student_logits, targets, reduction='mean')

        # Combined
        loss = self.alpha * loss_kd + (1 - self.alpha) * loss_task

        loss_dict = {
            'loss_kd': loss_kd.item(),
            'loss_task': loss_task.item(),
            'loss_total': loss.item()
        }

        return loss, loss_dict


class MSEKDLoss(nn.Module):
    """
    Knowledge Distillation using MSE loss (alternative to KL divergence)
    Sometimes works better for certain tasks
    """

    def __init__(self, temperature=1.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, targets):
        """
        Compute MSE-based KD loss

        Args:
            student_logits: [batch_size, num_classes]
            teacher_logits: [batch_size, num_classes]
            targets: [batch_size]

        Returns:
            loss: Combined loss
            loss_dict: Loss components
        """
        # Soften logits
        student_soft = F.softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        # MSE loss
        loss_kd = F.mse_loss(student_soft, teacher_soft, reduction='mean')

        # Task loss
        loss_task = F.cross_entropy(student_logits, targets, reduction='mean')

        # Combined
        loss = self.alpha * loss_kd + (1 - self.alpha) * loss_task

        loss_dict = {
            'loss_kd': loss_kd.item(),
            'loss_task': loss_task.item(),
            'loss_total': loss.item()
        }

        return loss, loss_dict


def create_kd_loss(config):
    """
    Factory function to create KD loss based on config

    Args:
        config: PKDOTConfig or dict with KD settings

    Returns:
        nn.Module: KD loss module
    """
    # Extract params
    if hasattr(config, 'student'):
        student_config = config.student
    else:
        student_config = config

    temperature = student_config.get('kd_temperature', 4.0)
    alpha = student_config.get('kd_alpha', 0.7)
    loss_type = student_config.get('kd_loss_type', 'kl_div')
    routing_strategy = student_config.get('routing_strategy', 'prototypicality')

    # Select loss based on routing strategy and type
    if routing_strategy == "prototypicality":
        # Per-sample teacher routing
        return MultiTeacherKDLoss(temperature, alpha)

    elif routing_strategy == "ensemble":
        # Ensemble all teachers
        ensemble_weights = student_config.get('ensemble_weights', None)
        return EnsembleKDLoss(temperature, alpha, ensemble_weights)

    elif loss_type == "mse":
        return MSEKDLoss(temperature, alpha)

    else:
        # Default: standard KD
        return KnowledgeDistillationLoss(temperature, alpha)
