#!/usr/bin/env python3
"""
Configuration class for Prototypicality-Guided Multi-Teacher Knowledge Distillation
Extended to support multiple teachers and student training
"""

class PKDOTConfig:
    """Configuration class for multi-teacher KD with prototypicality guidance"""

    def __init__(self):
        # Experiment metadata
        self.experiment_name = "PKDOT_baseline"
        self.wandb_project = "PKDOT_Multimodal_Emotion"

        # Dataset settings
        self.train_dataset = "IEMO"  # or "MSPI", "MSPP"
        self.evaluation_mode = "cross_corpus"  # "loso", "cross_corpus", "both"
        self.val_split = 0.2

        # Random seeds for reproducibility
        self.seeds = [42]  # Can be [42, 87739, 1829] for multi-seed
        self.seed = 42

        # Prototypicality settings
        self.prototypicality_method = "vad_distance"  # "vad_distance", "feature_centroid", "learned"
        self.quantile_split = [0.33, 0.67]  # Thresholds for easy/medium/hard
        self.normalize_prototypicality = True

        # Expected VAD values for difficulty calculation (same as Emotion2Vec_Text)
        self.expected_vad = {
            0: [3.0, 2.5, 3.0],  # neutral
            1: [4.0, 3.8, 3.8],  # happy
            2: [1.8, 2.2, 2.0],  # sad
            3: [1.8, 4.2, 4.0]   # anger
        }
        self.difficulty_method = "euclidean_distance"

        # Teacher configurations (3 teachers: easy, medium, full)
        self.teachers = {
            "easy": self._create_teacher_config("easy", [0.0, 0.33]),
            "medium": self._create_teacher_config("medium", [0.33, 0.67]),
            "full": self._create_teacher_config("full", [0.0, 1.0])
        }

        # Student configuration
        self.student = self._create_student_config()

        # Class labels
        self.class_names = ["neutral", "happy", "sad", "anger"]
        self.num_classes = 4

    def _create_teacher_config(self, name, difficulty_range):
        """Create configuration for a single teacher"""
        return {
            "name": name,
            "difficulty_range": difficulty_range,  # [min, max] normalized prototypicality

            # Model architecture
            "modality": "both",  # "audio", "text", or "both"
            "fusion_type": "concat",  # "concat", "cross_attention", "gated", "adaptive"
            "fusion_hidden_dim": 512,
            "num_attention_heads": 8,
            "audio_dim": 768,
            "text_model_name": "bert-base-uncased",
            "freeze_text_encoder": True,
            "text_max_length": 128,
            "hidden_dim": 1024,
            "dropout": 0.1,

            # Training settings
            "num_epochs": 30,
            "batch_size": 64,
            "learning_rate": 1e-5,
            "weight_decay": 5e-6,
            "early_stopping_patience": 5,  # Default early stopping

            # Curriculum learning (only for "full" teacher typically)
            "use_curriculum_learning": (name == "full"),
            "curriculum_epochs": 15,
            "curriculum_pacing": "sqrt",
            "curriculum_type": "difficulty",
            "use_difficulty_scaling": True,
            "use_speaker_disentanglement": True,

            # Class weights
            "class_weights": {
                "neutral": 1.0,
                "happy": 1.0,
                "sad": 1.0,
                "anger": 1.0
            }
        }

    def _create_student_config(self):
        """Create configuration for the student model"""
        return {
            # Model architecture (can be same or different from teachers)
            "modality": "both",
            "fusion_type": "concat",
            "fusion_hidden_dim": 512,
            "num_attention_heads": 8,
            "audio_dim": 768,
            "text_model_name": "bert-base-uncased",
            "freeze_text_encoder": True,
            "text_max_length": 128,
            "hidden_dim": 1024,
            "dropout": 0.1,

            # Training settings
            "num_epochs": 30,
            "batch_size": 64,
            "learning_rate": 1e-5,
            "weight_decay": 5e-6,
            "early_stopping_patience": 5,  # Default early stopping

            # Knowledge distillation parameters
            "kd_temperature": 4.0,  # Temperature for KD (higher = softer)
            "kd_alpha": 0.7,  # Weight for KD loss (1 - alpha for task loss)
            "routing_strategy": "prototypicality",  # "prototypicality", "confidence", "ensemble"

            # Advanced KD options
            "use_soft_labels": True,  # Use teacher soft labels
            "use_hard_labels": True,  # Also use ground truth
            "kd_loss_type": "kl_div",  # "kl_div", "mse", "cosine"
        }

    def __repr__(self):
        """String representation of config"""
        config_str = "PKDOT Configuration:\n"
        config_str += f"  Experiment: {self.experiment_name}\n"
        config_str += f"  Train Dataset: {self.train_dataset}\n"
        config_str += f"  Prototypicality Method: {self.prototypicality_method}\n"
        config_str += f"  Quantile Split: {self.quantile_split}\n"
        config_str += f"  Teachers: {list(self.teachers.keys())}\n"
        config_str += f"  Student KD Temperature: {self.student['kd_temperature']}\n"
        config_str += f"  Student KD Alpha: {self.student['kd_alpha']}\n"
        return config_str

    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
