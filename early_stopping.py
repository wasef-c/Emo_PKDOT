#!/usr/bin/env python3
"""
Early Stopping utility for PKDOT training
"""

class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True, mode='max'):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            restore_best_weights (bool): Whether to restore model weights from the best epoch.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the 
                       quantity monitored has stopped decreasing; in 'max' mode it will 
                       stop when the quantity has stopped increasing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = lambda a, b: a < b - self.min_delta
        elif mode == 'max':
            self.monitor_op = lambda a, b: a > b + self.min_delta
        else:
            raise ValueError(f"Mode {mode} is unknown, should be 'min' or 'max'")
    
    def __call__(self, current, model=None):
        """
        Check if early stopping criteria is met.
        
        Args:
            current (float): Current value of the monitored metric
            model: PyTorch model (optional, for saving best weights)
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best is None:
            self.best = current
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.wait
                return True
        return False
    
    def restore_weights(self, model):
        """Restore the best weights to the model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
        
    def get_best_score(self):
        """Get the best score achieved."""
        return self.best