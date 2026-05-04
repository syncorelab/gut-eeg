import torch
import copy

class EarlyStopping:
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        """
        Early stopping utility.

        Parameters
        ----------
        patience : int
            Number of epochs to wait after last meaningful improvement.
        min_delta : float
            Minimum change to qualify as an improvement.
        mode : str
            "min" for metrics like validation loss,
            "max" for metrics like validation accuracy.
        restore_best_weights : bool
            If True, stores and restores the best model weights.
        """
        if mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score = None
        self.best_epoch = None
        self.best_state_dict = None
        self.counter = 0
        self.should_stop = False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True

        if self.mode == "min":
            return score < (self.best_score - self.min_delta)
        else:
            return score > (self.best_score + self.min_delta)

    def step(self, score: float, model: torch.nn.Module, epoch: int):
        """
        Update early stopping state after each validation epoch.

        Parameters
        ----------
        score : float
            Current validation metric.
        model : torch.nn.Module
            Model being trained.
        epoch : int
            Current epoch number.
        """
        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

            if self.restore_best_weights:
                self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore(self, model: torch.nn.Module):
        """Restore best model weights if available."""
        if self.restore_best_weights and self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)