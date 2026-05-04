"""
Classifier head for the final emotion prediction stage.

This module takes the feature vector produced by the previous part of the
model and maps it to 3 output classes.
"""

import torch.nn as nn


class Classifier(nn.Module):
    def __init__(
        self,
        in_feature,
        class_dropout,
        linear1_hidden_size,
        activation_fn,
    ):
        """
        Build a small fully connected classifier.

        Parameters
        ----------
        in_feature : int
            Size of the incoming feature vector from the previous model block.
        class_dropout : float
            Dropout probability used before the final output layer.
        linear1_hidden_size : int
            Number of hidden units in the classifier layer.
        activation_fn : str
            Name of the activation function to use.
        """
        super().__init__()
        self.activation = self._get_activation(activation_fn)

        # First linear layer that transforms the incoming feature vector.
        self.fc1 = nn.Linear(in_feature, linear1_hidden_size)

        # Dropout helps regularize the classifier head.
        self.dropout = nn.Dropout(p=class_dropout)

        # Final output layer for 3-class emotion classification.
        self.fc_out = nn.Linear(linear1_hidden_size, 3)

    def _get_activation(self, name: str):
        """Return the activation module selected by name."""
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        elif name == "leakyrelu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation_fn: {name}")

    def forward(self, x):
        """Map input features to class logits."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x