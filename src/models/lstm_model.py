import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class LSTMNet(nn.Module):
    """
    LSTM module for sequential feature modeling.

    This block expects input shaped as:
        (batch_size, sequence_length, feature_dim)

    It returns the full output sequence from the LSTM:
        (batch_size, sequence_length, hidden_size * num_directions)

    No extra activation is applied after the LSTM.
    """

    def __init__(
        self,
        input_feature_dim: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        lstm_dropout: float,
        bidirectional: bool,
    ):
        super().__init__()

        self.input_feature_dim = int(input_feature_dim)
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.bidirectional = bool(bidirectional)

        self.lstm = nn.LSTM(
            input_size=self.input_feature_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=int(lstm_num_layers),
            dropout=float(lstm_dropout) if int(lstm_num_layers) > 1 else 0.0,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

    @property
    def out_feature_dim(self) -> int:
        """Return the feature size of each timestep in the LSTM output."""
        return self.lstm_hidden_size * (2 if self.bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check that the input has shape (batch, time, features).
        if x.ndim != 3:
            raise ValueError(
                f"LSTMNet expected a 3D tensor (B, T, F). Got shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.input_feature_dim:
            raise ValueError(
                "LSTMNet got wrong feature dim. "
                f"Expected last dim = {self.input_feature_dim} but got {x.shape[-1]}. "
                "If your CNN outputs (B, C, T), you probably forgot x = x.permute(0, 2, 1)."
            )

        # Run the sequence through the LSTM and return all timesteps.
        x, _ = self.lstm(x)

        return x


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over the time dimension.

    Input:
        x: (batch_size, sequence_length, feature_dim)

    Output:
        pooled: (batch_size, feature_dim)

    If requested, the module can also return the attention weights:
        weights: (batch_size, sequence_length)

    The attention score for each timestep is computed with:
        score_t = v^T tanh(W x_t)
    """

    def __init__(
        self,
        feature_dim: int,
        attn_hidden_dim: int,
        attn_dropout: float = 0.0,
        return_attention: bool = False
    ):
        super().__init__()

        self.feature_dim = int(feature_dim)
        self.attn_hidden_dim = int(attn_hidden_dim)
        self.return_attention_default = bool(return_attention)

        # Learn a scalar attention score for each timestep.
        self.attn_W = nn.Linear(self.feature_dim, self.attn_hidden_dim)
        self.attn_v = nn.Linear(self.attn_hidden_dim, 1, bias=False)

        self.attn_dropout = nn.Dropout(float(attn_dropout)) if attn_dropout and attn_dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        return_attention: Optional[bool] = None
    ):
        # Check that the input has shape (batch, time, features).
        if x.ndim != 3:
            raise ValueError(f"AttentionPooling expected (B, T, F). Got {tuple(x.shape)}")
        if x.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {x.shape[-1]}")

        if return_attention is None:
            return_attention = self.return_attention_default

        # Keep the original timestep features for the final weighted sum.
        feats = x

        # Compute unnormalized attention scores for each timestep.
        h = self.attn_W(feats)
        h = torch.tanh(h)
        h = self.attn_dropout(h)
        scores = self.attn_v(h)

        # Normalize scores across time so they sum to 1 for each sequence.
        weights = torch.softmax(scores, dim=1)

        # Compute a weighted sum of the original features across time.
        pooled = torch.sum(weights * feats, dim=1)

        if return_attention:
            return pooled, weights.squeeze(-1)

        return pooled