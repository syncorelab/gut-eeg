"""
Full CNN-LSTM-attention classification model for 3-class prediction.

This model combines:
1. A CNN feature extractor over the input signal
2. An LSTM to model temporal structure in the CNN features
3. Attention pooling to summarize the sequence
4. A classifier head that outputs class logits
"""

import torch.nn as nn
from .cnn_model import cnnNet
from .lstm_model import LSTMNet, AttentionPooling
from .classifier_model import Classifier


class GUTNet(nn.Module):
    def __init__(
        self,
        conv1_hidden_channels,
        conv2_hidden_channels,
        conv3_output_channels,
        conv1_kernel_size,
        conv2_kernel_size,
        conv3_kernel_size,
        conv1_stride,
        conv2_stride,
        conv3_stride,
        maxpool_ks,
        conv_dropout,
        activation_fn="relu",
        n_conv_blocks=3,
        lstm_hidden_size=None,
        lstm_num_layers=None,
        lstm_dropout=None,
        bidirectional=None,
        attn_hidden_dim=None,
        attn_dropout_attn=None,
        return_attention=False,
        class_dropout=0.5,
        linear1_hidden_size=128,
    ):
        """
        Build the full CNN-LSTM-attention model.

        The CNN extracts local temporal features, the LSTM models their
        sequence over time, attention pooling compresses the sequence into
        one feature vector, and the classifier head maps that vector to
        3 output classes.
        """
        super().__init__()

        # CNN backbone for initial feature extraction from the raw input.
        self.cnn = cnnNet(
            conv1_hidden_channels=conv1_hidden_channels,
            conv2_hidden_channels=conv2_hidden_channels,
            conv3_output_channels=conv3_output_channels,
            conv1_kernel_size=conv1_kernel_size,
            conv2_kernel_size=conv2_kernel_size,
            conv3_kernel_size=conv3_kernel_size,
            conv1_stride=conv1_stride,
            conv2_stride=conv2_stride,
            conv3_stride=conv3_stride,
            maxpool_ks=maxpool_ks,
            conv_dropout=conv_dropout,
            activation_fn=activation_fn,
            n_conv_blocks=n_conv_blocks,
        )

        # The LSTM input size depends on the output channels of the last
        # active CNN block.
        cnn_output_dim = self.cnn.output_channels

        # LSTM sequence model over CNN features.
        self.lstm = LSTMNet(
            input_feature_dim=cnn_output_dim,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            lstm_dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        # Attention pooling reduces the full LSTM sequence to one vector.
        lstm_output_dim = lstm_hidden_size * (2 if bidirectional else 1)
        self.attn_pool = AttentionPooling(
            feature_dim=lstm_output_dim,
            attn_hidden_dim=attn_hidden_dim,
            attn_dropout=attn_dropout_attn,
        )

        # Final classifier that produces 3 class logits.
        self.head = Classifier(
            in_feature=lstm_output_dim,
            class_dropout=class_dropout,
            linear1_hidden_size=linear1_hidden_size,
            activation_fn=activation_fn,
        )

        # Stored for compatibility or future use if attention weights should
        # later be exposed by the model.
        self.return_attention = return_attention

    def forward(self, x):
        """
        Pass the input through the CNN, LSTM, attention pooling, and
        classifier head.
        """
        x = self.cnn(x)

        # Reorder from (batch, channels, time) to (batch, time, features)
        # because the LSTM expects time as the middle dimension.
        x = x.permute(0, 2, 1)

        x = self.lstm(x)
        x = self.attn_pool(x)
        x = self.head(x)

        return x