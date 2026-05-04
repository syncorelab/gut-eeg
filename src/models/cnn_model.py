"""
CNN feature extractor for multichannel time-series classification.

This module applies 1 to 3 convolutional blocks to the input signal and
returns the extracted feature maps. The number of active blocks is
controlled by the n_conv_blocks parameter.
"""

import torch.nn as nn


class cnnNet(nn.Module):
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
        activation_fn,
        n_conv_blocks,
    ):
        """
        Build a CNN with up to 3 convolutional blocks.

        Each block contains:
        1. 1D convolution
        2. Batch normalization
        3. Activation function
        4. Max pooling
        5. Dropout
        """
        super().__init__()

        if n_conv_blocks not in [1, 2, 3]:
            raise ValueError(f"n_conv_blocks must be 1, 2, or 3, got {n_conv_blocks}")

        self.n_conv_blocks = n_conv_blocks
        self.activation = self._get_activation(activation_fn)

        # First convolutional block. This always runs and expects 19 input channels.
        self.conv1 = nn.Conv1d(
            in_channels=19,
            out_channels=conv1_hidden_channels,
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding="same"
        )
        self.bn1 = nn.BatchNorm1d(conv1_hidden_channels)
        self.dropout1 = nn.Dropout(conv_dropout)
        self.pool1 = nn.MaxPool1d(maxpool_ks)

        # Second convolutional block. This is used only if n_conv_blocks >= 2.
        self.conv2 = nn.Conv1d(
            in_channels=conv1_hidden_channels,
            out_channels=conv2_hidden_channels,
            kernel_size=conv2_kernel_size,
            stride=conv2_stride,
            padding="same"
        )
        self.bn2 = nn.BatchNorm1d(conv2_hidden_channels)
        self.dropout2 = nn.Dropout(conv_dropout)
        self.pool2 = nn.MaxPool1d(maxpool_ks)

        # Third convolutional block. This is used only if n_conv_blocks >= 3.
        self.conv3 = nn.Conv1d(
            in_channels=conv2_hidden_channels,
            out_channels=conv3_output_channels,
            kernel_size=conv3_kernel_size,
            stride=conv3_stride,
            padding="same"
        )
        self.bn3 = nn.BatchNorm1d(conv3_output_channels)
        self.dropout3 = nn.Dropout(conv_dropout)
        self.pool3 = nn.MaxPool1d(maxpool_ks)

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

    def _forward_block(self, x, conv, bn, pool, dropout):
        """Run one convolutional block."""
        x = conv(x)
        x = bn(x)
        x = self.activation(x)
        x = pool(x)
        x = dropout(x)
        return x

    def forward(self, x):
        """
        Pass the input through the active convolutional blocks and return the
        final feature maps.
        """
        x = self._forward_block(x, self.conv1, self.bn1, self.pool1, self.dropout1)

        if self.n_conv_blocks >= 2:
            x = self._forward_block(x, self.conv2, self.bn2, self.pool2, self.dropout2)

        if self.n_conv_blocks >= 3:
            x = self._forward_block(x, self.conv3, self.bn3, self.pool3, self.dropout3)

        return x

    @property
    def output_channels(self):
        """
        Return the number of output channels produced by the last active
        convolutional block.
        """
        if self.n_conv_blocks == 1:
            return self.conv1.out_channels
        elif self.n_conv_blocks == 2:
            return self.conv2.out_channels
        elif self.n_conv_blocks == 3:
            return self.conv3.out_channels
        else:
            raise ValueError(f"Invalid n_conv_blocks: {self.n_conv_blocks}")