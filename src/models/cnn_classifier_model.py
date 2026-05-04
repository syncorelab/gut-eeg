"""
Combine the CNN feature extractor and classifier head into one full model
for 3-class emotion classification.
"""

import torch.nn as nn
from .cnn_model import cnnNet
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
        class_dropout,
        linear1_hidden_size,
        activation_fn,
        n_conv_blocks,
    ):
        """
        Build the full CNN-based classifier model.

        The model consists of:
        1. A CNN feature extractor
        2. Global average pooling over time
        3. A classifier head that outputs 3 class logits
        """
        super().__init__()

        # CNN backbone that extracts temporal features from the input signal.
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

        # Classifier head that adapts automatically to the CNN output size.
        self.head = Classifier(
            in_feature=self.cnn.output_channels,
            class_dropout=class_dropout,
            linear1_hidden_size=linear1_hidden_size,
            activation_fn=activation_fn,
        )

    def forward(self, x):
        """
        Run the input through the CNN, reduce the time dimension with global
        average pooling, and return class logits.
        """
        x = self.cnn(x)
        x = x.mean(dim=-1)
        x = self.head(x)
        return x