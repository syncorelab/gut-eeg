"""
Configuration file for random hyperparameter search for the CNN classifier.

This file defines the distributions used to sample hyperparameters for
training runs. The parameters are grouped into:
1. Data hyperparameters
2. Model hyperparameters
3. Training hyperparameters

These distributions are mainly used together with the CNN classifier model
and the training loop.
"""

HPARAM_DISTS = {
    "DATA_HPARAMS": {
        # Batch size controls how many samples are processed before each
        # gradient update.
        "batch_size":  {"dist": "categorical", "values": [8, 16, 32, 64, 128]},
    },
    "MODEL_HPARAMS": {
        # Number of convolutional blocks included in the model.
        "n_conv_blocks": {"dist": "categorical", "values": [1, 2, 3]},

        # Output channel sizes for each convolutional block.
        "conv1_hidden_channels": {"dist": "categorical", "values": [8, 16, 32, 64]},
        "conv2_hidden_channels": {"dist": "categorical", "values": [16, 32, 64, 128]},
        "conv3_output_channels": {"dist": "categorical", "values": [32, 64, 128, 256]},

        # Kernel sizes determine how many time points each convolution sees
        # at once.
        "conv1_kernel_size": {"dist": "categorical", "values": [3, 5, 7, 9]},
        "conv2_kernel_size": {"dist": "categorical", "values": [3, 5, 7, 9]},
        "conv3_kernel_size": {"dist": "categorical", "values": [3, 5, 7, 9]},

        # Stride controls how far the convolution moves at each step.
        "conv1_stride": {"dist": "categorical", "values": [1]},
        "conv2_stride": {"dist": "categorical", "values": [1]},
        "conv3_stride": {"dist": "categorical", "values": [1]},

        # Max-pooling reduces temporal resolution and helps compress features.
        "maxpool_ks": {"dist": "categorical", "values": [1, 2, 3, 4]},

        # Dropout rates used for regularization in the convolutional and
        # classifier parts of the model.
        "conv_dropout":  {"dist": "uniform", "low": 0.2, "high": 0.5},
        "class_dropout": {"dist": "uniform", "low": 0.1, "high": 0.5},

        # Size of the hidden linear layer before the final classifier output.
        "linear1_hidden_size": {"dist": "categorical", "values": [64, 128, 256, 512]},

        # Activation function used in the model layers.
        "activation_fn": {"dist": "categorical", "values": ["relu", "gelu", "leakyrelu"]},
    },
    "TRAINING_HPARAMS": {
        # Early stopping can stop training when validation performance no
        # longer improves.
        "use_early_stopping": {"dist": "categorical", "values": [False]},
        "early_stopping_patience": {"dist": "categorical", "values": [10, 20, 30]},
        "early_stopping_min_delta": {"dist": "uniform", "low": 0.0001, "high": 0.01},

        # Optimizer and training settings.
        "learning_rate": {"dist": "loguniform", "low": 1e-5, "high": 3e-3},
        "num_epochs":    {"dist": "categorical", "values": [150, 200]},
        "betas":         {"dist": "categorical", "values": [(0.5, 0.999), (0.9, 0.999)]},
        "weight_decay":  {"dist": "loguniform", "low": 1e-5, "high": 1e-2},
    },
}

