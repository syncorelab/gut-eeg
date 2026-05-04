"""
Hyperparameter configuration for random search for a CNN + LSTM + Attention model.

This file defines sampling distributions for hyperparameters used during training.
The parameters are grouped into:
1. Data hyperparameters (how the input windows are constructed)
2. Model hyperparameters (architecture of CNN, LSTM, and attention components)
3. Training hyperparameters (optimizer and training behavior)

These distributions are used to generate different model configurations
during random search.
"""

HPARAM_DISTS = {
    "DATA_HPARAMS": {
        # Number of time points in each input window.
        "window_size": {"dist": "categorical", "values": [10240, 20480, 30720]},
        
        # Fraction of overlap between consecutive windows.
        "overlap":     {"dist": "categorical", "values": [0.0, 0.25, 0.5, 0.75]},
        
        # Number of samples per batch during training.
        "batch_size":  {"dist": "categorical", "values": [16, 32]},
    },

    "MODEL_HPARAMS": {
        # Number of convolutional blocks used for feature extraction.
        "n_conv_blocks": {"dist": "categorical", "values": [2, 3]},

        # Output channel sizes for each convolutional block.
        "conv1_hidden_channels": {"dist": "categorical", "values": [16, 32]},
        "conv2_hidden_channels": {"dist": "categorical", "values": [16, 32, 64]},
        "conv3_output_channels": {"dist": "categorical", "values": [32, 64, 128]},

        # Kernel sizes define how many time points each convolution covers.
        "conv1_kernel_size": {"dist": "categorical", "values": [3, 5, 7, 9]},
        "conv2_kernel_size": {"dist": "categorical", "values": [3, 5, 7, 9]},
        "conv3_kernel_size": {"dist": "categorical", "values": [3, 5, 7, 9]},

        # Stride controls how far the convolution moves along the time axis.
        "conv1_stride": {"dist": "categorical", "values": [1]},
        "conv2_stride": {"dist": "categorical", "values": [1]},
        "conv3_stride": {"dist": "categorical", "values": [1]},

        # Max-pooling reduces temporal resolution and compresses features.
        "maxpool_ks": {"dist": "categorical", "values": [1, 2, 3, 4]},

        # Dropout applied in convolutional layers for regularization.
        "conv_dropout":  {"dist": "uniform", "low": 0.2, "high": 0.5},

        # LSTM layer parameters for temporal modeling.
        "lstm_hidden_size": {"dist": "categorical", "values": [64, 128, 256, 512]},
        "lstm_num_layers": {"dist": "categorical", "values": [1, 2, 3]},
        "lstm_dropout": {"dist": "uniform", "low": 0.1, "high": 0.5},
        
        # Whether the LSTM processes the sequence in both directions.
        "bidirectional": {"dist": "categorical", "values": [True, False]},

        # Attention layer parameters for weighting important time steps.
        "attn_hidden_dim": {"dist": "categorical", "values": [32, 64, 128, 256]},
        "attn_dropout_attn": {"dist": "uniform", "low": 0.1, "high": 0.5},

        # Dropout in the classifier layers.
        "class_dropout": {"dist": "uniform", "low": 0.1, "high": 0.5},

        # Size of the hidden fully connected layer before output.
        "linear1_hidden_size": {"dist": "categorical", "values": [64, 128, 256, 512]},

        # Activation function used throughout the model.
        "activation_fn": {"dist": "categorical", "values": ["relu", "gelu", "leakyrelu"]},
    },

    "TRAINING_HPARAMS": {
        # Early stopping disabled for all runs in this configuration.
        "use_early_stopping": {"dist": "categorical", "values": [False]},
        
        # Early stopping parameters (unused if disabled).
        "early_stopping_patience": {"dist": "categorical", "values": [10, 20, 30]},
        "early_stopping_min_delta": {"dist": "uniform", "low": 0.0001, "high": 0.01},

        # Optimizer settings.
        "learning_rate": {"dist": "loguniform", "low": 1e-5, "high": 3e-3},
        "num_epochs":    {"dist": "categorical", "values": [150, 200]},
        "betas":         {"dist": "categorical", "values": [(0.5, 0.999), (0.9, 0.999)]},
        "weight_decay":  {"dist": "loguniform", "low": 1e-5, "high": 1e-2},
    },
}
