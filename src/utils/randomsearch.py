import random
import math


def sample_from_spec(spec: dict):
    d = spec["dist"]
    if d == "categorical":
        return random.choice(spec["values"])
    if d == "uniform":
        return random.uniform(spec["low"], spec["high"])
    if d == "loguniform":
        low, high = spec["low"], spec["high"]
        return 10 ** random.uniform(math.log10(low), math.log10(high))
    raise ValueError(f"Unknown dist: {d}")


def sample_hparams(hparam_dists: dict) -> dict:
    sampled = {}
    for section_name, section in hparam_dists.items():
        sampled[section_name] = {}
        for k, spec in section.items():
            sampled[section_name][k] = sample_from_spec(spec)
    return sampled


def is_valid_config(cfg: dict) -> tuple[bool, str]:
    m = cfg["MODEL_HPARAMS"]
    d = cfg["DATA_HPARAMS"]
    t = cfg["TRAINING_HPARAMS"]

    n_blocks = m["n_conv_blocks"]

    # Only enforce monotonic channels for the blocks that are actually used
    if n_blocks == 1:
        pass
    elif n_blocks == 2:
        if not (m["conv1_hidden_channels"] <= m["conv2_hidden_channels"]):
            return False, "channels_not_monotonic_for_2_blocks"
    elif n_blocks == 3:
        if not (
            m["conv1_hidden_channels"]
            <= m["conv2_hidden_channels"]
            <= m["conv3_output_channels"]
        ):
            return False, "channels_not_monotonic_for_3_blocks"
    else:
        return False, "invalid_n_conv_blocks"

    return True, "all_valid"