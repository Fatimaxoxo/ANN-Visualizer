import numpy as np


def run_perceptron_forward(inputs, weights, bias, activation="sigmoid"):
    """Compute a perceptron forward pass manually with visible intermediate steps."""
    x = np.array(inputs, dtype=float)
    w = np.array(weights, dtype=float)

    products = x * w
    weighted_sum = float(np.sum(products) + bias)

    if activation == "sigmoid":
        activation_value = 1.0 / (1.0 + np.exp(-weighted_sum))
    elif activation == "relu":
        activation_value = max(weighted_sum, 0.0)
    elif activation == "tanh":
        activation_value = np.tanh(weighted_sum)
    elif activation == "linear":
        activation_value = weighted_sum
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    return {
        "inputs": x.tolist(),
        "weights": w.tolist(),
        "bias": float(bias),
        "products": products.tolist(),
        "weighted_sum": weighted_sum,
        "activation": activation,
        "activation_value": float(activation_value),
    }
