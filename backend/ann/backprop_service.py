import numpy as np


def run_backpropagation(inputs, weights_output, weights_hidden, target, output_activation, hidden_activation, learning_rate):
    """Compute a simple 1-hidden-layer backpropagation flow manually for display."""
    x = np.array(inputs, dtype=float)
    w_out = np.array(weights_output, dtype=float)
    w_hidden = np.array(weights_hidden, dtype=float)

    # Hidden layer pre-activation and activation
    hidden_linear = np.dot(x, w_hidden)
    hidden_activation_value = 1.0 / (1.0 + np.exp(-hidden_linear))

    # Output pre-activation and activation
    output_linear = float(np.dot(hidden_activation_value, w_out))
    output_activation_value = 1.0 / (1.0 + np.exp(-output_linear))

    # Error and delta at the output
    error = float(output_activation_value - target)
    delta_output = error * output_activation_value * (1 - output_activation_value)

    # Gradient and weight update for the output layer
    gradients_output = [delta_output * hidden_activation_value[0], delta_output * hidden_activation_value[1]]
    updated_output_weights = [w_out[0] - learning_rate * gradients_output[0], w_out[1] - learning_rate * gradients_output[1]]

    # Hidden delta using the output delta and output weight
    hidden_delta = [delta_output * w_out[0] * hidden_activation_value[0] * (1 - hidden_activation_value[0]),
                    delta_output * w_out[1] * hidden_activation_value[1] * (1 - hidden_activation_value[1])]

    # Hidden weight updates
    gradients_hidden = [hidden_delta[0] * x[0], hidden_delta[0] * x[1], hidden_delta[1] * x[0], hidden_delta[1] * x[1]]
    updated_hidden_weights = [w_hidden[0] - learning_rate * gradients_hidden[0],
                              w_hidden[1] - learning_rate * gradients_hidden[1],
                              w_hidden[2] - learning_rate * gradients_hidden[2],
                              w_hidden[3] - learning_rate * gradients_hidden[3]]

    return {
        "inputs": x.tolist(),
        "hidden_linear": hidden_linear.tolist(),
        "hidden_activation": hidden_activation_value.tolist(),
        "output_linear": output_linear,
        "output_activation": output_activation_value,
        "target": target,
        "error": error,
        "delta_output": delta_output,
        "gradients_output": gradients_output,
        "updated_output_weights": updated_output_weights,
        "hidden_delta": hidden_delta,
        "updated_hidden_weights": updated_hidden_weights,
        "learning_rate": learning_rate,
    }
