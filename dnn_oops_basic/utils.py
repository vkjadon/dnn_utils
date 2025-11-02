import pickle
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

def save_model(filename, network):
    with open(filename, 'wb') as f:
        pickle.dump(network, f)
    print(f"Model saved to {filename}")

def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model

def summary(model, input_shape):
    """
    Display model architecture details:
    layer name, input/output shape, and parameter count.
    """
    print("\n MODEL SUMMARY")
    print("=" * 60)

    x = np.random.randn(*input_shape)  # dummy input
    total_params = 0
    table = []

    for idx, layer in enumerate(model):
        layer_name = layer.__class__.__name__

        # Forward pass through this layer to get output shape
        x = layer.forward(x)
        output_shape = x.shape

        # Calculate number of trainable parameters if any
        params = 0
        if hasattr(layer, 'weights'):
            params += layer.weights.size
        if hasattr(layer, 'bias'):
            params += layer.bias.size
        total_params += params

        table.append([idx + 1, layer_name, input_shape, output_shape, params])
        input_shape = output_shape  # next layer input shape = this layer output

    print(tabulate(table, headers=["#", "Layer", "Input Shape", "Output Shape", "Params"]))
    print("=" * 60)
    print(f"Total trainable parameters: {total_params:,}")
    print("=" * 60)

def plot_cost(costs, learning_rate):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title(f"Learning rate = {learning_rate}")
    plt.show()