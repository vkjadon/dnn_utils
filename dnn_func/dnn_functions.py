import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def plot_image(train_x, train_y):
    classes = ["non-cat", "cat"]

    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(train_x[i])
        plt.title(classes[int(train_y[0, i])])
        plt.axis("off")

    plt.suptitle("Sample Images from Cat vs Non-Cat Dataset", fontsize=14)
    plt.show()

def load_dataset(dataset_path = "datasets"):
    print(f"\nUsing dataset from: {dataset_path}")

    train_path = os.path.join(dataset_path, "train_catvnoncat.h5")
    test_path = os.path.join(dataset_path, "test_catvnoncat.h5")

    with h5py.File(train_path, "r") as train_dataset:
        train_x = np.array(train_dataset["train_set_x"][:])
        train_y = np.array(train_dataset["train_set_y"][:])

    with h5py.File(test_path, "r") as test_dataset:
        test_x = np.array(test_dataset["test_set_x"][:])
        test_y = np.array(test_dataset["test_set_y"][:])

    train_y = train_y.reshape((1, train_y.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))
    return train_x, train_y, test_x, test_y

def reshape_and_normalize(train_set_x, train_y):
    train_set_x=train_set_x.reshape(train_set_x.shape[0],-1).T
    X_train = train_set_x / 255.

    y_train = train_y.reshape((1, train_y.shape[1]))
    return X_train, y_train

def initialize_parameters_rnd(output_size, input_size, layer):
  random_state = 2
  rng = np.random.default_rng(random_state)

  parameters = {}
  W_matrix = rng.standard_normal((output_size, input_size)) * 0.01
  b_array = np.zeros((output_size, 1))

  Weight  = "W" + str(layer)
  bias  = "b" + str(layer)

  parameters = { Weight : W_matrix, bias : b_array }

  return parameters

def forward_linear(a, W, b):
  Z = np.dot(W, a) + b
  return Z

def forward_activation(z, activation="relu"):
  if activation=='relu':
    A = np.maximum(0,z)

  elif activation=='sigmoid':
    A = 1/(1+np.exp(-z))

  elif activation == "tanh":
    A = np.tanh(z)

  return A

def backward_activation(dA, forward_activation_input, activation = "relu"):
  if activation=='relu':
    dAdZ = [1 if z > 0 else 0 for z in forward_activation_input]
  elif activation=='sigmoid':
    s = 1/(1 + np.exp(-forward_activation_input))
    dAdZ = s * (1 - s)
  elif activation == "tanh":
    dAdZ = 1 - np.tanh(forward_activation_input) ** 2

  dZ = dA * dAdZ

  return dZ

def backward_linear(dJdZ, previousLayerA, layerW):
  m_train = dJdZ.shape[1]

  dA = np.dot(layerW.T, dJdZ)

  dW = (1 / m_train) * np.dot(dJdZ, previousLayerA.T)
  db = (1 / m_train) * np.sum(dJdZ, axis=1, keepdims=True)

  return dA, dW, db

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(L):
        Weight = "W" + str(l + 1)
        bias = "b" + str(l + 1)
    
        parameters[Weight] = parameters[Weight] - learning_rate * grads["dW" + str(l + 1)]
        parameters[bias] = parameters[bias] - learning_rate * grads["db" + str(l + 1)]
    
    return parameters

def compute_cost(A, Y):
    m = Y.shape[1]
    cost = - np.sum(np.multiply(np.log(A), Y) + np.multiply((1 - Y), np.log(1 - A))) / m
    cost = np.squeeze(cost)
    return cost

def plot_cost(costs, learning_rate):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title(f"Learning rate = {learning_rate}")
    plt.show()

def save_model(filename, parameters, network_model, activations):
    """Save trained model and parameters to a file."""
    model_data = {
        "parameters": parameters,
        "network_model": network_model,
        "activations": activations
    }
    with open(filename, "wb") as f:
        pickle.dump(model_data, f)
    print(f"Model saved successfully to '{filename}'")