import numpy as np
from dnn_functions import *

def predict(X, parameters, activations):
    A = X
    L = len(parameters) // 2
    
    for l in range(L):
        W = parameters["W" + str(l + 1)]
        b = parameters["b" + str(l + 1)]
        
        Z = forward_linear(A, W, b)
        A = forward_activation(Z, activations[l])
    return A

def load_model(filename):
    with open(filename, "rb") as f:
        model_data = pickle.load(f)
    print(f"Model loaded from '{filename}'")
    return model_data["parameters"], model_data["network_model"], model_data["activations"]

if __name__ == "__main__":

# Load the dataset
    train_set_x, train_y, test_set_x, test_y = load_dataset()

    X_train, y_train = reshape_and_normalize(train_set_x, train_y)

    nx = X_train.shape[0]
    m_train = y_train.shape[1]

# Load model
    parameters, network_model, activations = load_model("trained_model.pkl")

# Training Accuracy
    A2 = predict(X_train, parameters, activations)
    y_pred = np.array([1 if pred > 0.5 else 0 for pred in A2[0]]).reshape(1, y_train.shape[1])
    print(f"Training Accuracy : {(np.sum(y_pred == y_train))/y_train.shape[1]}")

# Test Accuracy
    X_test, y_test = reshape_and_normalize(test_set_x, test_y)
    A2 = predict(X_test, parameters, activations)
    y_pred = np.array([1 if pred > 0.5 else 0 for pred in A2[0]]).reshape(1, y_test.shape[1])
    print(f"Training Accuracy : {(np.sum(y_pred == y_test))/y_test.shape[1]}")