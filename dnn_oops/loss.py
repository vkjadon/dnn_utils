import numpy as np

class BinaryCrossEntropy:
    def forward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred)) / y_true.shape[1]
