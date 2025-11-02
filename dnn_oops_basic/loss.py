import numpy as np

class BinaryCrossEntropy:
    def forward(y_true, y_pred):
        return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

    def backward(y_true, y_pred):
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

class MeanSquareError:
    def forward(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)