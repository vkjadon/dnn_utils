import numpy as np
from metrics import accuracy_score

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss):
        self.loss = loss

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_gradient, learning_rate):
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, learning_rate)

    def predict(self, x):
        output = self.forward(x)
        return (output > 0.5).astype(int)

    def evaluate(self, x, y):
        preds = self.predict(x)
        acc = accuracy_score(y, preds)
        loss_val = self.loss.forward(y, self.forward(x))
        return acc, loss_val

    def train(self, x_train, y_train, epochs, learning_rate, verbose=100):
        for epoch in range(epochs):
            output = self.forward(x_train)
            loss_value = self.loss.forward(y_train, output)
            grad = self.loss.backward(y_train, output)
            self.backward(grad, learning_rate)

            if (epoch + 1) % verbose == 0 or epoch == 0:
                acc = accuracy_score(y_train, (output > 0.5).astype(int))
                print(f"Epoch {epoch+1}/{epochs} | Loss: {loss_value:.4f} | Accuracy: {acc:.2f}%")
