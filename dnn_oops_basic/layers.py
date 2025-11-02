import numpy as np

class Layer:
  def __init__(self):
    self.input = None
    self.output = None

  def forward(self, input):
    pass

  def backward(self, output_gradient, learning_rate):
    pass

class Dense(Layer):
  def __init__(self, input_size, output_size):
    super().__init__()
    random_state = 2
    rng = np.random.default_rng(random_state)
    self.weights = rng.standard_normal((output_size, input_size)) * 0.1
    self.bias = rng.standard_normal((output_size, 1)) * 0.0
    # self.bias = rng.standard_normal((output_size, 1))

  def forward(self, input):
    self.input = input
    return np.dot(self.weights, self.input) + self.bias

  def backward(self, output_gradient, learning_rate):
    weights_gradient = np.dot(output_gradient, self.input.T)
    input_gradient = np.dot(self.weights.T, output_gradient)
    bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)

    self.weights -= learning_rate * weights_gradient
    self.bias -= learning_rate * bias_gradient
    return input_gradient
  
class Sigmoid(Layer):
    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-self.input))
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.output * (1 - self.output))

class Tanh(Layer):
    def forward(self, input):
        self.input = input
        self.output = np.tanh(self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (1 - self.output ** 2)

class ReLU(Layer):
    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)
