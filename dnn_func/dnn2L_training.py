import numpy as np
from dnn_functions import *

# Load the dataset
train_set_x, train_y, test_set_x, test_y = load_dataset()

# Plot sample images from the dataset
# plot_image(train_set_x, train_y)

print(f"\nTraining set: {train_set_x.shape} {train_y.shape}")

X_train, y_train = reshape_and_normalize(train_set_x, train_y)

print(f"\nTraining set: {X_train.shape} {y_train.shape}")

nx = X_train.shape[0]
m_train = y_train.shape[1]

network_model = [nx, 4, 1]

parameters = {}

parameters["W1"], parameters["b1"] = initialize_parameters_rnd(network_model[1], network_model[0], 1).values()
parameters["W2"], parameters["b2"] = initialize_parameters_rnd(network_model[2], network_model[1], 2).values()
print(parameters["W2"], parameters["b2"])

activations = ["tanh", "sigmoid"]

learning_rate=0.01
max_iteration=2500
A0 = X_train

cost=np.zeros((max_iteration))

for i in range(max_iteration):

  # Forward Step : Hidden Layer
  Z1 = forward_linear(A0, parameters["W1"], parameters["b1"])
  A1 = forward_activation(Z1, activations[0])

  # Forward Step : Output Layer
  Z2 = forward_linear(A1, parameters["W2"], parameters["b2"])
  A2 = forward_activation(Z2, activations[1])

  # Calculate Cost
  cost[i] = compute_cost(A2, y_train)
  
  # Cost Derivatives : Output Layer
  dA2 = - (np.divide(y_train, A2) - np.divide(1 - y_train, 1 - A2))
  
  dZ2 = backward_activation(dA2, Z2, activations[1])
  dA1, dW2, db2 = backward_linear(dZ2, A1, parameters["W2"])

  # Cost Derivatives : Hidden Layer
  dZ1 = backward_activation(dA1, Z1, activations[0])  
  dA0, dW1, db1 = backward_linear(dZ1, X_train, parameters["W1"])

  #Update Parameters
  gradients = {"dW1" : dW1, "db1" : db1, "dW2" : dW2, "db2" : db2}
  parameters = update_parameters(parameters, gradients, learning_rate)

  # print(f"Parameters : {parameters}")
  if i%100==0:
    print("The cost after ", i , " iteration.", cost[i])

# plot_cost(cost, learning_rate)

# Save the trained model
save_model("dnn_func/trained_model.pkl", parameters, network_model, activations)