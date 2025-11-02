import numpy as np
from dataset import Dataset
from layers import Dense, Sigmoid, ReLU, Tanh
import loss as loss
from utils import *

# Create a dataset object and load data
data = Dataset("datasets").load().preprocess()

X_train, y_train = data.get_train_data()
X_test, y_test = data.get_test_data()

print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

# Build the model
DL1=Dense(input_size=X_train.shape[0], output_size=20)
DL2=Dense(input_size=20, output_size=5)
ODL=Dense(input_size=5, output_size=1)

AL1=ReLU()
AL2=ReLU()
OAL=Sigmoid()
network = [DL1, AL1, DL2, AL2, ODL, OAL]

# Show architecture
summary(network, input_shape=(X_train.shape[0], X_train.shape[1]))

epochs = 3000        
learning_rate = 0.0075 
cost=np.zeros(epochs)

for epoch in range(epochs):

  # Forward Computation
  output = X_train
  for layer in network:
    output = layer.forward(output)

  # Loss and Grad Computation
  cost[epoch] = loss.BinaryCrossEntropy.forward(y_train, output)
  
  grad = loss.BinaryCrossEntropy.backward(y_train, output)

  # Backward Computation

  for layer in reversed(network):
    grad = layer.backward(grad, learning_rate)

  if epoch % 100 == 0:
    print(f"Epoch {epoch}, Loss = {cost[epoch]}")

output = X_train
for layer in network:
  output = layer.forward(output)
preds = (output > 0.5).astype(int)
accuracy = np.mean(preds == y_train) * 100
print(f"Training Accuracy of loaded model: {accuracy:.2f}%")

plot_cost(cost, learning_rate)

# Save the model
save_model("dnn_oops_basic/trained_model.pkl", network)