import numpy as np
from dataset import Dataset
from layers import Dense, Sigmoid, ReLU
from loss import BinaryCrossEntropy
from model import NeuralNetwork
from utils import *

# Create a dataset object and load data
data = Dataset("datasets").load().preprocess()

X_train, y_train = data.get_train_data()
X_test, y_test = data.get_test_data()

print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

# Build the model
model = NeuralNetwork()
model.add(Dense(input_size=X_train.shape[0], output_size=4))
model.add(ReLU())
model.add(Dense(input_size=4, output_size=1))
model.add(Sigmoid())    

# Show architecture
ModelSummary.summary(model, input_shape=(X_train.shape[0], X_train.shape[1]))

# Set the loss function
model.use(BinaryCrossEntropy())

# Train the model
model.train(X_train, y_train, epochs=5000, learning_rate=0.01, verbose=100)

# Evaluate the model
train_acc, train_loss = model.evaluate(X_train, y_train)
test_acc, test_loss = model.evaluate(X_test, y_test)
print(f"\nTraining Accuracy: {train_acc:.2f}%, Loss: {train_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%, Loss: {test_loss:.4f}")

# # Save the model
# save_model(model, "models/dnn_oops_model.pkl")
