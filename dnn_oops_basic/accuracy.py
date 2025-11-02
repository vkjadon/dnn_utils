import numpy as np
from  utils import load_model

network = load_model("dnn_oops_basic/trained_model.pkl")

print("\nEvaluating loaded model on test data...")

# Create a dataset object and load data
from dataset import Dataset
data = Dataset("datasets").load().preprocess()

X_train, y_train = data.get_train_data()
print(f"Training set: {X_train.shape}, {y_train.shape}")

output = X_train
for layer in network:
    output = layer.forward(output)
preds = (output > 0.5).astype(int)
accuracy = np.mean(preds == y_train) * 100
print(f"Training Accuracy of loaded model: {accuracy:.2f}%")

X_test, y_test = data.get_test_data()   
print(f"Test set: {X_test.shape}, {y_test.shape}")

# Forward Computation
output = X_test
for layer in network:
    output = layer.forward(output)
preds = (output > 0.5).astype(int)
accuracy = np.mean(preds == y_test) * 100
print(f"Test Accuracy of loaded model: {accuracy:.2f}%")

