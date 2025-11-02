import os
import h5py
import numpy as np

class Dataset:
    def __init__(self, dataset_path="datasets"):
        """
        Initializes the dataset loader with the given path.
        """
        self.dataset_path = dataset_path
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    def load(self):
        """
        Loads the dataset from HDF5 files.
        """
        print(f"\nUsing dataset from: {self.dataset_path}")

        train_path = os.path.join(self.dataset_path, "train_catvnoncat.h5")
        test_path = os.path.join(self.dataset_path, "test_catvnoncat.h5")

        with h5py.File(train_path, "r") as train_dataset:
            self.train_x = np.array(train_dataset["train_set_x"][:])
            self.train_y = np.array(train_dataset["train_set_y"][:])

        with h5py.File(test_path, "r") as test_dataset:
            self.test_x = np.array(test_dataset["test_set_x"][:])
            self.test_y = np.array(test_dataset["test_set_y"][:])

        # Reshape labels
        self.train_y = self.train_y.reshape((1, self.train_y.shape[0]))
        self.test_y = self.test_y.reshape((1, self.test_y.shape[0]))

        print("Dataset loaded successfully.")
        return self

    def preprocess(self):
        """
        Reshapes and normalizes the dataset.
        """
        # Flatten images
        train_x_flatten = self.train_x.reshape(self.train_x.shape[0], -1).T
        test_x_flatten = self.test_x.reshape(self.test_x.shape[0], -1).T

        # Normalize pixel values
        self.train_x = train_x_flatten / 255.
        self.test_x = test_x_flatten / 255.

        print("Dataset reshaped and normalized.")
        return self

    def get_train_data(self):
        """Returns the training data."""
        return self.train_x, self.train_y

    def get_test_data(self):
        """Returns the test data."""
        return self.test_x, self.test_y
