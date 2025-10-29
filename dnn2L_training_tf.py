import tensorflow as tf
import numpy as np
from tensorflow import keras

# reference the function via tf.keras to avoid static resolver issues
image_dataset_from_directory = tf.keras.utils.image_dataset_from_directory

print("TensorFlow version:", tf.__version__)