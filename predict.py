import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

my_model = keras.models.load_model("my_model")
img_for_pred = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_data_dir, target_size=target_size, batch_size=batch_size, shuffle=False)




predictions = my_model.predict(img_for_pred, verbose=0)
np.round(predictions)
print("PREDICTIONS")
print(predictions)
print("ARGMAX PREDICTIONS")
print(np.argmax(predictions, axis=-1))
print(labels[np.argmax(predictions, axis=-1)])