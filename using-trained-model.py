import numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator


test_data_dir = 'my_photo'
test_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_data_dir, target_size=(64,64), batch_size=2, shuffle=False)

my_model = keras.models.load_model('asl_model.h5', compile=True)

predictions = my_model.predict(test_generator)
np.round(predictions)
print("PREDICTIONS")
print(predictions)
print("ARGMAX PREDICTIONS")
print(np.argmax(predictions, axis=-1))

