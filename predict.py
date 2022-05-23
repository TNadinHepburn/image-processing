import matplotlib.pyplot as plt, numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from math import sqrt, ceil

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
target_size = (64,64)
test_data_dir = 'asl_alphabet_test'
my_model = keras.models.load_model("asl_model.h5")
img_for_pred = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_data_dir, target_size=target_size, batch_size=1, shuffle=False)

def plotImages(images):
    subplot_size_x = ceil(sqrt(len(images)))
    if subplot_size_x > 5:
        subplot_size_y = ceil(len(images)/5)
        subplot_size_x = 5
    else:
        subplot_size_y = subplot_size_x
    if subplot_size_x == 1 and subplot_size_y == 1:
        plt.imshow(images[0])
    else:
        fig, axes = plt.subplots(5,5,figsize=(64,64))
        axes = axes.flatten()
        for img, ax in zip(images, axes):
            ax.imshow(img)
            ax.axis('off')
    plt.tight_layout()
    plt.show()


predictions = my_model.predict(img_for_pred, verbose=0)
np.round(predictions)
print("PREDICTIONS")
print(predictions)
print("ARGMAX PREDICTIONS")
print(np.argmax(predictions, axis=-1))
count = 0
for result in np.argmax(predictions, axis=-1):
    print(labels[result])
    print(img_for_pred[count][1])
    plotImages(img_for_pred[count][0])
    count +=1