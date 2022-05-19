import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, Activation

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam_v2
from keras.metrics import categorical_crossentropy
from keras.models import load_model
from sklearn.metrics import confusion_matrix

import os, shutil, random, glob, matplotlib.pyplot as plt, itertools, numpy as np

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0],True)


def plotImages(images):
    fig, axes = plt.subplots(8,8,figsize=(64,64))
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()








val_frac = 0.1
batch_size = 64
target_size = (64, 64)
data_dir = "C:\\Users\\Computing\\Desktop\\asl_alphabet_train\\"
test_data_dir = "C:\\Users\\Computing\\Desktop\\asl_alphabet_test\\"
data_augmentor = ImageDataGenerator(samplewise_center=True, 
                                    samplewise_std_normalization=True, 
                                    validation_split=val_frac)

train_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, shuffle=True, subset="training")
val_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, subset="validation")
test_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.process_input).flow_from_directory(directory = test_data_dir, target_size=target_size, batch_size=batch_size, shuffle=False)
print(train_generator)
print(val_generator)
print(test_generator)
img, label = next(train_generator)

plotImages(img)
print(label)



target_dims = (64, 64, 3) # add channel for RGB
n_classes = 26

my_model = Sequential()
my_model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=target_dims))
my_model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
my_model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))
my_model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))
my_model.add(Flatten())
my_model.add(Dropout(0.5))
my_model.add(Dense(512, activation='relu'))
my_model.add(Dense(n_classes, activation='softmax'))

print(my_model.summary())
my_model.compile(optimizer=adam_v2, loss='categorical_crossentropy', metrics=["accuracy"])
my_model.fit(train_generator, epochs=3, validation_data=val_generator, shuffle=True, verbose=2)


test_img, test_label = next(test_generator)
plotImages(test_img)
print(test_label)
print(test_generator.classes)

predictions = my_model.predict(test_generator, verbose=0)
np.round(predictions)
print(predictions)
