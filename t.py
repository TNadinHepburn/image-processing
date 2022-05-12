import tensorflow as tf, numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# ensure consistency across runs
from numpy.random import seed
seed(1)
tf.random.set_seed(2)
# Imports to view data
import cv2
from glob import glob
from matplotlib import pyplot as plt
from numpy import floor
import random
from keras.models import load_model

def plot_three_samples(letter):
    print("Samples images for letter " + letter)
    base_path = 'C:\\Users\\Computing\\Desktop\\archivezipped\\archive\\asl_alphabet_train\\'
    img_path = base_path + letter + '\\**'
    path_contents = glob(img_path)
    print(img_path)

    plt.figure(figsize=(16,16))
    imgs = random.sample(path_contents, 3)
    plt.subplot(131)
    plt.imshow(cv2.imread(imgs[0]))
    plt.subplot(132)
    plt.imshow(cv2.imread(imgs[1]))
    plt.subplot(133)
    plt.imshow(cv2.imread(imgs[2]))

plot_three_samples('A')

data_dir = "C:\\Users\\Computing\\Desktop\\archivezipped\\archive\\asl_alphabet_train\\"
target_size = (64, 64)
target_dims = (64, 64, 3) # add channel for RGB
n_classes = 26
val_frac = 0.1
batch_size = 64

data_augmentor = ImageDataGenerator(samplewise_center=True, 
                                    samplewise_std_normalization=True, 
                                    validation_split=val_frac)

train_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, shuffle=True, subset="training")
val_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, subset="validation")

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

my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

#my_model.fit_generator(train_generator, epochs=3, validation_data=val_generator)

my_model.save('ASL.h5')

testData = data_augmentor.flow_from_directory("C:\\Users\\Computing\\Desktop\\archivezipped\\archive\\asl_alphabet_test\\",target_size=target_size, batch_size=batch_size)

loadedmodel = load_model("..\\image-processing\\ASL.h5")
predict = loadedmodel.predict(testData)
print(predict)
