import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

import glob, itertools, os, random, shutil

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
validation_percent = 0.1
batch_size = 50
target_size = (96, 96)
target_dims = (96, 96, 3) # add channel for RGB
n_classes = 26
epoch = 10
data_dir = 'D:/asl_alphabet_train'
test_data_dir = 'asl_alphabet_test'
data_augmentor = ImageDataGenerator(samplewise_center=True, 
                                    samplewise_std_normalization=True, 
                                    validation_split=validation_percent)

train_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, subset="training") 
val_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, subset="validation")

def expandTestSet(numToExpand):
    for label in labels:
        for c in random.sample(glob.glob('asl_alphabet_train/'+label+'/*'),numToExpand):
            shutil.move(c, 'asl_alphabet_test/'+label)

def createModel():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=target_dims))
    model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
    model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))
    model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    return model

def trainModel(model):
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit(x=train_generator, validation_data=val_generator, shuffle=True, epochs=epoch, verbose=1)

def saveModel(model):
    model.save('asl_model.h5')
