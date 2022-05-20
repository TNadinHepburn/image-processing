import matplotlib
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, Activation

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import itertools

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

def plot_confusion_matrix(cm, classes,normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)

    if normalize:
        cm=cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized")
    else:
        print("Not Normalized")
    thresh =cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment="center",color="white" if cm[i,j]>thresh else "black")
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Pred")

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
val_frac = 0.1
batch_size = 50
target_size = (64, 64)
target_dims = (64, 64, 3) # add channel for RGB
n_classes = 26
epoch = 5
data_dir = 'asl_alphabet_train'
test_data_dir = 'asl_alphabet_test'
data_augmentor = ImageDataGenerator(samplewise_center=True, 
                                    samplewise_std_normalization=True, 
                                    validation_split=val_frac)

def expandTestSet(numToExpand):
    
    for label in labels:
        for c in random.sample(glob.glob('asl_alphabet_train/'+label+'/*'),numToExpand):
            shutil.move(c, 'asl_alphabet_test/'+label)

expandTestSet(650)

train_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, subset="training") 
val_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, subset="validation")
test_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_data_dir, target_size=target_size, batch_size=batch_size, shuffle=False)
print("Train")
print(train_generator)
print("Valid")
print(val_generator)
print("Test")
print(test_generator)
print(test_generator.class_indices)

img, label = next(train_generator)

# plotImages(img)
# print("-------------LABELS-------------")
# print(label)
# print("-------------LABELS-------------")


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

print("Model Summary")
print(my_model.summary())
print("Compiling")
my_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=["accuracy"])
print("Starting Fit")
my_model.fit(x=train_generator, validation_data=val_generator, shuffle=True, epochs=epoch, verbose=1)


test_img, test_label = next(test_generator)
plotImages(test_img)
print("Test Label")
print(test_label)
print("Test Classes")
print(test_generator.classes)

predictions = my_model.predict(test_generator, verbose=0)
np.round(predictions)
print("PREDICTIONS")
print(predictions)
print("ARGMAX PREDICTIONS")
print(np.argmax(predictions, axis=-1))

cm = confusion_matrix(y_true=test_generator.classes, y_pred=np.argmax(predictions, axis=-1))
cm_plot_labels = labels
print("matrix")
plot_confusion_matrix(cm=cm, classes=cm_plot_labels,title="Confusion")
print("done matrix\nsaved")

my_model.save('asl_model.h5')