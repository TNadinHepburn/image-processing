import matplotlib.pyplot as plt, numpy as np, matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import glob
from math import sqrt, ceil

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
target_size = (64,64)

def loadModel():
    return load_model("asl_model.h5")

def plotImages(images,result):
    subplot_size_x = ceil(sqrt(len(images)))
    if subplot_size_x > 5:
        subplot_size_y = ceil(len(images)/5)
        subplot_size_x = 5
    else: 
        subplot_size_y = subplot_size_x
    if subplot_size_y == 1 and subplot_size_x == 1:
        subplot_size_x +=1
    fig, axes = plt.subplots(subplot_size_x,subplot_size_y,figsize=(64,64))
    axes = axes.flatten()
    count = 0
    for img, ax in zip(images, axes):
        ax.imshow(img)
        if isinstance(result,list):
            ax.set_title(result[count])
            count +=1
        else:
            ax.set_title(result)
    for ax in axes:
        ax.axis('off')
    plt.show()

def predictTestImages(ASL_model):
    data_dir = 'asl_alphabet_test'
    img_for_pred = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=data_dir, target_size=target_size, batch_size=1, shuffle=False)
    predictions = ASL_model.predict(img_for_pred, verbose=0)
    np.round(predictions)
    count = 0
    pre_images = []
    for img in glob.glob(data_dir + "/*/*.jpg"):
        pre_images.append(mpimg.imread(img))
    all_labels = []
    for result in np.argmax(predictions, axis=-1):
        all_labels.append(labels[result])
        count += 1
    return pre_images, all_labels

def predictFileImage(ASL_model, filepath):
    data_dir = filepath
    img_for_pred = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=data_dir, target_size=target_size, batch_size=1, shuffle=False)
    predictions = ASL_model.predict(img_for_pred, verbose=0)
    np.round(predictions)
    count = 0
    pre_images = []
    for img in glob.glob(data_dir + "/*/*.jpg"):
        pre_images.append(mpimg.imread(img))
    all_labels = []
    for result in np.argmax(predictions, axis=-1):
        all_labels.append(labels[result])
        count += 1
    return pre_images,all_labels