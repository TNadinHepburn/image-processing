import matplotlib.pyplot as plt, numpy as np, matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import glob
from math import sqrt, ceil

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
target_size = (64,64)

def loadModel():
    return keras.models.load_model("asl_model.h5")

def plotImages(images,result="Title"):
    subplot_size_x = ceil(sqrt(len(images)))+1
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
        count = 0
        for img, ax in zip(images, axes):
            ax.imshow(img)
            if isinstance(result,list):
                ax.set_title(result[count])
                count +=1
            else:
                ax.set_title(result)
            ax.axis('off')
    #plt.tight_layout()
    plt.show()

def testImages(ASL_model):
    test_data_dir = 'asl_alphabet_test'
    img_for_pred = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_data_dir, target_size=target_size, batch_size=1, shuffle=False)
    predictions = ASL_model.predict(img_for_pred, verbose=0)
    #np.round(predictions)
    print("PREDICTIONS")
    print(predictions)
    print("ARGMAX PREDICTIONS")
    print(np.argmax(predictions, axis=-1))
    np.round(predictions)
    count = 0
    pre_images = []
    for img in glob.glob(test_data_dir+"/Test/*.jpg"):
        pre_images.append(mpimg.imread(img))
    #plotImages(pre_images)
    all_labels = []
    for result in np.argmax(predictions, axis=-1):
        print(labels[result])
        all_labels.append(labels[result])
        print(img_for_pred[count][1])
        #plotImages(img_for_pred[count][0],labels[result])
        count += 1
    plotImages(pre_images,all_labels)

my_model = loadModel()
testImages(my_model)