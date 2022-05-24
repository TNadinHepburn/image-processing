#import functions from other main files
from network_model import *
from predict import *
from os.path import isfile,isdir 
from os import environ  
# fix for tensorflow bug where cant find .dll file 
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def program(model):
    option = "0"
    while option == "1" or option == "2" or option == "3" or option == "0":
        print("Select option\n 1 Test Images   2 From Filepath   3 From Webcam   0 Quit")
        option = input("-->")
        if option not in ["1","2","3","0"]:
            print("Invalid Choice")
            continue
        else:
            if option == "0":
                print("EXITING PROGRAM\nthank you for using our Sign Language Recognition System!")
                break
            elif option == "1":
                testImages(model)
            elif option == "2":
                fileImages(model)
            elif option == "3":
                webcamImages(model)

def testImages(model):
    img, outputs = predictTestImages(model)
    plotImages(img,outputs)

def fileImages(model):
    filepath = ""
    while not isdir(filepath):
        filepath = input("Enter the filepath of the image(s): ")
    img, outputs = predictFileImage(model,filepath)
    plotImages(img, outputs)

    
def webcamImages():
    print("Sorry this has not been implemented yet.\nPlease try another option!")
    return
    

if __name__ == "__main__":
    print(isfile('asl_model.h5'))
    if isfile('./asl_model.h5'):
        print("loading model")
        my_model = loadModel()
    else:
        my_model = createModel()
        trainModel(my_model)
        saveModel(my_model)
    program(my_model)
    