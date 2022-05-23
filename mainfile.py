from genericpath import exists
from select import select
from network_model import *
from predict import *
from os.path import exists 

def program():
    option = 1
    while option == "1" or option == "2" or option == "3" or option == "0":
        print("Select option\n 1, Test Images   2, From Filepath   3, From Webam   0, Quit")
        option = input("-->")
        if option not in ["1","2","3","0"]:
            print("Invalid Choice")
            continue
        else:
            if option == "4":
                break
            elif option == "1":
                testImages()
            elif option == "2":
                fileImages()
            elif option == "3":
                webcamImages()

def testImages():
    return
    
def fileImages():
    return
    
def webcamImages():
    return
    



if __name__ == "__main__":
    print("Main")
    if not exists('asl_model.h5'):
        my_model = createModel()
        trainModel(my_model)
        saveModel(my_model)
    else:
        program()
else:
   print("File one executed when imported")