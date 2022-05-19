import math, matplotlib.pyplot as plt, csv
from random import seed
from random import random

weights=[[[],[],[],[],[],[]],[[],[],[],[],[],[]]] # [[[h1],[h2],[h3],[h4],[h5],[h6],[h7],[h8],[h9],[h10],[h11],[h12],[h13],[h14],[h15],[h16]],[[o1],[o2],[o3],[o4],[o5]]]
data = [] # [x1,x2,x3,xN,n63,x0]
testData = [[0.494033307,0.482931852,-8.24E-07,0.417855054,0.417343855,-0.014167378,0.369764924,0.175123915,-0.042198487,0.430588692,0.253838867,0.006179879,0.426681966,0.313921362,-0.075941093,0.490547448,0.269097328,-0.003988515,0.47149691,0.358890772,-0.063531183,0.547154665,0.292188585,-0.0200752,0.517734945,0.391214907,-0.038846064,0.600146115,0.320904166,-0.03716559,0.567856252,0.373509467,-0.03685914,1],
        [0.64219147,0.853945792,6.62E-07,0.577955127,0.803323746,-0.029385868,0.70032084,0.671527803,-0.065868452,0.560148358,0.609077573,-0.02196159,0.597676575,0.390106201,-0.084746793,0.614846587,0.600703537,-0.027069857,0.647533,0.364271164,-0.078483686,0.665277779,0.614367604,-0.036569458,0.695524514,0.398637861,-0.082342915,0.713083386,0.643980682,-0.050152525,0.733919561,0.46935761,-0.081936464,1],
        []] # [x1,x2,x3,x0]
hiddenLayer = [0, 0, 0, 0, 0, 0, 1] # [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, x0]  
outputLayer = [0,0,0,0,0,0] # [o1, o2, o3, o4, o5]
target = [0,1]
learning_rate = 0.05
epochs = 10
error=[]

def oneEpoch(numEpoch):
    for i in range(numEpoch): 
        train()
        print(f'>epoch={i}, lrate={learning_rate}, error={error[-1]}')
        
def train():
    squDiff = 0
    for i in range (len(data)):
        forwardStep(data[i])
        backwardStep(i)
        # calculates squared error for current data
        for j in range(len(target[i])):
            squDiff += pow((target[i][j] - outputLayer[j]),2)
    # divide total squared error by total data sets
    squDiff = squDiff/len(data)
    # adds calculated error to array for displaying error
    error.append(squDiff/2)


def test():
    resultSM = []
    temp = []
    outRoundedRes = []
    # calculates outputs for each test data set
    for i in range(len(testData)):
        unroundedResults = forwardStep(testData[i])
        print(f"UNRUONDED   {unroundedResults}")
        # calculates softmax for each output value 
        for j in range(len(outputLayer)):
            denominator = 0
            for k in range(len(unroundedResults)):
                denominator += math.exp(unroundedResults[k])
            softMax = math.exp(unroundedResults[j])/denominator
            temp.append(softMax)
        resultSM.append(temp)
        outRoundedRes.append([round(i) for i in unroundedResults])
    # rounds results for outputting
    #roundedRes = [round(i) for i in unroundedResults]
    # outputs results 
    print("Test Input: " + str(testData))
    print("Results: " + str(outRoundedRes))
    print("Probability Distribution: " + str(resultSM))
    

def forwardStep(currentData):
    # updates values in hidden layer nodes (sigmoid activation function)
    for i in range(len(hiddenLayer)-1):
        net = 0
        for j in range(len(currentData)):
            # print(f"{i}")
            # print(f"{j}")
            net += currentData[j] * weights[0][i][j]
        hiddenLayer[i] = 1/(1+math.exp(-net))
    # updates values in output layer nodes (no activation function)
    for i in range(len(outputLayer)):
        # net = sum ( hidden layer )
        net = 0
        for j in range(len(hiddenLayer)):
            net += hiddenLayer[j] * weights[1][i][j]
        outputLayer[i] = net
    return outputLayer

def backwardStep(dataIndex):    
    # OUTPUT LAYER weights calculation 
    errorO = [0,0,0,0,0,0]
    # for each output value calculate the error (target value - output value)
    for i in range(len(target[dataIndex])):
        errorO[i] = target[dataIndex][i]-outputLayer[i] 

    # HIDDENLAYER weights calculation
    errorH = [0,0,0,0,0,0]
    # for each hidden layer node calculate error (hidden node value * (1 - hidden node value) * [weight i * output value i ])
    for i in range(len(hiddenLayer)-1):
        errorH[i] += weights[1][0][i] * errorO[0] + weights[1][1][i] * errorO[1]
        errorH[i] *= (hiddenLayer[i]*(1-hiddenLayer[i]))
    # update weights using calculated error values 
    # ouput layer weights (learning rate * output error * hidden layer value)
    for i in range(len(weights[1])):
        for j in range(len(weights[1][i])):
            weights[1][i][j] += learning_rate * errorO[i] * hiddenLayer[j]
    # hidden layer weights (learning rate * hidden error * input value)
    for i in range(len(weights[0])):
        for j in range(len(weights[0][i])):
            weights[0][i][j] += learning_rate*errorH[i]*data[dataIndex][j]
    

def initialize():
    for i in range(len(weights[0])):
        for j in range(34):
            weights[0][i].append(random()/10) 
    for i in range(len(weights[1])):
        for j in range(7):
            weights[1][i].append(random()/10) 
    with open('landmark_data_less.csv') as file_data:
        reader_obj = csv.DictReader(file_data)
        counter = 0
        for row in reader_obj:
            counter += 1
            if not row or counter % 500:
                continue
            temp = []

            for column in row:
                temp.append(float(row[column].strip()))
            expected_output = bin(int(temp.pop(-1))).replace("0b", "")
            output_list = []
            while len(expected_output) < 6:
                expected_output = "0" + expected_output
            target.append([])
            for i in range(6):
                output_list.append(int(expected_output[i]))
            target[-1] = (output_list)
            temp.append(float(1))
            data.append(temp)




def plotLearningCurve():
    x_data = []
    y_data = []
    x_data.extend([i for i in range(0,len(error))])
    y_data.extend([error[i] for i in range(0,len(error))])
    fig, ax = plt.subplots()
    ax.set(xlabel="Epoch", ylabel="Squared Error")
    ax.plot(x_data,y_data, 'tab:green')
    plt.show()

seed(1)
initialize()
oneEpoch(epochs)
test()
plotLearningCurve()
print(error)