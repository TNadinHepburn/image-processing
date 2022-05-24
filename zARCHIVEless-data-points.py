import math, matplotlib.pyplot as plt, csv
from random import seed
from random import random

weights=[[[],[],[],[],[],[]],[[],[],[],[],[],[]]] # [[[h1],[h2],[h3],[h4],[h5],[h6],[h7],[h8],[h9],[h10],[h11],[h12],[h13],[h14],[h15],[h16]],[[o1],[o2],[o3],[o4],[o5]]]
data = [] # [x1,x2,x3,xN,n63,x0]
testData = [[0.463518441,0.92686379,6.35E-07,0.321434617,0.859846652,-0.082073823,0.39715752,0.620985806,-0.190377191,0.322028905,0.553306043,-0.03673353,0.352727294,0.237372875,-0.159110054,0.404726654,0.539621592,-0.03239233,0.422845721,0.185251266,-0.144762367,0.476121187,0.554035783,-0.040553235,0.491699576,0.221878469,-0.137410372,0.555575788,0.596967518,-0.058512568,0.554998875,0.328116417,-0.128209844,1],
    [0.303500056,0.723771274,-8.19E-07,0.252692401,0.596981764,-0.050183892,0.521591425,0.438717365,-0.105741531,0.356195271,0.394687504,0.016557427,0.380482048,0.458758861,-0.105698988,0.435125649,0.440335691,0.022459725,0.417961776,0.524879634,-0.082208768,0.49788034,0.485852599,0.014666094,0.439195901,0.568788111,-0.022849988,0.554215074,0.538358808,0.001447252,0.738425374,0.354242861,-0.01077427,1]] # [x1,x2,x3,x0]
hiddenLayer = [0, 0, 0, 0, 0, 0, 1] # [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, x0]  
outputLayer = [0,0,0,0,0,0] # [o1, o2, o3, o4, o5]
target = []
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