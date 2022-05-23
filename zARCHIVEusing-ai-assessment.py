import math, matplotlib.pyplot as plt, csv
from random import seed
from random import random

weights=[[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]],[[],[],[],[],[],[]]] # [[[h1],[h2],[h3],[h4],[h5],[h6],[h7],[h8],[h9],[h10],[h11],[h12],[h13],[h14],[h15],[h16]],[[o1],[o2],[o3],[o4],[o5]]]
data = [] # [x1,x2,x3,xN,n63,x0]
testData = [[0.32956916093826294,0.6961977481842041,-9.386293982061034e-07,0.23235349357128143,0.5888901948928833,-0.05306851491332054,0.16404414176940918,0.4737946689128876,-0.08913229405879974,0.14185485243797302,0.36720237135887146,-0.12674663960933685,0.14570802450180054,0.2801334857940674,-0.15400832891464233,0.3408844769001007,0.4107007384300232,-0.02964964136481285,0.39739078283309937,0.3535575270652771,-0.09393184632062912,0.36376646161079407,0.42467546463012695,-0.1290716826915741,0.3270830512046814,0.48549598455429077,-0.14122921228408813,0.4186210632324219,0.45483291149139404,-0.030552830547094345,0.46769315004348755,0.39362409710884094,-0.10843540728092194,0.4120967984199524,0.4730384349822998,-0.13876476883888245,0.36678287386894226,0.5279684662818909,-0.14375591278076172,0.4815126657485962,0.5004124045372009,-0.04225020110607147,0.5250647664070129,0.451424777507782,-0.12095769494771957,0.4607829749584198,0.5222423672676086,-0.12483332306146622,0.40810438990592957,0.5702273845672607,-0.10511675477027893,0.5324620008468628,0.550774335861206,-0.05729476362466812,0.6152440309524536,0.4790381193161011,-0.10233645141124725,0.668606698513031,0.4374459385871887,-0.1077684834599495,0.7157112956047058,0.38712671399116516,-0.10099152475595474,1],
    [0.6953368186950684,0.7704723477363586,-1.0707856290537165e-06,0.5425465106964111,0.6864079236984253,-0.050457362085580826,0.44692546129226685,0.5436028242111206,-0.07805132865905762,0.5106975436210632,0.4009968936443329,-0.11339282989501953,0.6282622814178467,0.33767688274383545,-0.14107801020145416,0.4824395477771759,0.42333608865737915,0.013118136674165726,0.4709431231021881,0.30790042877197266,-0.07122708112001419,0.4974766969680786,0.3793332874774933,-0.12926949560642242,0.5187520384788513,0.4599032402038574,-0.14835147559642792,0.583371639251709,0.4012635052204132,0.009766820818185806,0.5920369029045105,0.28462255001068115,-0.09707550704479218,0.6103554368019104,0.4017331302165985,-0.15217693150043488,0.6206409335136414,0.5076659917831421,-0.15561555325984955,0.679893970489502,0.3927503526210785,-0.007013979833573103,0.693289041519165,0.28784865140914917,-0.10214617103338242,0.6835176348686218,0.4157843589782715,-0.11677881330251694,0.672195315361023,0.5300747752189636,-0.08716253936290741,0.7754546999931335,0.39708399772644043,-0.028386801481246948,0.7597158551216125,0.25145620107650757,-0.06547188758850098,0.746045708656311,0.15490826964378357,-0.06693591177463531,0.735335648059845,0.068352609872818,-0.050519704818725586,1]] # [x1,x2,x3,x0]
hiddenLayer = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] # [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, x0]  
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
    errorH = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # for each hidden layer node calculate error (hidden node value * (1 - hidden node value) * [weight i * output value i ])
    for i in range(len(hiddenLayer)-1):
        errorH[i] += weights[1][0][i] * errorO[0] + weights[1][1][i] * errorO[1]
        errorH[i] *= (hiddenLayer[i]*(1-hiddenLayer[i]))
    print(f"errorH{errorH} {len(errorH)}")
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
        for j in range(64):
            weights[0][i].append(random()/10) 
    for i in range(len(weights[1])):
        for j in range(17):
            weights[1][i].append(random()/10) 
    with open('landmark_data.csv') as file_data:
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