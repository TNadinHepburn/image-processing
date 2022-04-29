from random import seed
from random import random
from math import exp
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_hidden2, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	hidden_layer2 = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_hidden2)]
	network.append(hidden_layer2)
	output_layer = [{'weights':[random() for i in range(n_hidden2 + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

def transfer_derivative(output):
	return output * (1.0 - output)

def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(neuron['output'] - expected[j])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] -= l_rate * neuron['delta']

def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


seed(1)
network = initialize_network(63,21,5,1)
data = [0.46226274967193604, 0.9314163327217102, 5.156756515134475e-07, 0.32102906703948975, 0.8690335750579834, -0.07793261855840683, 0.24366290867328644, 0.7515475153923035, -0.11654634773731232, 0.2979912757873535, 0.6468672752380371, -0.14985805749893188, 0.39187464118003845, 0.623251736164093, -0.1791105419397354, 0.32519495487213135, 0.5505350232124329, -0.029445800930261612, 0.325386106967926, 0.4003191888332367, -0.06686026602983475, 0.33738353848457336, 0.3117583990097046, -0.10946670919656754, 0.35294440388679504, 0.23573267459869385, -0.14561727643013, 0.40735289454460144, 0.5388861298561096, -0.02627790905535221, 0.40598493814468384, 0.3707590699195862, -0.05778143182396889, 0.4149753451347351, 0.26960375905036926, -0.09782474488019943, 0.42631837725639343, 0.18573495745658875, -0.13045066595077515, 0.4770398736000061, 0.5552796721458435, -0.03574608266353607, 0.4818314015865326, 0.39628124237060547, -0.06673657149076462, 0.48592931032180786, 0.29939866065979004, -0.1007314920425415, 0.49119555950164795, 0.22285771369934082, -0.12700650095939636, 0.5550819039344788, 0.601193368434906, -0.054694127291440964, 0.5563027262687683, 0.47325441241264343, -0.08398215472698212, 0.5544518828392029, 0.393740177154541, -0.10495159775018692, 0.5554479360580444, 0.32560425996780396, -0.1217215284705162, None]
expected = [1,0]
output = forward_propagate(network,data)
print(output)
backward_propagate_error(output,expected)
#print(network[0])

seed(1)
dataset = [[0.46226274967193604, 0.9314163327217102, 5.156756515134475e-07, 0.32102906703948975, 0.8690335750579834, -0.07793261855840683, 0.24366290867328644, 0.7515475153923035, -0.11654634773731232, 0.2979912757873535, 0.6468672752380371, -0.14985805749893188, 0.39187464118003845, 0.623251736164093, -0.1791105419397354, 0.32519495487213135, 0.5505350232124329, -0.029445800930261612, 0.325386106967926, 0.4003191888332367, -0.06686026602983475, 0.33738353848457336, 0.3117583990097046, -0.10946670919656754, 0.35294440388679504, 0.23573267459869385, -0.14561727643013, 0.40735289454460144, 0.5388861298561096, -0.02627790905535221, 0.40598493814468384, 0.3707590699195862, -0.05778143182396889, 0.4149753451347351, 0.26960375905036926, -0.09782474488019943, 0.42631837725639343, 0.18573495745658875, -0.13045066595077515, 0.4770398736000061, 0.5552796721458435, -0.03574608266353607, 0.4818314015865326, 0.39628124237060547, -0.06673657149076462, 0.48592931032180786, 0.29939866065979004, -0.1007314920425415, 0.49119555950164795, 0.22285771369934082, -0.12700650095939636, 0.5550819039344788, 0.601193368434906, -0.054694127291440964, 0.5563027262687683, 0.47325441241264343, -0.08398215472698212, 0.5544518828392029, 0.393740177154541, -0.10495159775018692, 0.5554479360580444, 0.32560425996780396, -0.1217215284705162, 1]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 21, 5, n_outputs)
train_network(network, dataset, 0.5, 20, n_outputs)
for layer in network:
	print(layer)


class neural_network(object):
    def __init__(self, n_inputs=63, n_hidden=21, n_hidden2=5, n_outputs=1):
        self.network = list()
        hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        self.network.append(hidden_layer)
        hidden_layer2 = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_hidden2)]
        self.network.append(hidden_layer2)
        output_layer = [{'weights':[random() for i in range(n_hidden2 + 1)]} for i in range(n_outputs)]
        self.network.append(output_layer)