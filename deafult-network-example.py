from math import exp
from random import seed
from random import random

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
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

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
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

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] -= l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

seed(1)
dataset = [[0.46226274967193604, 0.9314163327217102, 5.156756515134475e-07, 0.32102906703948975, 0.8690335750579834, -0.07793261855840683, 0.24366290867328644, 0.7515475153923035, -0.11654634773731232, 0.2979912757873535, 0.6468672752380371, -0.14985805749893188, 0.39187464118003845, 0.623251736164093, -0.1791105419397354, 0.32519495487213135, 0.5505350232124329, -0.029445800930261612, 0.325386106967926, 0.4003191888332367, -0.06686026602983475, 0.33738353848457336, 0.3117583990097046, -0.10946670919656754, 0.35294440388679504, 0.23573267459869385, -0.14561727643013, 0.40735289454460144, 0.5388861298561096, -0.02627790905535221, 0.40598493814468384, 0.3707590699195862, -0.05778143182396889, 0.4149753451347351, 0.26960375905036926, -0.09782474488019943, 0.42631837725639343, 0.18573495745658875, -0.13045066595077515, 0.4770398736000061, 0.5552796721458435, -0.03574608266353607, 0.4818314015865326, 0.39628124237060547, -0.06673657149076462, 0.48592931032180786, 0.29939866065979004, -0.1007314920425415, 0.49119555950164795, 0.22285771369934082, -0.12700650095939636, 0.5550819039344788, 0.601193368434906, -0.054694127291440964, 0.5563027262687683, 0.47325441241264343, -0.08398215472698212, 0.5544518828392029, 0.393740177154541, -0.10495159775018692, 0.5554479360580444, 0.32560425996780396, -0.1217215284705162,0.0385],[0.20712348818778992, 0.5157108902931213, 1.9960930330853444e-06, 0.22716300189495087, 0.6331173777580261, -0.25889885425567627, 0.2976991534233093, 0.7131827473640442, -0.37821635603904724, 0.3858698308467865, 0.7858003377914429, -0.42473602294921875, 0.4895453155040741, 0.8609329462051392, -0.4556249976158142, 0.4626978933811188, 0.4894869923591614, -0.4032337963581085, 0.6202471256256104, 0.6108436584472656, -0.45597976446151733, 0.6990780830383301, 0.7180361747741699, -0.44953596591949463, 0.7435925006866455, 0.7964105606079102, -0.4345115125179291, 0.4874453544616699, 0.4556501805782318, -0.2573411464691162, 0.628730833530426, 0.5996648669242859, -0.29152315855026245, 0.5573223233222961, 0.6562872529029846, -0.24703890085220337, 0.48980382084846497, 0.6453130841255188, -0.2099331021308899, 0.5092374086380005, 0.45353224873542786, -0.11616108566522598, 0.6241728663444519, 0.5876694917678833, -0.1561087965965271, 0.5409363508224487, 0.6363728046417236, -0.13435405492782593, 0.48096948862075806, 0.6218634247779846, -0.11342298984527588, 0.5299922823905945, 0.458607941865921, 0.010314625687897205, 0.5906164646148682, 0.5581280589103699, -0.030662113800644875, 0.5413848757743835, 0.6043282747268677, -0.04077991470694542, 0.4940238296985626, 0.5961668491363525, -0.04139241203665733,0.615]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
print(n_outputs)

network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 20, n_outputs)
for layer in network:
	print(layer)