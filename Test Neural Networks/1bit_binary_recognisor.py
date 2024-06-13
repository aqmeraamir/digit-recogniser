# Import librariers
import numpy as np


LEARNING_RATE = 1

#-------------------------------------------
# Classes for neurons & layers
#-------------------------------------------
class Neuron:
   
    def __init__(self, weights_required):
        self.weights = np.random.uniform(-0.1, 0.1, weights_required)  # small random weights
        self.bias = np.random.uniform(-0.1, 0.1)  # small random bias

        self.weighted_sum = 0
        self.activation = 0

        self.weight_gradients = []
        self.bias_gradients = []
        
    def update_gradient(self, grad_w, grad_b):
        grad_w = np.array(grad_w).flatten()
        self.weights -= (np.array(grad_w) * LEARNING_RATE)
        self.bias -= (grad_b * LEARNING_RATE)
        

class Layer:
    def __init__ (self, neurons_required, weights_per_neuron,):
        self.neurons = []
        self.activations = []
        self.weighted_sums = []

        for i in range(neurons_required):
            self.neurons.append(Neuron(weights_per_neuron))

    def feedForward(self, previous_activations):
        self.weighted_sums = []
        self.activations = []

        for neuron in self.neurons:
            neuron.weighted_sum = 0
            neuron.activation = 0
        
            neuron.weighted_sum = np.dot(neuron.weights, previous_activations) + neuron.bias
            neuron.activation = sig(neuron.weighted_sum)
            
            self.weighted_sums.append(neuron.weighted_sum)
            self.activations.append(neuron.activation)

        self.weighted_sums = np.array(self.weighted_sums)
        self.activations = np.array(self.activations)

#-------------------------------------------
# Subroutines
#-------------------------------------------

# Sigmoid function
def sig(x):
    return 1/(1 + np.exp(-x))

# Derivatives & Cost function
def sig_derivative(x):
    sigmoid = sig(x)
    return sigmoid * (1 - sigmoid)

def weight_derivative(previous_activation, weighted_sum, error):
    return previous_activation * sig_derivative(weighted_sum) * 2 * error

def bias_derivative(weighted_sum, error):
    return sig_derivative(weighted_sum) * 2 * error

def activation_derivative(weights, weighted_sum, error):
    return weights * sig_derivative(weighted_sum) * 2 * error

def cost(given_activations, expected_activations):
    return np.sum(np.power(given_activations - expected_activations, 2))
   


# Feed forward through the neural network
def feedForwardLayers(input_activations, layers):

    index = 0
    for layer in layers:
        if index == 0: previous_activations = input_activations
        else: previous_activations = layers[index - 1].activations

        layer.feedForward(previous_activations)
        index += 1


# Back propagation with gradient descent
def backPropagation(input_activations, layer1, layer2, output_layer, expected_activations):

    # Output layer
    output_error = output_layer.activations[0] - expected_activations[0]
    output_weight_gradient = [weight_derivative(layer2.activations[0], output_layer.weighted_sums[0], output_error)]
    output_bias_gradient = bias_derivative(output_layer.activations[0], output_error)

    output_layer.neurons[0].update_gradient(output_weight_gradient, output_bias_gradient)


    # Layer 2
    layer2_activation_gradient = activation_derivative(output_layer.neurons[0].weights, output_layer.weighted_sums[0], output_error)
    layer2_error = layer2_activation_gradient[0]
    layer2_weight_gradient = weight_derivative(layer1.activations[0], layer2.weighted_sums[0], layer2_error)
    layer2_bias_gradient = bias_derivative(layer2.weighted_sums[0], layer2_error)

    layer2.neurons[0].update_gradient(layer2_weight_gradient, layer2_bias_gradient)

    # Layer 1
    layer1_activation_gradient = activation_derivative(layer2.neurons[0].weights, layer2.weighted_sums[0], layer2_error)
    layer1_error = layer1_activation_gradient
    layer1_weight_gradient = weight_derivative(input_activations[0], layer1.weighted_sums[0], layer1_error)
    layer1_bias_gradient = bias_derivative(layer1.weighted_sums[0], layer1_error)

    layer1.neurons[0].update_gradient(layer1_weight_gradient, layer1_bias_gradient)


# Fucntion to train the neural network
def train(input_activations, expected_activations, layers):
    layer1, layer2, output_layer = layers

    feedForwardLayers(input_activations, layers)
    print(f'Cost: {cost(output_layer.activations, expected_activations)}')
    backPropagation(input_activations, layer1, layer2, output_layer, expected_activations)   

    return cost(output_layer.activations, expected_activations)
    

# Predict a denary number based on 1bit binary by propagating it forward through the network
def recogniseBinary(input_array, layers):

    feedForwardLayers(input_array, layers)
    print(f'Probability of {input_array[0]} being 1: {np.round(layers[-1].activations[0], decimals=2) * 100}%')
     
    
#-------------------------------------------
# Main program
#-------------------------------------------


# Initialise layers
hidden_layer1 = Layer(1, 1)
hidden_layer2 = Layer(1, 1)
output_layer = Layer(1, 1)

layers = [hidden_layer1, hidden_layer2, output_layer]

zero_input = [0]
one_input = [1]


zero_output = [0]
one_output = [1]

def feedData():
    train(zero_input, zero_output, layers)
    train(one_input, one_output, layers)


for i in range(10000):
    feedData()



#print(layer1.activations)
recogniseBinary(zero_input, layers)
recogniseBinary(one_input, layers)


