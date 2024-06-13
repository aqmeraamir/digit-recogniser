# Import libraries
import numpy as np

# CONSTANTS
LEARNING_RATE = 0.1

#-------------------------------------------
# Classes for neurons & layers
#-------------------------------------------
class Neuron:
   
    def __init__(self, weights_required):
        #self.weights = np.random.uniform(-0.1, 0.1, weights_required)  # small random weights
        #self.bias = np.random.uniform(-0.1, 0.1)  # small random bias

        self.weights = np.random.randn(weights_required) * np.sqrt(2 / weights_required)
        #self.bias = np.random.randn() * np.sqrt(2 / weights_required)
        self.bias = 0

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

        for _ in range(neurons_required):
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

# Derivatives & Cost functions (taken from 3blue1brown youtube video titled 'Backpropagation calculus | Chapter 4, Deep learning') *slightly modified to use error as a parameter for (given - expected activation)
def sig_derivative(x):
    sigmoid = sig(x)
    return sigmoid * (1 - sigmoid)

def weight_derivative(previous_activation, weighted_sum, error):
    return previous_activation * sig_derivative(weighted_sum) * 2 * error

def bias_derivative(weighted_sum, error):
    return sig_derivative(weighted_sum) * 2 * error

def activation_derivative(weights, weighted_sum, error):
    return weights * sig_derivative(weighted_sum) * 2 * error

def cost(output_activations, expected_activations):
    return np.sum(np.power(output_activations - expected_activations, 2))
   

# Feed forward through the neural network
def feedForwardInput(input_activations, layers):
    previous_activations = input_activations

    for layer in layers:
        layer.feedForward(previous_activations)
        previous_activations = layer.activations


# Back propagation with gradient descent
def backPropagation(input_activations, layer1, layer2, output_layer, expected_activations):

    # Output layer
    output_errors = []
    for output_neuron, expected_activation in zip(output_layer.neurons, expected_activations):
        neuron_weight_gradients = []
        neuron_bias_gradient = 0

        output_error = output_neuron.activation - expected_activation
        output_errors.append(output_error)

        for layer2_neuron in layer2.neurons:
            neuron_weight_gradients.append(weight_derivative(layer2_neuron.activation, output_neuron.weighted_sum, output_error))

        neuron_bias_gradient = bias_derivative(output_neuron.weighted_sum, output_error)
        
        output_neuron.update_gradient(neuron_weight_gradients, neuron_bias_gradient)
        #output_neuron.weight_gradients.append(neuron_weight_gradients)
        #output_neuron.bias_gradients.append(neuron_bias_gradient)


    # Layer 2
    layer2_errors = []
    for l2_index, layer2_neuron in enumerate(layer2.neurons):
        layer2_error = 0
        neuron_weight_gradients = []
        neuron_bias_gradient = 0

        for o_index, output_neuron in enumerate(output_layer.neurons):
            layer2_error += activation_derivative(output_neuron.weights[l2_index], output_neuron.weighted_sum, output_errors[o_index])
      
        layer2_errors.append(layer2_error)

        for layer1_neuron in layer1.neurons:
            neuron_weight_gradients.append(weight_derivative(layer1_neuron.activation, layer2_neuron.weighted_sum, layer2_error))
        
        neuron_bias_gradient = bias_derivative(layer2_neuron.weighted_sum, layer2_error)
        
        layer2_neuron.update_gradient(neuron_weight_gradients, neuron_bias_gradient)
        #layer2_neuron.weight_gradients.append(neuron_weight_gradients)
        #layer2_neuron.bias_gradients.append(neuron_bias_gradient)


    # Layer 1
    for l1_index, layer1_neuron in enumerate(layer1.neurons):
        layer1_error = 0
        neuron_weight_gradients = []
        neuron_bias_gradient = 0

        for l2_index, layer2_neuron in enumerate(layer2.neurons):
            layer1_error += activation_derivative(layer2_neuron.weights[l1_index], layer2_neuron.weighted_sum, layer2_errors[l2_index])

        for activation in input_activations:
            neuron_weight_gradients.append(weight_derivative(activation, layer1_neuron.weighted_sum, layer1_error))

        neuron_bias_gradient = bias_derivative(layer1_neuron.weighted_sum, layer1_error)

        layer1_neuron.update_gradient(neuron_weight_gradients, neuron_bias_gradient)
        #layer1_neuron.weight_gradients.append(neuron_weight_gradients)
        #ayer1_neuron.bias_gradients.append(neuron_bias_gradient)


# Fucntion to train the neural network
def train(input_activations, expected_activations, layers):
    layer1, layer2, output_layer = layers

    feedForwardInput(input_activations, layers)
    #print(f'Cost: {cost(output_layer.activations, expected_activations)}')
    backPropagation(input_activations, layer1, layer2, output_layer, expected_activations)   

    return cost(output_layer.activations, expected_activations)


# Predict a denary value, using an array with binary 
def recogniseBinary(input_array, layers):

    feedForwardInput(input_array, layers)
    output_activations = layers[-1].activations
    highest_value = np.argmax(output_activations)

    print(f'----------- {str(input_array)} IS LIKELY: {highest_value} ----------')
    print(f'Probability of it being 0: {np.round(layers[-1].activations[0], decimals=2) * 100}%')
    print(f'Probability of it being 1: {np.round(layers[-1].activations[1], decimals=2) * 100}%')
    print(f'Probability of it being 2: {np.round(layers[-1].activations[2], decimals=2) * 100}%')
    print(f'Probability of it being 3: {np.round(layers[-1].activations[3], decimals=2) * 100}%')
    print('------------------------------------------\n')

def updateGradients(layers):
    for layer in layers:
        for neuron in layer.neurons:
            if neuron.weight_gradients:
                avg_weight_grad = np.mean(neuron.weight_gradients, axis=0)
                avg_bias_grad = np.mean(neuron.bias_gradients)
                neuron.update_gradient(avg_weight_grad, avg_bias_grad)
            neuron.weight_gradients.clear()  # Clear gradients after updating
            neuron.bias_gradients.clear()
        
def feedData():
    costs = []
    costs.append(train(zero_input, zero_expected_output, layers))
    costs.append(train(three_input, three_expected_output, layers))
    costs.append(train(two_input, two_expected_output, layers))
    costs.append(train(one_input, one_expected_output, layers))

    print(f'Average costs: {np.mean(costs)}')

#-------------------------------------------
# Main program
#-------------------------------------------


# Initialise layers
hidden_layer1 = Layer(2, 2)
hidden_layer2 = Layer(2, 2)
output_layer = Layer(4, 2)

layers = [hidden_layer1, hidden_layer2, output_layer]

zero_input = [0, 0]
one_input = [0, 1]
two_input = [1, 0]
three_input = [1, 1]

zero_expected_output = [1, 0, 0, 0]
one_expected_output = [0, 1, 0, 0]
two_expected_output = [0, 0, 1, 0]
three_expected_output = [0, 0, 0, 1]


for i in range(50000):
    feedData()


# User interface - ask user for a denary number, and then propagate it through the nerwork
binary_number = input("Enter a 2-bit binary number: ")
input_array = [int(char) for char in list(binary_number)]

recogniseBinary(one_input, layers)