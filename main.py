'''
Project Name: Neural Network for Digit Recognition
Github: https://github.com/aqmeraamir

Notes:
- this program is based on the first 4 videos of 3blue1brown's playlist on deep learning on youtube (https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- entire neural network is mostly created from scratch, without libraries such as TensorFlow or PyTorch
- consists of 4 layers:
    1. input layer (784 input neurons that each contain a grayscalue value 0-1 of a pixel), hidden 
    2. hidden layer 1 (16 neurons, each with 784 weights)
    3. hidden layer 2 (16 neurons, each with 16 weights)
    4. output layer (10 neurons with each representing a digit between 0-9; each neuron has 16 weights)

- includes a graph of how cost changes as the network is trained

'''

# Importing libraries
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math
import os 

# CONSTANTS
LEARNING_RATE = 0.01

LAYER1_SIZE = 90
LAYER2_SIZE = 90
OUTPUT_SIZE = 10


#-------------------------------------------
# Classes for neurons & layers
#-------------------------------------------
class Neuron:
   
    def __init__(self, weights_required):
        #self.weights = np.random.uniform(-0.1, 0.1, weights_required)  # small random weights
        self.weights = np.random.randn(weights_required) * np.sqrt(2 / weights_required)

        #self.bias = np.random.uniform(-0.1, 0.1)  # small random bias
        self.bias = 0

        self.weighted_sum = 0
        self.activation = 0

        self.weight_gradients = []
        self.bias_gradients = []

    def update_gradient(self, grad_w, grad_b):
        grad_w = np.array(grad_w).flatten()
        self.weights -= (np.array(grad_w) * LEARNING_RATE)
        self.bias -= (grad_b * LEARNING_RATE)
    
    # Functions to save and load the parameters (weights & biases) of the neuron 
    def getParams(self):  
        return {'weights': self.weights, 'bias': self.bias}

    def setParams(self, params):
        self.weights = params['weights']
        self.bias = params['bias']
    
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
            neuron.weighted_sum = np.dot(neuron.weights, previous_activations) + neuron.bias        
            self.weighted_sums.append(neuron.weighted_sum)
        
        if len(self.neurons) == OUTPUT_SIZE:
            self.activations = softmax(self.weighted_sums)
            
        else:
            self.activations = sig(self.weighted_sums)
        
        for neuron, activation in zip(self.neurons, self.activations):
            neuron.activation = activation



    # Functions to save and load parameters of the layer
    def getParams(self):
        return [neuron.getParams() for neuron in self.neurons]
    
    def setParams(self, params):
        for neuron, param in zip(self.neurons, params):
            neuron.setParams(param)

#-------------------------------------------
# Subroutines
#-------------------------------------------

# Math functions
def sig(x):
    x = np.array(x)
    return 1/(1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # stability improvement
    return exp_x / exp_x.sum(axis=0)

def round_sf(number, sf):
    if number == 0:
        return 0.0
    
    magnitude = math.floor(math.log10(abs(number)))
    precision = sf - 1 - magnitude
    return round(number, precision)


# Derivatives & Cost functions (taken from 3blue1brown youtube video titled 'Backpropagation calculus | Chapter 4, Deep learning' *slightly modified to use error as a parameter for (given - expected activation)
def sig_derivative(x):
    sigmoid = sig(x)
    return sigmoid * (1 - sigmoid)

def softmax_derivative(softmax_output):
    s = softmax_output.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

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

        output_error = output_neuron.activation - expected_activation
        output_errors.append(output_error)
 
        neuron_weight_gradients = weight_derivative(layer2.activations, output_neuron.weighted_sum, output_error)
        neuron_bias_gradient = bias_derivative(output_neuron.weighted_sum, output_error)
        
        output_neuron.update_gradient(neuron_weight_gradients, neuron_bias_gradient) # Adjust the weights & biases with gradients


    # Layer 2
    layer2_errors = []
    for l2_index, layer2_neuron in enumerate(layer2.neurons):

        output_weights = []
        output_weighted_sums = []
        for output_neuron in output_layer.neurons:
            output_weights.append(output_neuron.weights[l2_index])
            output_weighted_sums.append(output_neuron.weighted_sum)

        layer2_error = np.sum(activation_derivative(output_weights, output_weighted_sums, output_errors))
        layer2_errors.append(layer2_error)

        neuron_weight_gradients = weight_derivative(layer1.activations, layer2_neuron.weighted_sum, layer2_error)
        neuron_bias_gradient = bias_derivative(layer2_neuron.weighted_sum, layer2_error)
        
        layer2_neuron.update_gradient(neuron_weight_gradients, neuron_bias_gradient) # Adjust the weights & biases with gradients
 

    # Layer 1
    input_activations = np.array(input_activations)
    for l1_index, layer1_neuron in enumerate(layer1.neurons):

        l2_weights = []
        l2_weighted_sums = []
        for layer2_neuron in layer2.neurons:
            l2_weights.append(layer2_neuron.weights[l1_index])
            l2_weighted_sums.append(layer2_neuron.weighted_sum)

        layer1_error = np.sum(activation_derivative(l2_weights, l2_weighted_sums, layer2_errors))

        neuron_weight_gradients = weight_derivative(input_activations, layer1_neuron.weighted_sum, layer1_error)
        neuron_bias_gradient = bias_derivative(layer1_neuron.weighted_sum, layer1_error)

        layer1_neuron.update_gradient(neuron_weight_gradients, neuron_bias_gradient) # Adjust the weights & biases with gradients


# Function to train the neural network
def train(input_activations, expected_activations, layers):
    feedForwardInput(input_activations, layers)
    backPropagation(input_activations, layers[0], layers[1], layers[-1], expected_activations)   

    return cost(layers[-1].activations, expected_activations)


# Retrieve the grayscale value of each pixel in an image
def initialiseImage(file_path):
    try: 
        image = Image.open(file_path)
        alpha_channel = image.split()[3]
        image_array = np.array(alpha_channel)

        gray_scale_array = image_array / 255 # Create an array with the grayscale value of each pixel
        gray_scale_array = [item for sublist in gray_scale_array for item in sublist] # Flatten the array

        if len(gray_scale_array) < 784: 
            print('ERROR: Invalid image, must be a 28x28 png.')
        else:
            return gray_scale_array
        
    except:
        print(f"\nERROR: File path '{file_path}' is invalid")
         

# Functions to save & load the neural network model
def saveModel(layers, filename='model_data.pkl'):
    model_params = [layer.getParams() for layer in layers]
    with open(filename, 'wb') as file:
        pickle.dump(model_params, file)
    
def loadModel(filename='model_data.pkl'):
    with open(filename, 'rb') as file:
        model_params = pickle.load(file)

    # Reinitialise the layers
    layer1 = Layer(LAYER1_SIZE, 784)
    layer2 = Layer(LAYER2_SIZE, LAYER1_SIZE)
    output_layer = Layer(OUTPUT_SIZE, LAYER2_SIZE)

    layers = [layer1, layer2, output_layer]

    for layer, params in zip(layers, model_params):
        layer.setParams(params)

    return layers


# Functions to initialise (reset), train or feed input into the model for the user interface
def initialiseModel():
    layers = [Layer(LAYER1_SIZE, 784), Layer(LAYER2_SIZE, LAYER1_SIZE), Layer(OUTPUT_SIZE, LAYER2_SIZE)]
    saveModel(layers)
    return layers

def trainModel(iterations, use_live_graph, use_graph):
    def update_plot(average_costs):
        line.set_linewidth(0.35)
        line.set_data(range(len(average_costs)), average_costs)

        ax.set_ylim(0, 1)

        ax.relim()
        ax.autoscale_view()

        plt.draw()
        plt.pause(0.005)

    # Initialize plotting for a real-time graph
    if use_live_graph:
        plt.ion() 
        ax = plt.subplots()[1]
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Cost')
        line, = ax.plot([], [], 'r-') 

    average_costs = []
    for i in range(iterations):
        costs = []
        for j in range(10):
            image_array = initialiseImage(f'MNIST_dataset/{j}/{j}/{i}.png')

            expected_output = [0] * 10
            expected_output[j] = 1
            cost_of_image = train(image_array, expected_output, layers)
            costs.append(cost_of_image)

        if (i + 1) % 150 == 0: saveModel(layers)  # Save the model every 100 iterations

        average_cost = np.mean(costs)
        average_costs.append(average_cost)

        print(f'{i} - Average Cost: {average_cost}')
        
        # Update the real-time graphwith new data
        if use_live_graph: update_plot(average_costs)

    # Plot a graph for average costs
    if use_graph:
        plt.xlabel('Iteration')
        plt.ylabel('Training error')
        plt.ylim(0, 1)
        plt.plot(range(len(average_costs)), average_costs, linewidth=0.35)
     
    if use_live_graph: plt.ioff()  # Turn off interactive mode
    
    # Calculate the average decrease in cost after training
    n = 50
    average_start = sum(average_costs[:n]) / n 
    average_end = sum(average_costs[-n:]) / n

    return average_start - average_end


# Predict what digit is in an image by propagating its array through the layers
def recogniseDigit(filename, layers):
    image_array = initialiseImage(filename)
    
    if image_array:
        feedForwardInput(image_array, layers)
        
        # Display probabilities
        highest_number = 0
        count = 0
        print('\nNumber | Probability (%)')
        print('-------|-----------------')

        for i in layers[2].activations:
            if i == layers[2].activations.max():
                highest_number = count
            
            percentage = str(np.round(i * 100, decimals = 2)) + '%'
            print(f'{count:^7}|{percentage:^14}')
            count += 1

        print(f'\nNUMBER IS LIKELY: {highest_number}')


#-----------------------------------------
# Main program
#-------------------------------------------

while True:
    # If data file doesn't exist, creat one
    if os.path.exists('model_data.pkl') == False: 
        initialiseModel()

    # Initialise layers by loading them
    layers = loadModel('model_data.pkl')

    # User interface
    print('\n---------- Handwritten Digit Recognisor - MENU ----------')
    print('1 - Recognise a handwritten digit')
    print('2 - Train the neural network using the MNIST dataset')
    print('3 - Reinitialise the neural network (must be retrained)')
    print('4 - Exit')
    print('---------------------------------------------------------')
    menu_input = input('Choose action (1-4): ')

    match menu_input:
        case '1':
            file_path = input('Enter filepath/url to a handwritten image (28x28 png): ')
            recogniseDigit(file_path, layers)

        case '2':
            iterations = int(input('With how many images do you want to train the network (30-10000)? '))
            cost_decrease = trainModel(iterations, True, False)  
            plt.show() # Display the graph
            print(f'\nMean decrease in cost: {round_sf(cost_decrease, 4)}\nTraining complete.')  

        case '3':
            initialiseModel()
            print('Successfully reset the model.')

        case '4':
            print('Exiting the proram')
            break;
