# Neural Network for Digit Recognition 
I made this network to introduce myself to deep learning, and it is largely based of [3Blue1Brown's playlist](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) on neural networks, where the concepts, and relevant math was explained, although I used other resources too (mostly for explaining the math or other concepts such as the types of training). 

Note: It does **not** use libraries such as TensorFlow or PyTorch, as I thought it would be more useful in teaching myself such concepts by creating it from scratch.


## Architecture
This neural network is able to take a 28x28 image as an input, and feed its pixels through the network to output 10 probabilities for the image of consisting a digit. The architecture consists of **4 layers**, including:
- Input Layer (784 neurons, each with a grayscale value of a pixel)
- Hidden Layer 1 (16 neurons)
- Hidden Layer 2 (16 neurons)
- Output Layer (10 neurons, each representing a digit)

so, in total, ~13,000 weights & biases. And for the activations, **sigmoid** is used.

</br>

The program consists of several other features as well, which help analyse and use the data effectively, such as:
- Loss vs Epoch graph (which can be updated live as the network is being trained, or shown as a static once training is complete)
- Method to save updated weights & biases using the pickle library.
- Easily scalable, and will work with any number of neurons in each layer

</br>


