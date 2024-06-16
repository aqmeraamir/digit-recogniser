# Neural Network for Digit Recognition 
I made this network (my first) to introduce myself to deep learning, and it is largely based on [3Blue1Brown's playlist](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) on neural networks, where the concepts, and relevant math was explained, although I used other resources too (mostly for explaining the math such as the activation functions or other concepts such as the types of training). 

Note: It does **not** use libraries such as TensorFlow or PyTorch, as I thought it would be more useful in teaching myself such concepts by creating it from **scratch**.


## Architecture
This neural network is able to take a 28x28 image as an input, and feed its pixels through the network to output 10 probabilities for the image of consisting a digit. The architecture consists of **4 layers**, including:
- Input Layer (784 neurons, each with a grayscale value of a pixel)
- Hidden Layer 1 (16 neurons)
- Hidden Layer 2 (16 neurons)
- Output Layer (10 neurons, each representing a digit)

so, in total, ~13,000 weights & biases. And for the activations, **sigmoid** is used, overall making it a MLP (multilayer perceptron) with fully connected neurons and a non-linear activation function.

<div align="center">
  <img src="https://github.com/aqmeraamir/digit-recognising-neural-network/blob/main/images/network_architecture.png" alt="Neural Network Archhitecture" width=500></img>
  
  <i>credit: 3Blue1Brown</i>
</div>

</br>



## Training
The network uses back-propagation and gradient descent, with the required calculus and linear algebra from [Chapter 4, Deep Learning](https://www.youtube.com/watch?v=tIeHLnjs5U8&t=307s)

The repository includes a pre-trained ```model_data.pkl``` file, which has been trained using **mini-batch** gradient descent with the [**MNIST dataset**](https://www.kaggle.com/datasets/hojjatk/mnist-dataset). However, the program can easily be modified to change
the batch size, and switch to **full batch** gradient descent or **stochastic** gradient descent. 

The pre-trained data can be reset, and the user can specify the number of epochs they wish to train the network with. 

Training also includes a **Loss vs Epoch graph** (also loss vs iteration) which can be shown live as its trained or statically once training is complete, helping find the perfect number of epochs required. Some demonstrations:

<table align="center">
    <tr>
        <td align="center">
            <img src="https://github.com/aqmeraamir/digit-recognising-neural-network/blob/main/images/graph1.png" alt="Graph 1" style="width:500px; height:300px;">
            <br><i>10,000 samples per digit; batch size of 10; learning rate 0.01; 25 epochs</i>
        </td>
        <td align="center">
            <img src="https://github.com/aqmeraamir/digit-recognising-neural-network/blob/main/images/graph2.gif" alt="Graph 2" style="width:500px; height:300px;">
            <br><i>stochastic gradient descent (used low no. of samples for display purposes); learning rate 0.01</i>
        </td>
    </tr>
</table>

</br>
 
 The program consists of several other features as well, which help analyse and use the data effectively, such as:
- Method to save updated weights & biases using the pickle library.
- Uses layer & neuron classes so it is easily scalable, and will work with any number of neurons in each layer.
- User interface

<img src="https://github.com/aqmeraamir/digit-recognising-neural-network/blob/main/images/user_interface.png" alt="User Interface" style="width:500px; height:300px;"> 


*There are two other programs in ```Test Neural Networks/``` which were used to help create the final image recognition program. For example the 1 bit recogniser, which helped integratating back propagation as it was much easier working with a single neuron in each layer, before developing the back propagation for multiple neurons.*
