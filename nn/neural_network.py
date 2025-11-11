import numpy as np
from activation_functions import *
from numpy._core.numerictypes import float64
from numpy._typing import NDArray

class Neural_network():
    def __init__(self, hidden_layers:list[int] = [24],num_outputs:int = 24,num_inputs:int=24,learning_rate:float=0.5, activation_functions:list = [sigmoid]) -> None:
        """
        Initializes the neural network
        Parameters:
            hidden_layers (list[int]): Size of hidden layers, each entry is a layer
            num_outputs (int): Number of outputs, defaults to 24 as we want to predict the next position.
            num_inputs (int): Number of inputs, defaults to 24 for pose.
            learning_rate (float): rate of learning
        """
        self.learning_rate = learning_rate
        # weights and bias in a list, each index corresponds to a layers weights and bias
        self.weights: list[NDArray[float64]] = []
        self.bias: list[NDArray[float64]] = []
        self.derivatives: list[NDArray[float64]] = []
        # used to get size for weights and bias
        self.layers: list[int]= [num_inputs] + hidden_layers + [num_outputs]
        # used to store each layers activation function
        self.activations:list[function] = activation_functions

        self.outputs: list[NDArray[float64]] = [np.zeros(self.layers[l]) for l in range(len(self.layers))]

        for layer in range(len(self.layers)-1):
            # Init layer weights, bias and derivatives
            self.weights.append(np.array([]))
            self.bias.append(np.array([]))
            self.derivatives.append(np.array([]))
            # Create input*output sized matrix for weights
            self.weights[layer] = np.random.rand(self.layers[layer],self.layers[layer+1])
            # Copy matrix format for derivatives, fill with zeros.
            self.derivatives[layer] = np.zeros((self.layers[layer],self.layers[layer+1]))
            # Create bias for each perceptron
            self.bias[layer] = np.random.rand(1,self.layers[layer+1])


    def feed_forward(self,input_vector:NDArray[float64]) -> NDArray[float64]:
        """
        Predict the next angles for the joints
        Parameters:
            input_vector (list[float]): This should be a list of angles which has a length of 24
        Returns:
            final layer of NN output, for the next joint prediction.
        """
        # set the first output as input vector
        output = input_vector
        self.outputs[0] = output

        # need to workout how to transpose for 

        for i, w in enumerate(self.weights):
            next_output = np.dot(output,w)
            # activation function
            # output = sigmoid(next_output)
            output = self.activations[i](next_output)
            self.outputs[i+1] = output
        return output

    def back_propagation(self,error:NDArray[float64]) :
        """
        Method goes backwards through layers and calculate errors needed to update weights
        Parameters:
            error (NDArray[float64]): Error
        """
        # this only works for 1 value, since error is a list of errors how is this calculated?
        for i in reversed(range(len(self.derivatives))):
            # work backwards with index from last layer to first
            # get previous layer
            # the output of last layer it is still + 1, but works out since there are 3 layers over all of weights between input hidden and out
            output = self.outputs[i+1]

            # get derititive function for this layers activation
            derivative_function:function = ACTIVATION_DERITIVIVE_MAP[self.activations[i]]

            delta = error*derivative_function(output)

            delta_fixed = delta.reshape(delta.shape[0],-1).T 

            # current layer output
            layer_output = self.outputs[i]
            layer_output = layer_output.reshape(layer_output.shape[0],-1)

            # erroring here because output[2] has a format of 24*3*24
            self.derivatives[i] = np.dot(layer_output,delta_fixed)

            error = np.dot(delta, self.weights[i].T)
