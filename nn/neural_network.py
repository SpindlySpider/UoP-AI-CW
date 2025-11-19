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

        self.outputs: list[NDArray[float64]] = [np.zeros((self.layers[l])) for l in range(len(self.layers))]
        self.delta:list[NDArray[float64]] = []

        for layer in range(len(self.layers)-1):
            # Init layer weights, bias and derivatives
            self.weights.append(np.array([]))
            self.bias.append(np.array([]))
            self.derivatives.append(np.array([]))
            self.delta.append(np.array([]))
            # Create input*output sized matrix for weights
            self.weights[layer] = np.random.rand(self.layers[layer],self.layers[layer+1]) - 0.5
            # Copy matrix format for derivatives, fill with zeros.
            self.derivatives[layer] = np.zeros((self.layers[layer],self.layers[layer+1]))
            # Create bias for each perceptron
            self.bias[layer] = np.random.rand(self.layers[layer+1])

            # Fill out null values for delta, so we can access later for bias update.
            self.delta[layer] = np.zeros((self.layers[layer+1]))



    def feed_forward(self,input_vector:NDArray[float64]) -> NDArray[float64]:
        """
        Predict the next angles for the joints
        Parameters:
            input_vector (list[float]): This should be a list of angles which has a length of 24 x batch size
        Returns:
            final layer of NN output, for the next joint prediction.
        """
        # set first layer to input vector
        output:NDArray[float64] = input_vector
        self.outputs[0] = output

        for i, w in enumerate(self.weights):
            next_output = np.dot(output, w) + self.bias[i]
            # next_output = np.dot(output, w) 
            output = self.activations[i](next_output)
            self.outputs[i+1] = output
        return output

    def back_propagation(self,error:NDArray[float64]) :
        """
        Method goes backwards through layers and calculate errors needed to update weights
        Parameters:
            error (NDArray[float64]): Error
        """
        for i in reversed(range(len(self.derivatives))):
            # work backwards with index from last layer to first
            output = self.outputs[i+1]

            # workout which function should be used for this layer
            derivative_function:function = ACTIVATION_DERITIVIVE_MAP[self.activations[i]]

            # error value here needs to be derititve of MSE
            # is the output here causing it to be batch size error?
            delta = error * derivative_function(output)
            # save delta / error signal to update bias later
            # reshape delta for matrix multiplication and transpose
            # delta_fixed = delta.reshape(delta.shape[0],-1).T

            self.delta[i] = np.mean(delta,axis=0)

            # set to previous layer

            next_layer = self.outputs[i]

            self.derivatives[i] = np.dot(next_layer.T, delta)
            self.derivatives[i] = self.derivatives[i]/error.shape[0]

            # Set error for next layer to back propagate
            # this bit breaks?
            error = np.dot(delta, self.weights[i].T)
