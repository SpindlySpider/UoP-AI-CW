import numpy as np
from numpy._core.numerictypes import float64
from numpy._typing import NDArray

from activation_functions import *

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
        # set class variables
        self.learning_rate = learning_rate
        self.layers = [num_inputs] + hidden_layers + [num_outputs]
        self.activations = activation_functions
        # for each layer make a list of weights and biases -1 because there are no weights from before input layer
        # goes through each layer and makes weights to next layer
        self.weights = [[] for _ in range(len(self.layers)-1)]
        # goes through each layer and defines derivative matrices for backprop
        self.derivatives = [[] for _ in range(len(self.layers)-1)]
        # biases for each layer
        self.bias = [[] for _ in range(len(self.layers)-1)]
        # delta for each layer
        self.delta = [[] for _ in range(len(self.layers)-1)]
        # store outputs for each layer
        self.unactivated_outputs = [np.zeros((1,x)) for x in self.layers]
        self.outputs = [np.zeros((1,x)) for x in self.layers]

        # initialize weights and biases
        for layer_idx in range(len(self.layers)-1):
            # get next layer index
            next_idx = layer_idx + 1
            # initialize weights with random values between -0.5 and 0.5
            self.weights[layer_idx] = np.random.rand(self.layers[layer_idx],self.layers[next_idx]) -0.5
            # initialize derivatives to random values
            self.derivatives[layer_idx] = np.random.rand(self.layers[layer_idx],self.layers[next_idx])
            # initialize biases to zero minus 0.5
            self.bias[layer_idx] = np.zeros((1,self.layers[next_idx])) - 0.5
            # initialize delta to zero
            self.delta[layer_idx] = np.zeros((1,self.layers[next_idx]))



    def feed_forward(self,input_vector:NDArray[float64]) -> NDArray[float64]:
        """
        Predict the next angles for the joints
        Parameters:
            input_vector (list[float]): This should be a list of angles which has a length of 24 x batch size
        Returns:
            final layer of NN output, for the next joint prediction, will be output x batch size.
        """
        # set input as output of first layer
        output = input_vector
        # store input layer outputs
        self.unactivated_outputs[0] = output
        # store input layer outputs
        self.outputs[0] = output

        # go through each layer
        for i in range(len(self.weights)):
            # calculate next layer output
            next_out = np.dot(output, self.weights[i]) + self.bias[i]
            # set output of layer after input layer
            self.unactivated_outputs[i+1] = next_out
            output = self.activations[i](next_out)
            self.outputs[i+1] = output
        return output

    def back_propagation(self,error:NDArray[float64],verbose:bool = False) :
        """
        Method goes backwards through layers and calculate errors needed to update weights
        Parameters:
            error (NDArray[float64]): The error from the output layer
            verbose (bool): Whether to print debug information
        """
        if verbose: print("========starting back prop========")

        # go backwards through each layer
        for i in range(len(self.weights)-1,-1,-1):
            if verbose: print(f"----start {i} layer bp ---")
            # get outputs from current layer
            current_outputs = self.outputs[i+1]
            if verbose: print("error shape:",error.shape)

            # with current layer outputs
            # get derivative function
            derivative = ACTIVATION_DERITIVIVE_MAP[self.activations[i]]
            if verbose: print('derititive selected:',derivative)

            # calculate error signal
            error_signal = error * derivative(current_outputs)


            # set delta for the current layer
            self.delta[i] = error_signal

            if verbose: print("error signal:",error_signal.shape)

            # the layer before the current layer
            prev_layer = self.outputs[i]

            # set gradient for the current batch
            if verbose: print(f"modifying derititive {i}/{len(self.derivatives)-1} - dot product of prev:{prev_layer.shape} x err:{error_signal.T.shape}")

            # calculate derivative for the current layer
            self.derivatives[i] = np.dot(prev_layer.T,error_signal)

            if verbose: print(f"produced derititive matrix of size {self.derivatives[i].shape} compared to weights of size {self.weights[i].shape}")

            error = np.dot(error_signal,self.weights[i].T)

            if verbose: print(f"setting error matrix to size {error.shape}")

            if verbose: print(f"--{i} layer passed bp---")
            
        if verbose: print("========finished back prop========")
