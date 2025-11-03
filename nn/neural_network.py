import numpy as np
from activation_functions import *
from numpy._core.numerictypes import float64
from numpy._typing import NDArray

class Neural_network():
    def __init__(self, hidden_layers:list[int] = [24,24],num_outputs:int = 24,num_inputs:int=24,learning_rate:float=0.05) -> None:
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
        self.outputs: NDArray[float64] = np.zeros((len(self.layers),num_outputs))

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


    def feed_forward(self,input_vector:list[float]) -> NDArray[float64]:
        """
        Predict the next angles for the joints
        Parameters:
            input_vector (list[float]): This should be a list of angles which has a length of 24
        Returns:
            final layer of NN output, for the next joint prediction.
        """
        # set the first output as first vector
        output = input_vector
        self.outputs[0] = output

        for i, w in enumerate(self.weights):
            next_output = np.dot(output,w)
            # activation function
            output= sigmoid(next_output)
            self.outputs[i+1] = output
        # return final output
        return output

    def back_propagation(self,error:NDArray[float64]) :
        """
        Method goes backwards through layers and calculate errors needed to update weights
        Parameters:
            error (somthing): Error
        """
        # starts at last index goes down to first
        for i in range(len(self.derivatives),0,-1):
            # get previous layer
            output = self.outputs[i+1]

            derivative = error*sigmoid_derivitive(output)

            # reshape array into appropriate size
            derivative_fixed = derivative.reshape(derivative.shape[0],-1).T 
            layer_output = self.outputs[i]
            layer_output = layer_output.reshape(layer_output.shape[0],-1)
            # reshape to get column array for multiplication

            self.derivatives[i] = np.dot(layer_output,derivative_fixed)

            error = np.dot(self.derivatives, self.weights[i].T)
            # update error for next layer of network

    def gradient_descent(self):
        """
        Method used to update weights based on derivatives calculated in back propagation
        """
        for i in range(len(self.weights)):
            # update the weight by adding derivative*learning_rate
            self.weights[i] += self.derivatives[i]*self.learning_rate
