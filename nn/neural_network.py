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
        self.layers = [num_inputs] + hidden_layers + [num_outputs]
        self.activations = activation_functions
        # for each layer make a list of weights and biases -1 because there are no weights from before input layer
        self.weights = [[] for _ in range(len(self.layers)-1)]
        self.derivatives = [[] for _ in range(len(self.layers)-1)]
        self.bias = [[] for _ in range(len(self.layers)-1)]
        self.delta = [[] for _ in range(len(self.layers)-1)]

        # should be layers x outputs per layer
        self.unactivated_outputs = [np.zeros((1,x)) for x in self.layers]
        self.outputs = [np.zeros((1,x)) for x in self.layers]
        print(len(self.unactivated_outputs),self.unactivated_outputs[0].shape)

        for layer_idx in range(len(self.layers)-1):
            next_idx = layer_idx + 1
            # make weights current layer x next layer, so we have fully connected layer wij -> output1 ect
            self.weights[layer_idx] = np.random.rand(self.layers[layer_idx],self.layers[next_idx])
            self.derivatives[layer_idx] = np.random.rand(self.layers[layer_idx],self.layers[next_idx])
            self.bias[layer_idx] = np.zeros((1,self.layers[next_idx]))
            self.delta[layer_idx] = np.zeros((1,self.layers[next_idx]))



    def feed_forward(self,input_vector:NDArray[float64]) -> NDArray[float64]:
        """
        Predict the next angles for the joints
        Parameters:
            input_vector (list[float]): This should be a list of angles which has a length of 24 x batch size
        Returns:
            final layer of NN output, for the next joint prediction, will be output x batch size.
        """
        output = input_vector
        self.unactivated_outputs[0] = output
        self.outputs[0] = output
        for i in range(len(self.weights)):
            # ignore first layer but forward pass for the rest
            # print("weights and output shape",self.weights[i].shape,output.shape)
            # print("bias shape",self.bias[i].shape)

            next_out = np.dot(output, self.weights[i]) + self.bias[i]


            # set output of layer after input layer
            self.unactivated_outputs[i+1] = next_out
            output = self.activations[i](next_out)
            self.outputs[i+1] = output
            # print("---passed 1 weight layer ---")
        return output

    def back_propagation(self,error:NDArray[float64],verbose:bool = False) :
        """
        Method goes backwards through layers and calculate errors needed to update weights
        Parameters:
            error (NDArray[float64]): Error
        """
        # https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/
        if verbose: print("========starting back prop========")
        # must get error for each output NN
        for i in range(len(self.weights)-1,-1,-1):
            if verbose: print(f"----start {i} layer bp ---")

            current_outputs = self.outputs[i+1]
            if verbose: print("error shape:",error.shape)

            # with current layer outputs
            derivative = ACTIVATION_DERITIVIVE_MAP[self.activations[i]]
            if verbose: print('derititive selected:',derivative)

            error_signal = error * derivative(current_outputs)
            self.delta[i] = error_signal
            if verbose: print("error signal:",error_signal.shape)

            # the layer before the current layer
            prev_layer = self.outputs[i]

            # set gradient for this bp
            if verbose: print(f"modifying derititive {i}/{len(self.derivatives)-1} - dot product of prev:{prev_layer.shape} x err:{error_signal.T.shape}")

            self.derivatives[i] = np.dot(prev_layer.T,error_signal)

            if verbose: print(f"produced derititive matrix of size {self.derivatives[i].shape} compared to weights of size {self.weights[i].shape}")

            error = np.dot(error_signal,self.weights[i].T)

            if verbose: print(f"setting error matrix to size {error.shape}")


            if verbose: print(f"--{i} layer passed bp---")

        if verbose: print("========finished back prop========")
