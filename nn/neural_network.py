import numpy as np
from activation_functions import *
from numpy._core.numerictypes import float64
from numpy._typing import NDArray

class Neural_network():
    def __init__(self, hidden_layers:list[int] = [24],num_outputs:int = 24,num_inputs:int=24,learning_rate:float=0.5) -> None:
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
        # set the first output as first vector
        output = input_vector
        self.outputs[0] = output

        for i, w in enumerate(self.weights):
            next_output = np.dot(output,w)
            # activation function
            output = sigmoid(next_output)
            # I think this is the error
            self.outputs[i+1] = output
        # print("ff output len:",len(self.outputs))
        # print("ff output Shape 1:",self.outputs[0].shape)
        # print("ff output Shape 2:",self.outputs[1].shape)
        # print("ff output Shape 3:",self.outputs[2].shape)
        # print("ff output Shape 4:",self.outputs[3].shape)
        # return final output
        return output

    def back_propagation(self,error:NDArray[float64]) :
        """
        Method goes backwards through layers and calculate errors needed to update weights
        Parameters:
            error (NDArray[float64]): Error
        """
        # starts at last index goes down to first
        # need to do this for each layer?
        # what is this function doing?
        for i in reversed(range(len(self.derivatives))):
            # print("length of derivatives",len(self.derivatives), "i",i)
            # print("error size:",error.shape)
            # work backwards with index from last layer to first

            # get previous layer
            # the output of last layer it is still + 1, but works out since there are 3 layers over all of weights between input hidden and out
            output = self.outputs[i+1]

            delta = error*sigmoid_derivitive(output)
            # is this because we are multipying error of 3 layers by 24*24 

            # print(f"delta {delta.shape}")

            # reshape array into 24 length and transpose
            # this reshape is causing errors, why is that
            # it is putting all 24 x 24 x 3 layers into 576 x 3
            # where it should be putting 24 x 72
            # print("delta shape",delta.shape)
            delta_fixed = delta.reshape(delta.shape[0],-1).T 

            # need to reshape so we can do dot product between layer output and delta fixed

            # current layer output
            layer_output = self.outputs[i]
            layer_output = layer_output.reshape(layer_output.shape[0],-1)
            # reshape to get column array for multiplication
            # print(f"delta {delta.shape} | c layer shape {layer_output.shape}")
            # print("bp output len:",len(self.outputs))
            # print("bp output Shape 1:",self.outputs[0].shape)
            # print("bp output Shape 2:",self.outputs[1].shape)
            # print("bp output Shape 3:",self.outputs[2].shape)
            # print("bp output Shape 4:",self.outputs[3].shape)

            # we are working out how much each weight contributed to the error
            # getting error as 24*24*3 being dot producted with 24*1 which is next layer, so we need to work out how for each layer we do just 1 in 24 array so its 24*24*1 24*1
            # erroring here because output[2] has a format of 24*3*24
            self.derivatives[i] = np.dot(layer_output,delta_fixed)

            error = np.dot(delta, self.weights[i].T)
            # update error for next layer of network

    def gradient_descent(self):
        """
        Method used to update weights based on derivatives calculated in back propagation
        """
        for i in range(len(self.weights)):
            # update the weight by adding derivative*learning_rate
            self.weights[i] += self.derivatives[i]*self.learning_rate
