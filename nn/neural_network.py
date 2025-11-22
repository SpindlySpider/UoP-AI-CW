import numpy as np
from nn.activation_functions import *
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
    """LEGACY MODULE REMOVED

    This module previously contained a NumPy-based Neural_network implementation.
    It has been removed in favor of a PyTorch implementation at `nn.torch_model.TorchNet`.

    If you need the legacy implementation back, restore it from version control.
    """

    raise ImportError("nn.neural_network has been removed. Use nn.torch_model.TorchNet instead.")


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
            error (NDArray[float64]): Error
        """
        if verbose: print("========starting back prop========")
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
