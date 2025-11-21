from neural_network import Neural_network
import numpy as np

def gradient_descent(nn:Neural_network) -> Neural_network:
    """
    Optimiser used to update weights based on derivatives calculated in back propagation.
    Parameters:
        nn (Neural_network): the neural network to perform gradient descent on.
    Returns:
        Neural network with updated weights and biases
    """
    for i in range(len(nn.weights)):
        nn.weights[i] = nn.weights[i] - nn.derivatives[i]*nn.learning_rate
        # np.average for averaging bias over batches
        nn.bias[i] = nn.bias[i] - np.average(nn.delta[i],axis=0)*nn.learning_rate
    return nn

def adam(nn:Neural_network) -> Neural_network:
    """
    Optimiser which uses momentum and RMSprop to adjust learning rate during training
    """
    # https://www.geeksforgeeks.org/deep-learning/adam-optimizer/

    return nn
