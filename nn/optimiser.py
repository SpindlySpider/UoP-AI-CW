from neural_network import Neural_network

def gradient_descent(nn:Neural_network) -> Neural_network:
    """
    Optimiser used to update weights based on derivatives calculated in back propagation.
    Parameters:
        nn (Neural_network): the neural network to perform gradient descent on.
    Returns:
        Neural network with updated weights and biases
    """
    for i in range(len(nn.weights)):
        nn.weights[i] =- (nn.derivatives[i]*nn.learning_rate)
        # nn.bias[i] =- (nn.delta[i]*nn.learning_rate)
    return nn

def adam(nn:Neural_network) -> Neural_network:
    """
    Optimiser which uses momentum and RMSprop to adjust learning rate during training
    """
    # https://www.geeksforgeeks.org/deep-learning/adam-optimizer/

    return nn
