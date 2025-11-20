from neural_network import Neural_network

def gradient_descent(nn:Neural_network,verbose:bool = False) -> Neural_network:
    """
    Optimiser used to update weights based on derivatives calculated in back propagation.
    Parameters:
        nn (Neural_network): the neural network to perform gradient descent on.
    Returns:
        Neural network with updated weights and biases
    """
    # verbose = True
    verbose = False
    if verbose: print("--=== start GD ===--")
    for i in range(len(nn.weights)):
        if verbose: print(f"index {i}/{len(nn.weights)-1} weights")
        # is this doing the correct weights?
        # setting all weights to the same thing ?  must be an issue with the derititives
        if verbose: print(f"old weights {nn.weights[i][0][0:2]} | bias {nn.bias[i][0][0:2]}")
        nn.weights[i] = nn.weights[i] - nn.derivatives[i]*nn.learning_rate
        nn.bias[i] = nn.bias[i] - nn.delta[i]*nn.learning_rate
        if verbose: print(f"new weights {nn.weights[i][0][0:2]} | bias {nn.bias[i][0][0:2]}")

    if verbose: print("--=== end GD ===--")
    return nn

def adam(nn:Neural_network) -> Neural_network:
    """
    Optimiser which uses momentum and RMSprop to adjust learning rate during training
    """
    # https://www.geeksforgeeks.org/deep-learning/adam-optimizer/

    return nn
