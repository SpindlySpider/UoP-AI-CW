from neural_network import Neural_network

def train_NN(input:list[list[float]],target:list[list[float]],epochs:int = 100,learning_rate:float = 0.05):
    """
    Train neural network to predict joints
    Parameters:
        input (list[list[float]]): Each item in input corresponds to an output on the same index in target. This argument should be a list of size 24 joint angles.
        target (list[list[float]]): Each item in target corresponds to target of input. This should be a list of size 24 joints which would be the next frame in the gait.
        epochs (int): Number of cycles to train.
        learning_rate (float): Learning rate of NN
    """
    nn: Neural_network = Neural_network()
    for i in range(epochs):
        print(f"| Epoch: {i} |")
        # If using numpy arrays we can just directly take away target-predict 
        # e.g. array1 - array2 = error
        # should probbaly do MSE. and then feed it to back prop and GD

    # after training serialize result so we can load for later.
