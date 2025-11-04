from numpy.typing import NDArray
from numpy._core.numerictypes import float64
from neural_network import Neural_network
from error_funcs import mse
from target_sol import produce_target
from custom_types import Gait

def train_NN(input_list:NDArray[float64],target_list:NDArray[float64],epochs:int = 100,learning_rate:float = 0.05):
    """
    Train neural network to predict joints
    Parameters:
        input_list (list[list[float]]): Each item in input corresponds to an output on the same index in target. This argument should be a list of size 24 joint angles.
        target_list (list[list[float]]): Each item in target corresponds to target of input. This should be a list of size 24 joints which would be the next frame in the gait.
        epochs (int): Number of cycles to train.
        learning_rate (float): Learning rate of NN
    """
    hidden_layers: list[int] = [24 for i in range(200)]
    nn: Neural_network = Neural_network(learning_rate=learning_rate,hidden_layers=hidden_layers)

    for i in range(epochs):
        mse_error=0
        for j, input in enumerate(input_list):
            # these should be [1,...,24]
            target = target_list[j]
            output = nn.feed_forward(input)
            # this can be improved
            error = (target - output)**2
            nn.back_propagation(error)
            nn.gradient_descent()

            mse_error = mse(target,output)
        print(f"| Epoch: {i} | MSE error {mse_error}")
    # maybe could pass in NN allow for modular design
    # after training serialize result so we can load for later.


def generate_training_data(gait_length:int = 500) -> tuple[list[list[float]],list[list[float]]]:
    """
    Generates training data to feed the NN
    Generates a gait, then splits input(N) and then output(N+1)
    Parameters:
        gait_length (int): The number of frames the gait will be generated for
    Returns:
        tuple with input data and output data, same indexes correspond to input and labeled output
    """
    total_gait:Gait = produce_target(gait_length,0.2,20,45,25)
    inputs:list[list[float]] = []
    outputs:list[list[float]] = []
    # label data
    for i in range(gait_length -1):
        # need to be -1 since output is N+1
        inputs.append(total_gait[i])
        outputs.append(total_gait[i+1])
    return (inputs,outputs)
