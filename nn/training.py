from numpy.typing import NDArray
from numpy._core.numerictypes import float64
from neural_network import Neural_network
import numpy as np
from error_funcs import mse
from target_sol import produce_target
from custom_types import Gait
import optimiser

def train_NN(nn:Neural_network,input_list:NDArray[float64],target_list:NDArray[float64],epochs:int, batch_size:int) -> Neural_network:
    """
    Train neural network to predict next pose
    Parameters:
        nn (Neural_network): Neural network to train
        input_list (list[list[float]]): Each item in input corresponds to an output on the same index in target. This argument should be a list of size 24 joint angles.
        target_list (list[list[float]]): Each item in target corresponds to target of input. This should be a list of size 24 joints which would be the next frame in the gait.
    Returns:
        Trained neural network
    """

    # batches mean how many times you feed forward before performing bp per epoch
    # useful doc: https://www.geeksforgeeks.org/deep-learning/batch-size-in-neural-network/
    for epoch in range(epochs):
        for b in range(0,len(input_list)-batch_size,batch_size):
            # make sure that we have consistant element sizes - batch size from max
            mse_error:NDArray[float64] = np.array([])
            # feed forward b many times 
            # sample randomly b_size and run through nn
            # has to be a better way
            batch_idxs = [idx for idx in range(b,b+batch_size)]
            # this should iterate through chosen batch inputs

            # generates list of errors for output, each of the 24 outputs
            # e.g. [1,2,3,4.5,...]
            # so list of errors should be 1 list with all of them averaged
            predicts: list = []
            for i in batch_idxs:
                # get all predictions for this batch
                input = input_list[i]
                predicts.append(nn.feed_forward(input))
            # transform predict and target from batch_sizex24 to 24xbatch_size
            targets = target_list[b:b+batch_size].T
            predict = np.array(predicts).T
            for output_idx in range(targets.shape[0]):
                # calculate error for each output (24)
                # we want to pass in each row of outputs compared to targets
                # so we transpose the numpy array so instead of batch_size x outputs (24)
                # it is instead outputs (24) x batch size, then we can do MSE for each row and get a mean value for each output
                mse_error = np.append(mse_error,mse(targets[output_idx],predict[output_idx]))
            nn.back_propagation(mse_error)
            nn = optimiser.gradient_descent(nn)
        print(f"epoch {epoch} | mean loss {np.average(mse_error)}")
    return nn


def generate_training_data(gait_length:int = 500) -> tuple[Gait,Gait]:
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
