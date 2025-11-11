from numpy.typing import NDArray
from numpy._core.numerictypes import float64
from neural_network import Neural_network
import numpy as np
from error_funcs import mse
import optimiser

def train_NN(nn:Neural_network,input_list:NDArray[float64],target_list:NDArray[float64],epochs:int, batch_size:int) -> Neural_network:
    """
    Train neural network to predict next pose
    Parameters:
        nn (Neural_network): Neural network to train
        input_list (NDArray[float64]): Each item in input corresponds to an output on the same index in target. This argument should be a list of size 24 joint angles.
        target_list (NDArray[float64]): Each item in target corresponds to target of input. This should be a list of size 24 joints which would be the next frame in the gait.
        epochs (int): How many epochs to train for.
        batch_size (int): Size of batch per epoch, before performing BP
    Returns:
        Trained neural network
    """

    # useful doc: https://www.geeksforgeeks.org/deep-learning/batch-size-in-neural-network/
    for epoch in range(epochs):
        for b in range(0,len(input_list)-batch_size,batch_size):
            # make sure that we have consistant element sizes - batch size from max
            mse_error:NDArray[float64] = np.array([])
            # feed forward b many times
            batch_idxs = [idx for idx in range(b,b+batch_size)]
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


def test_NN(nn:Neural_network,input_list:NDArray[float64],target_list:NDArray[float64]):
    """
    Test trained neural network on input data to find MSE
    Parameters:
        nn (Neural_network): Neural network to train
        input_list (NDArray[float64]): Each item in input corresponds to an output on the same index in target. This argument should be a list of size 24 joint angles.
        target_list (NDArray[float64]): Each item in target corresponds to target of input. This should be a list of size 24 joints which would be the next frame in the gait.
    """
    predicts = []
    for input in input_list:
        predicts.append(nn.feed_forward(input))
    error = mse(target_list,predicts)
    print(f"tested nn on {len(input_list)} dataset |  MSE loss is: {error}")
