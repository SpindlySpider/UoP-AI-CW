from numpy.typing import NDArray
from numpy._core.numerictypes import float64
import numpy as np

import nn.graph_results as graph_results
import nn.input_data as input_data
from nn.neural_network import Neural_network
from nn.error_funcs import mse
import nn.optimiser as opt

def train_NN(nn:Neural_network,input_list:NDArray[float64],target_list:NDArray[float64],epochs:int, batch_size:int, optimiser:str) -> Neural_network:
    """
    Train neural network to predict next pose
    Parameters:
        nn (Neural_network): Neural network to train
        input_list (NDArray[float64]): Each item in input corresponds to an output on the same index in target. This argument should be a list of size 24 joint angles.
        target_list (NDArray[float64]): Each item in target corresponds to target of input. This should be a list of size 24 joints which would be the next frame in the gait.
        epochs (int): How many epochs to train for.
        batch_size (int): Size of batch per epoch, before performing BP
        optimiser (str): Optimiser to use for training ("gradient_descent" or "adam")
    Returns:
        Trained neural network
    """
    # useful doc: https://www.geeksforgeeks.org/deep-learning/batch-size-in-neural-network/
    # keep track of loss per epoch for graphing
    loss_per_epoch:list[float] = []
    # train for specified epochs
    for epoch in range(epochs):
        # shuffle data for each batch so NN doesn't learn order
        input_list,target_list = input_data.shuffle_data(input_list,target_list)
        # keep track of error per batch
        mse_der_error:NDArray[float64] = np.array([])
        # total loss for epoch
        mse_loss: float = 0
        # iterate over batches
        for b in range(0,len(input_list)-batch_size,batch_size):
            # feed forward b many times
            mse_der_error = np.array([])

            # get targets and inputs for batch
            targets = target_list[b:b+batch_size]
            inputs = input_list[b:b+batch_size]

            # feed forward batch inputs
            predict = nn.feed_forward(inputs)

            # transpose so we are comparing each output e.g. 24 x batch size rather than batch size x 24
            mse_der_error =   predict.T - targets.T
            # reshape to batch size x outputs
            mse_der_error =   mse_der_error.reshape(mse_der_error.shape[0],-1).T

            # loss for plotting
            mse_loss += mse(targets.T,predict.T)

            # back propogate and gradient descent step
            nn.back_propagation(mse_der_error)
            if optimiser == "gradient_descent":
                nn = opt.gradient_descent(nn)
            elif optimiser == "adam":
                nn = opt.adam(nn)

        # average loss for epoch
        loss_per_epoch.append(mse_loss/((len(input_list) // batch_size)))

        print(f"mean loss {loss_per_epoch[-1]} | epoch: {epoch}")
    graph_results.plot_loss_graph(loss_per_epoch,epochs,batch_size)
    return nn


def test_NN(nn:Neural_network,input_list:NDArray[float64],target_list:NDArray[float64]):
    """
    Test trained neural network on input data to find MSE
    Parameters:
        nn (Neural_network): Neural network to train
        input_list (NDArray[float64]): Each item in input corresponds to an output on the same index in target. This argument should be a list of size 24 joint angles.
        target_list (NDArray[float64]): Each item in target corresponds to target of input. This should be a list of size 24 joints which would be the next frame in the gait.
    """
    # collect predictions
    predicts = []
    # feed forward each input
    for input in input_list:
        # get prediction
        predicts.append(nn.feed_forward(input))
    # convert predictions to numpy array
    predicts = np.array(predicts)
    # reshape predicts as they retain batch size
    predicts = predicts.reshape(predicts.shape[0],-1)
    # calculate mse
    error = mse(target_list,predicts)
    print(f"tested nn on {len(input_list)} dataset |  MSE loss is: {error}")
