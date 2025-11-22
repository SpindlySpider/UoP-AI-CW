from numpy.typing import NDArray
import nn.graph_results as graph_results
from numpy._core.numerictypes import float64
from nn.neural_network import Neural_network
import nn.input_data as input_data
import numpy as np
from nn.error_funcs import mse
import nn.optimiser as optimiser
import curses

def train_NN(nn:Neural_network,input_list:NDArray[float64],target_list:NDArray[float64],epochs:int, batch_size:int, curses_enabled:bool = False) -> Neural_network:
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
    if curses_enabled:
        curses.filter()
        curses.initscr()
        x = curses.COLS
        screen =  curses.newwin(15, x, 0, 0)

    loss_per_epoch = []
    for epoch in range(epochs):
        # shuffle data for each batch so NN doesn't learn order
        input_list,target_list = input_data.shuffle_data(input_list,target_list)
        mse_der_error:NDArray[float64] = np.array([])
        mse_loss: float = 0
        for b in range(0,len(input_list)-batch_size,batch_size):
            # feed forward b many times
            mse_der_error = np.array([])

            targets = target_list[b:b+batch_size]
            """LEGACY MODULE REMOVED

            This module provided NumPy-based training utilities and has been replaced
            by `nn.torch_training.train_torch`. Importing this module will raise an
            ImportError to make the migration explicit.
            """

            raise ImportError("nn.training has been removed. Use nn.torch_training.train_torch instead.")

