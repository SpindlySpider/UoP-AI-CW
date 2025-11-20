from numpy.typing import NDArray
import graph_results
from numpy._core.numerictypes import float64
from neural_network import Neural_network
import input_data
import numpy as np
from error_funcs import mse
import optimiser
import curses

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
    # curses.filter()
    # curses.initscr()
    # x = curses.COLS
    # screen =  curses.newwin(15, x, 0, 0)
    # last_mse_error = 0

    loss_per_epoch = []
    for epoch in range(epochs):
        # shuffle data for each batch
        input_list,target_list = input_data.shuffle_data(input_list,target_list)
        mse_der_error:NDArray[float64] = np.array([])
        mse_loss: float = 0
        for b in range(0,len(input_list)-batch_size,batch_size):
            # feed forward b many times
            mse_der_error = np.array([])

            targets = target_list[b:b+batch_size]
            inputs = input_list[b:b+batch_size]
            predict = nn.feed_forward(inputs)

            mse_der_error =   predict.T - targets.T
            # print("mse before shape",mse_der_error.shape)
            mse_der_error =   mse_der_error.reshape(mse_der_error.shape[0],-1).T
            # print("mse after shape",mse_der_error.shape)
            # reshape to 1 x outputs



            mse_loss += mse(targets.T,predict.T)

            nn.back_propagation(mse_der_error)
            nn = optimiser.gradient_descent(nn)

        # print(np.average(mse_der_error))
        loss_per_epoch.append((mse_loss/(len(input_list) // batch_size))*100)
        #NOTE: let me know if the cli stuff is too messy, we can move the ncurses into its own function :)


        # needs to be mse_loss/number of batches in batch
        print("mean loss",loss_per_epoch[-1])

        # print("mean loss", np.average((mse_loss)*100))
        # print("mean loss", last_mse_error)

    #     percent = round((epoch/epochs)*40)
    #     status = f"|{percent*'#'}{(40-percent)*'-'}| epoch: {epoch}/{epochs} | mean loss: {np.average(mse_der_error)} |"
    #     screen.addstr(0,2,status)
    #     screen.refresh()
    # curses.endwin()

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
    predicts = []
    for input in input_list:
        predicts.append(nn.feed_forward(input))
    error = mse(target_list,predicts)
    print(f"tested nn on {len(input_list)} dataset |  MSE loss is: {error}")
