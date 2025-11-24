import numpy as np
from numpy._core.numerictypes import float64
from numpy._typing import NDArray

def mse(target_list:NDArray[float64],predict_list:NDArray[float64]) -> float:
    """
    Calculates the mean squared error
    Parameters:
        target_list (list): List of target values
        predict_list (list): List of predict values
    Returns:
        float specifying error
    """

    # convert to target and predict arrays into numpy arrays
    t_list = np.array(target_list)
    p_list = np.array(predict_list)
    # calculate mean squared error
    error_list = (t_list - p_list)**2
    # return np.average(error_list,axis=1)
    return np.average(error_list)
