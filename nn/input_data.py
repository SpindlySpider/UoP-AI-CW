import sys
from nn.load_and_predict import normalize
from pathlib import Path
from numpy.typing import NDArray
import numpy as np
import random

# Add parent directories to access ga module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ga.custom_types import Gait
from ga.target_sol import produce_target
from ga.custom_types import *



def get_training_data() -> tuple[Gait,Gait]:
    """
    Loads Gait from GA results and splits data into input(N) and then output(N+1)
    Returns:
        tuple with input data and output data, same indexes correspond to input and labeled output
    """

    # get target gait from GA results
    total_gait:Gait = load_gait()
    gait_length:int = len(total_gait)

    # normalize data between 0 and 1
    total_gait = normalize(np.array(total_gait))

    # split into input and output data
    inputs:list[list[float]] = []
    outputs:list[list[float]] = []

    # generate input output pairs
    # gait_length -1 because output is N+1
    for i in range(gait_length):
        inputs.append(total_gait[i])
        if i < gait_length - 1:
            outputs.append(total_gait[i+1])
        else:
            # Wrap around: last frame predicts first frame (cyclic gait)
            outputs.append(total_gait[0])
    return (inputs,outputs)


def shuffle_data(input:list,label:list) -> tuple[NDArray,NDArray]:
    """
    Shuffles input and label data in the same order
    Parameters:
        input (list): Input data
        label (list): Label data
    Returns:
        Shuffled input and label data as numpy arrays
    """
    # generate permutated indexes
    permutated_idxs:NDArray = np.random.permutation(len(input))
    # apply to input and label
    shuffled_in, shuffled_label = np.array(input)[permutated_idxs],np.array(label)[permutated_idxs]
    return (shuffled_in,shuffled_label)

def seperate_train_test_data(train_ratio:float = 0.8) -> dict[str:tuple[Gait,Gait]]:
    """
    Separates GA results into training and test data to train NN
    Parameters:
        train_ratio (float): Value between 0 - 1, to specify the ratio of data points to be used as training and test data, this value corresponds to the training data amount. so a 0.8 will result in 80% of the data being used for training.
    Returns:
        A dictionary which contains the training data and test data. The values are (input,label). Keys are "training","test"
    """

    # initialize empty lists
    input: Gait = []
    label: Gait = []

    input,label = get_training_data()


    # determine slicing index
    data_points = len(input)
    slice_idx:int = round((data_points) * train_ratio)
    # shuffle data
    input,label = shuffle_data(input,label)

    # create data dictionary to hold training and test data
    data:dict[str: tuple[Gait,Gait]] = {"training":(),"test":()}
    # slice and convert to numpy array
    data["training"] = (np.array(input[0:slice_idx]), np.array(label[0:slice_idx]))
    data["test"] = (np.array(input[slice_idx:]), np.array(label[slice_idx:]))

    return data

def load_gait(filepath:str = "./ga/ga_results.txt") -> Gait:
    """
    Loads gait from specified filepath
    Parameters:
        filepath (str): filepath where ga is located
    """

    gait:Gait = []
    try:
        file = open(filepath,"r")
        for line in file:
            # split line by , and convert to float
            frame = [float(i) for i in line.split(",")]
            gait.append(frame)
    except:
        print(f"Unable to load gait from {filepath} are you sure it exists?")
        sys.exit()
    return gait
