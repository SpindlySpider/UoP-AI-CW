"""
Module used to serialise NN class load and dump
"""
import pickle
from pathlib import Path

from nn.neural_network import Neural_network

def save(nn:Neural_network,out:str="nn.pickle"):
    """
    Saves Neural network model using pickle.
    Parameters:
        nn (Neural_network): Neural network to save.
        out (str): Name of file to save to, defaults to "nn.pickle"
    """

    file = open(out,"wb")
    pickle.dump(nn,file)
    file.close()

def load(file_name:str="nn.pickle") -> Neural_network:
    """
    Loads Neural network model using pickle.
    Parameters:
        file_name (str): Name of file to load, defaults to "nn.pickle"
    Returns:
        Neural network from file.
    """

    try:
        file = open(file_name,"rb")
        nn = pickle.load(file)
        file.close()
        return nn
    except:
        print(f"error could not load {file_name}: does it exist?")
