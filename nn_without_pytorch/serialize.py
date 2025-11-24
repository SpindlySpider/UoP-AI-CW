"""
Module used to serilize NN class load and dump
"""
import pickle
from pathlib import Path

from nn_without_pytorch.neural_network import Neural_network

def save(nn:Neural_network,out:str="nn.pickle"):
    """
    Saves Neural network model using pickle.
    Parameters:
        nn (Neural_network): Neural network to save.
        out (str): Name of file to save to, defaults to "nn.pickle"
    """
    # Ensure the path is relative to nn_without_pytorch/ folder if just a filename
    if not Path(out).is_absolute() and not str(out).startswith('.'):
        out = str(Path(__file__).parent / out)
    
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
    # Ensure the path is relative to nn_without_pytorch/ folder if just a filename
    if not Path(file_name).is_absolute() and not str(file_name).startswith('.'):
        file_name = str(Path(__file__).parent / file_name)
    
    try:
        file = open(file_name,"rb")
        nn = pickle.load(file)
        file.close()
        return nn
    except:
        print(f"error could not load {file_name}: does it exist?")
