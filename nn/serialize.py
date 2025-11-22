"""
Module used to serilize NN class load and dump
"""
import sys
from pathlib import Path
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from nn.neural_network import Neural_network
except ImportError:
    from neural_network import Neural_network

def save(nn:Neural_network,out:str="nn.pickle"):
    """
    Saves Neural network model using pickle.
    Parameters:
        nn (Neural_network): Neural network to save.
        out (str): Name of file to save to, defaults to "nn"
    """
    file = open(out,"wb")
    pickle.dump(nn,file)
    file.close()

def load(file_name:str="nn.pickle") -> Neural_network:
    """
    Loads Neural network model using pickle.
    Parameters:
        file_name (str): Name of file to load, defaults to "nn"
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
