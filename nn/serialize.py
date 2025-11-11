"""
Module used to serilize NN class load and dump
"""

from neural_network import Neural_network
import pickle

def save(nn:Neural_network,out:str="nn"):
    file = open(out,"wb")
    pickle.dump(nn,file)
    file.close()

def load(file_name:str="nn") -> Neural_network:
    file = open(file_name,"rb")
    nn = pickle.load(file)
    file.close()
    return nn
