import numpy as np

def sigmoid(x): #Sigmoid activation function
    y=1.0/(1+np.exp(-x))
    return y

def sigmoid_derivitive(x):
    return x*(1.0-x)

# could add ReLu here and see how it does :) 
