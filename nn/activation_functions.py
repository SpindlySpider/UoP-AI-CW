import numpy as np




def sigmoid(x): #Sigmoid activation function
    y=1.0/(1+np.exp(-x))
    return y

def sigmoid_derivitive(x):
    return x*(1.0-x)

# could add ReLu here and see how it does :) 

def linear(x):
    return x

def linear_derivative(x):
    return 1

def tanh(x):
    return np.tanh(x)

def tanh_derititive(x):
    return 1 - (np.tanh(x)**2)

# used to track what derititive function to use
ACTIVATION_DERITIVIVE_MAP = {
    sigmoid:sigmoid_derivitive,
    linear: linear_derivative,
    tanh: tanh_derititive
}
