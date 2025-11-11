import numpy as np

# not using lambda functions here as pickle cannot serialize anonymous functions
def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_derivitive(x):return x*(1-x)

def linear(x): return x
def linear_derivative(x): return 1

def tanh(x): return (np.exp(2*x)-1) / (np.exp(2*x) + 1)
def tanh_derititive(x): return 1 - (tanh(x)**2)

# could add ReLu here and see how it does :) 

# used to map functions to their respective derivative function
ACTIVATION_DERITIVIVE_MAP = {
    sigmoid:sigmoid_derivitive,
    linear: linear_derivative,
    tanh: tanh_derititive
}
