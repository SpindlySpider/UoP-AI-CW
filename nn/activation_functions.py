import numpy as np

lower = -50
upper = 30
# not using lambda functions here as pickle cannot serialize anonymous functions
def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_derivitive(x):return x*(1-x)

def linear(x): return np.clip(x,min=lower,max=upper)
def linear_derivative(x): return 1

def tanh(x):
    x = np.clip(x,lower,upper)
    return (np.exp(2*x)-1) / (np.exp(2*x) + 1)
def tanh_derititive(x):
    x = np.clip(x,lower,upper)
    return 1 - (tanh(x)**2)

def relu(x): return np.maximum(x,0)
def relu_derititive(x): return np.where(x > 0,1, 0)

# could add ReLu here and see how it does :) 

# used to map functions to their respective derivative function
ACTIVATION_DERITIVIVE_MAP = {
    sigmoid:sigmoid_derivitive,
    linear: linear_derivative,
    tanh: tanh_derititive,
    relu: relu_derititive
}
