import numpy as np

sigmoid = lambda x: 1/(1+np.exp(-x))
sigmoid_derivitive = lambda x: x*(1-x)

linear = lambda x : x
linear_derivative = lambda x: 1

tanh = lambda x : (np.exp(2*x)-1) / (np.exp(2*x) + 1)
tanh_derititive = lambda x : 1 - (tanh(x)**2)

# could add ReLu here and see how it does :) 

# used to map functions to their respective derivative function
ACTIVATION_DERITIVIVE_MAP = {
    sigmoid:sigmoid_derivitive,
    linear: linear_derivative,
    tanh: tanh_derititive
}
