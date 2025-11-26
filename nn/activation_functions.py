import numpy as np

# clip is here so that values do not become too small or big for exp
lower = -50
upper = 30

# not using lambda functions here as pickle cannot serialize anonymous functions
# activation function sigmoid and its derivative
def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_derivitive(x):return x*(1-x)

# linear activation function and its derivative
def linear(x): return np.clip(x,min=lower,max=upper)
def linear_derivative(x): return 1

# activation function tanh and its derivative
def tanh(x):
    x = np.clip(x,lower,upper)
    return (np.exp(2*x)-1) / (np.exp(2*x) + 1)
def tanh_derititive(x):
    x = np.clip(x,lower,upper)
    return 1 - (tanh(x)**2)


# activation function ReLU and its derivative
def relu(x): return np.maximum(x,0)
def relu_derititive(x): return np.where(x > 0,1, 0)

# used to map functions to their respective derivative function
ACTIVATION_DERITIVIVE_MAP = {
    sigmoid:sigmoid_derivitive,
    linear: linear_derivative,
    tanh: tanh_derititive,
    relu: relu_derititive
}
