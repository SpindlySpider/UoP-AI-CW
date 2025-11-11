import serialize
from training import train_NN, generate_training_data
from activation_functions import *
from neural_network import Neural_network
import numpy as np


def main():
    inputs,outputs = generate_training_data(1000)
    inputs,outputs = np.array(inputs), np.array(outputs)
    hidden_layers = [24,24,48,48,24]
    # apply activation functions per layer
    activation_functions = [tanh,tanh,tanh,tanh,tanh,linear]
    learning_rate = 0.01
    nn = Neural_network(hidden_layers=hidden_layers,learning_rate=learning_rate,activation_functions=activation_functions)

    nn = train_NN(nn,inputs,outputs,800,1)
    serialize.save(nn)

if __name__ == "__main__":
    main()
