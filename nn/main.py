from training import train_NN, generate_training_data
from neural_network import Neural_network
import numpy as np


def main():
    inputs,outputs = generate_training_data(1000)
    inputs,outputs = np.array(inputs), np.array(outputs)
    hidden_layers = [24,24,48,92,92,48,48,24]
    learning_rate = 0.01
    nn = Neural_network(hidden_layers=hidden_layers,learning_rate=learning_rate)

    nn = train_NN(nn,inputs,outputs,1000,5)

if __name__ == "__main__":
    main()
