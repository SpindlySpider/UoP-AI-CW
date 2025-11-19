import serialize
import input_data
from training import *
from activation_functions import *
from neural_network import Neural_network


def main():
    # define hidden layer sizes
    hidden_layers = [24,48,48,24]
    # apply activation functions per layer
    # Using non linear functions for NN to find patterns, then linear to regress the result.
    # relu needs unactivated output of layer
    # activation_functions = [relu for _ in range(len(hidden_layers))] + [linear]
    activation_functions = [tanh,tanh,tanh,tanh,linear]
    learning_rate = 0.01
    nn = Neural_network(hidden_layers=hidden_layers,learning_rate=learning_rate,activation_functions=activation_functions)

    # get training data
    data = input_data.generate_train_test_data(0.80,500,1)
    train_in, train_out = data["training"]
    test_in, test_out = data["test"]

    # train and save resulting NN
    nn = train_NN(nn,train_in,train_out,100,1)
    serialize.save(nn)

    # test trained nn with unseen input data
    test_NN(nn,test_in,test_out)



if __name__ == "__main__":
    main()
