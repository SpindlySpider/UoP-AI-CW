import serialize
import input_data
from training import *
from activation_functions import *
from neural_network import Neural_network


def main():
    # define hidden layer sizes
    hidden_layers = [128,64,32]
    # apply activation functions per layer
    activation_functions = [sigmoid for _ in range(len(hidden_layers))] + [sigmoid]
    learning_rate = 0.01
    nn = Neural_network(hidden_layers=hidden_layers,learning_rate=learning_rate,activation_functions=activation_functions)

    # get training data
    data = input_data.generate_train_test_data(0.95,40,700)
    train_in, train_out = data["training"]
    test_in, test_out = data["test"]

    # train and save resulting NN
    nn = train_NN(nn,train_in,train_out,100,5,False)
    serialize.save(nn)

    # test trained nn with unseen input data
    test_NN(nn,test_in,test_out)


if __name__ == "__main__":
    main()
