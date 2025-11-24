import sys
from pathlib import Path

# Add parent directories to access ga module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Use relative imports within the package
import serialize as serialize
import input_data as input_data
from training import *
from activation_functions import *
from neural_network import Neural_network
from optimiser import *


# Define number of layers and neurons per layer
hidden_layers:list[int] = [128,64,32]
# Learning rate for NN training
learning_rate:float = 0.01
# File to save trained NN to
nn_save:str = "nn.pickle"
# Ratio of data to use for training vs testing
training_data_ratio:float = 0.95
# Length of gait data to generate
data_gait_length: int = 300
# Number of gait variations to generate
gait_variations:int = 700

# Size of training batches
training_batch_size:int = 1
# Number of training epochs
epochs:int = 100
opt:str = "gradient_descent"

# Default parameters for main function
defaults = {
    "hidden_layers":hidden_layers,
    "learning_rate":learning_rate,
    "nn_save":nn_save,
    "training_data_ratio":training_data_ratio,
    "data_gait_length":data_gait_length,
    "gait_variations": gait_variations,
    "training_batch_size":training_batch_size,
    "epochs":epochs,
    "optimiser": opt,
}


def main(hidden_layers=hidden_layers,learning_rate=learning_rate,nn_save=nn_save,training_data_ratio=training_data_ratio,data_gait_length=data_gait_length,gait_variations=gait_variations,training_batch_size=training_batch_size,epochs=epochs,optimiser=opt):
    """Main function to create, train, and test a neural network for gait generation.
    Args:
        hidden_layers (list[int], optional): List defining the number of neurons in each hidden layer. Defaults to hidden_layers.
        learning_rate (float, optional): Learning rate for training the neural network. Defaults to learning_rate.
        nn_save (str, optional): Filename to save the trained neural network. Defaults to nn_save.
        training_data_ratio (float, optional): Ratio of data to use for training vs testing. Defaults to training_data_ratio.
        data_gait_length (int, optional): Length of gait data to generate. Defaults to data_gait_length.
        gait_variations (int, optional): Number of gait variations to generate. Defaults to gait_variations.
        training_batch_size (int, optional): Size of training batches. Defaults to training_batch_size.
        epochs (int, optional): Number of training epochs. Defaults to epochs.                          
    """
    # apply activation functions per layer
    activation_functions = [sigmoid for _ in range(len(hidden_layers) + 1)]
    # create NN
    nn = Neural_network(hidden_layers=hidden_layers,learning_rate=learning_rate,activation_functions=activation_functions)

    # get training data
    data = input_data.generate_train_test_data(training_data_ratio,data_gait_length,gait_variations)
    # define training set
    train_in, train_out = data["training"]
    # define testing set
    test_in, test_out = data["test"]

    # train and save resulting NN
    nn = train_NN(nn,train_in,train_out,epochs,training_batch_size, optimiser)
    serialize.save(nn,nn_save)

    # test trained nn with unseen input data
    test_NN(nn,test_in,test_out)


if __name__ == "__main__":
    main()
