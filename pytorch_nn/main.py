import sys
from pathlib import Path

# Add parent directory to access nn_without_pytorch
sys.path.insert(0, str(Path(__file__).parent.parent))

import nn.input_data as input_data
from pytorch_nn.torch_model import TorchNet
from pytorch_nn.torch_training import train_torch, test_torch
from pytorch_nn.serialize import save_torch  

# Define number of layers and neurons per layer
hidden_layers: list[int] = [128, 64, 32]
# Learning rate for NN training
learning_rate: float = 0.01
# File to save trained NN to
nn_save: str = "nn.pth"
# Ratio of data to use for training vs testing
training_data_ratio: float = 0.95
# Length of gait data to generate
data_gait_length: int = 300
# Number of gait variations to generate
gait_variations: int = 700

# Size of training batches
training_batch_size: int = 1
# Number of training epochs
epochs: int = 100
opt: str = "sgd"

# Default parameters for main function
defaults = {
    "hidden_layers": hidden_layers,
    "learning_rate": learning_rate,
    "nn_save": nn_save,
    "training_data_ratio": training_data_ratio,
    "data_gait_length": data_gait_length,
    "gait_variations": gait_variations,
    "training_batch_size": training_batch_size,
    "epochs": epochs,
    "optimiser": opt,
}


def main(hidden_layers=hidden_layers, learning_rate=learning_rate, nn_save=nn_save, training_data_ratio=training_data_ratio, data_gait_length=data_gait_length, gait_variations=gait_variations, training_batch_size=training_batch_size, epochs=epochs, optimiser=opt):
    """Main function to create, train, and test a neural network for gait generation.
    Parameters:
        hidden_layers (list[int], optional): List defining the number of neurons in each hidden layer. Defaults to hidden_layers.
        learning_rate (float, optional): Learning rate for training the neural network. Defaults to learning_rate.
        nn_save (str, optional): Filename to save the trained neural network. Defaults to nn_save.
        training_data_ratio (float, optional): Ratio of data to use for training vs testing. Defaults to training_data_ratio.
        data_gait_length (int, optional): Length of gait data to generate. Defaults to data_gait_length.
        gait_variations (int, optional): Number of gait variations to generate. Defaults to gait_variations.
        training_batch_size (int, optional): Size of training batches. Defaults to training_batch_size.
        epochs (int, optional): Number of training epochs. Defaults to epochs.
        optimiser (str, optional): Optimizer to use for training. Defaults to opt.
    """
    # model with sigmoid activation for all layers (including output)
    model = TorchNet(input_size=24, hidden_sizes=hidden_layers, output_size=24, activation='sigmoid')

    # get training data
    data = input_data.generate_train_test_data(training_data_ratio, data_gait_length, gait_variations)
    # define training set
    train_in, train_out = data["training"]
    # define testing set
    test_in, test_out = data["test"]

    # train and save resulting model
    model, losses = train_torch(model, train_in, train_out, epochs=epochs, batch_size=training_batch_size, lr=learning_rate, optimizer_name=optimiser)
    save_torch(model, out=nn_save)

    # test trained model with unseen input data
    test_torch(model, test_in, test_out)


if __name__ == "__main__":
    main()
