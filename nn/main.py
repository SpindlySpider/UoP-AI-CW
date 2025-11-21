import nn.serialize as serialize
import nn.input_data as input_data
from nn.training import *
from nn.activation_functions import *
from nn.neural_network import Neural_network

#TODO: fix hidden layers modifiction and also it would be nice if we could have a activation function selection?

#BUG: hidden layers do not get modified correctly in CLI since it need a list input
hidden_layers:list[int] = [128,64,32]
learning_rate:float = 0.01
nn_save:str = "nn.pickle"
# 0.95 - 1
training_data_ratio:float = 0.95
data_gait_length: int = 40
gait_variations:int = 700

training_batch_size:int = 5
epochs:int = 100


defaults = {
    "hidden_layers":hidden_layers,
    "learning_rate":learning_rate,
    "nn_save":nn_save,
    "training_data_ratio":training_data_ratio,
    "data_gait_length":data_gait_length,
    "gait_variations": gait_variations,
    "training_batch_size":training_batch_size,
    "epochs":epochs
}


def main(hidden_layers=hidden_layers,learning_rate=learning_rate,nn_save=nn_save,training_data_ratio=training_data_ratio,data_gait_length=data_gait_length,gait_variations=gait_variations,training_batch_size=training_batch_size,epochs=epochs):
    # apply activation functions per layer
    activation_functions = [sigmoid for _ in range(len(hidden_layers) + 1)]
    nn = Neural_network(hidden_layers=hidden_layers,learning_rate=learning_rate,activation_functions=activation_functions)

    # get training data
    data = input_data.generate_train_test_data(training_data_ratio,data_gait_length,gait_variations)
    train_in, train_out = data["training"]
    test_in, test_out = data["test"]

    # train and save resulting NN
    nn = train_NN(nn,train_in,train_out,epochs,training_batch_size,False)
    serialize.save(nn,nn_save)

    # test trained nn with unseen input data
    test_NN(nn,test_in,test_out)


if __name__ == "__main__":
    main()
