from training import train_NN, generate_training_data
import numpy as np


def main():
    inputs,outputs = generate_training_data(1000)
    inputs,outputs = np.array(inputs), np.array(outputs)

    train_NN(inputs,outputs,10000,0.05)

if __name__ == "__main__":
    main()
