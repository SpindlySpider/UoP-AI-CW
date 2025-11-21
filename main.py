import nn.main as nn
import sys
import ga.main as ga
from utils import *


def main():
    #NOTE: maybe use dict instead as it would allow for great specification of choices and following choices.
    # also would be easier to visulise choice tree rather than a bunch of if statements IMO
    options = {
        "prompt":"What would you like to run?:",
        "options":["genetic algorithm","neural network","exit"]
    }
    choice = get_choice(options)
    if choice == "exit":
        sys.exit(0)

    #TODO: print default configuration and let user modify if they want for both,
    # display default values for all and let user modify as required + verify its not bad value
    # for example: 
    #   ga: generations (if we still have that), population size, gait length, crossover rate, mutation rate, graphs
    #   nn: hidden layer list, learning rate, activation funcs?, how much training data to gen, graphs,
    # also make the if statement less ugly lol


    if choice == "genetic algorithm":
        defaults = get_defaults(ga.defaults)
        ga.main(**defaults)
    else:
        #TODO: ask if user wants to train a new NN, or load a NN and train? or predict with existing NN
        nn.main()
    print("="*20)



if __name__ == "__main__":
    main()
