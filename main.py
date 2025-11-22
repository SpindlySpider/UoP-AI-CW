import nn.main as nn
import nn.load_and_predict as predict
import random
import sys
import ga.main as ga
from utils import *


def main():
    options = {
        "prompt":"What would you like to run?:",
        "options":["genetic algorithm","neural network","exit"]
    }
    choice = get_choice(options)
    if choice == "exit":
        sys.exit(0)

    if choice == "genetic algorithm":
        # get defaults and unpack configuration.
        defaults = get_defaults(ga.defaults)
        ga.main(**defaults)
    else:
        #TODO: ask if user wants to train a new NN, or load a NN and train? or predict with existing NN
        options = {
            "prompt":"Would you like to train or predict using existing model?:",
            "options":["train","predict","exit"]
        }
        choice = get_choice(options)
        if choice == "exit":
            sys.exit(0)
        elif choice == "train":
            defaults = get_defaults(nn.defaults)
            nn.main(**defaults)
        else:
            #TODO: allow use to input list
            options = {
                "prompt":"Random input or enter your own",
                "options":["random","manual input"]
            }
            choice = get_choice(options)
            input = []

            if choice == "random":
                input = [random.randint(-100,100) for _ in range(24)]
            else:
                input = handle_lists(float,24)

            predict_defaults = {
                "nn_path":"./nn.pickle",
                "output_file_name":"./predict_results.txt",
                "input":input,
                "gait_length":100
            }
            defaults = get_defaults(predict_defaults)
            #TODO: user need to be able to input nn location to load, and number of next predicts e.g. gait length, where to save to
            predict.load_and_predict(**defaults)
            # predict using model
    print("="*20)



if __name__ == "__main__":
    main()
