import sys
from pathlib import Path

import nn.main as nn
import nn.load_and_predict as predict
import pytorch_nn.main as pytorch_nn
import pytorch_nn.load_and_predict as pytorch_predict
import random
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
        # Choose between PyTorch and custom implementation
        options = {
            "prompt":"Which neural network implementation?:",
            "options":["from scratch implementation","pytorch","exit"]
        }
        nn_choice = get_choice(options)
        if nn_choice == "exit":
            sys.exit(0)
        
        # Set the correct modules based on choice
        if nn_choice == "pytorch":
            nn_module = pytorch_nn
            predict_module = pytorch_predict
            model_extension = "nn.pth"
        else:  # custom (numpy)
            nn_module = nn
            predict_module = predict
            model_extension = "nn.pickle"
        
        options = {
            "prompt":"Would you like to train or predict using existing model?:",
            "options":["train","predict","exit"]
        }
        choice = get_choice(options)
        if choice == "exit":
            sys.exit(0)
        elif choice == "train":
            defaults = get_defaults(nn_module.defaults)
            nn_module.main(**defaults)
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
                "nn_path":model_extension,
                "output_file_name":"predict_results.txt",
                "input":input,
                "gait_length":100
            }
            defaults = get_defaults(predict_defaults)

            # unpacks default values to named params of load_and_predict
            predict_module.load_and_predict(**defaults)
            
    print("="*20)



if __name__ == "__main__":
    main()
