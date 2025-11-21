import nn.main as nn
import ga.main as ga

def get_choice(options) -> int:
    """
    Get user choice on what to run
    Parameters:
        options: list of choices
    Returns:
        integer representing a choice form the list
    """
    #TODO: implmenet this functionality
    choice = 0
    for i in range(len(options)):
        print(f"{i}) {options[i]}")
    return choice

def main():
    #NOTE: maybe use dict instead as it would allow for great specification of choices and following choices.
    # also would be easier to visulise choice tree rather than a bunch of if statements IMO
    options = ["genetic algorithm","neural network"]
    print("="*20)
    print("what would you like to run :)?")
    get_choice(options)
    print("="*20)

    choice = 0

    print(f"okay running: {options[choice]}")
    print("="*20)


    #TODO: print default configuration and let user modify if they want for both,
    # display default values for all and let user modify as required + verify its not bad value
    # for example: 
    #   ga: generations (if we still have that), population size, gait length, crossover rate, mutation rate, graphs
    #   nn: hidden layer list, learning rate, activation funcs?, how much training data to gen, graphs,
    # also make the if statement less ugly lol

    if choice == 0:
        ga.main()
    else:
        #TODO: ask if user wants to train a new NN, or load a NN and train? or predict with existing NN
        nn.main()
    print("="*20)



if __name__ == "__main__":
    main()
