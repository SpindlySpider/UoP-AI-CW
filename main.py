import nn.main as nn
import ga.main as ga

def get_choice(options):
    """
    Get user choice on what to run
    Parameters:
        options: list of choices
    """
    for i in range(len(options)):
        print(f"{i}) {options[i]}")

def main():
    options = ["genetic algorithm","neural network"]
    # maybe use dict
    print("="*20)
    print("what would you like to run :)?")
    get_choice(options)
    print("="*20)

    choice = 1

    # print(f"okay running {choice}")
    print(f"okay running: {options[choice]}")
    print("="*20)

    # print default configuration and let user modify if they want

    if choice == 0:
        ga.main()
    else:
        nn.main()
    print("="*20)



if __name__ == "__main__":
    main()
