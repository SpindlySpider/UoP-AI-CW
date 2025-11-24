"""
This file provides utilities for GA and NN CLI.
"""

def get_choice(choice_dict:dict[str,any]) -> int | bool:
    """
    Get user choice on what to run
    Parameters:
        options: list of choices
    Returns:
        integer representing a choice form the list or bool for quit
    """
    choice = None
    print("="*20)
    print(choice_dict["prompt"])
    for i in range(len(choice_dict["options"])):
        print(f"  {i}) {choice_dict['options'][i]}")
    print("-"*20)
    while choice is None:
        user_input = input("> ")
        try:
            converted = int(user_input)
            if converted <= len(choice_dict["options"]) and converted >= 0:
                choice = converted
            else:
                print(f"{user_input} is not accepted, please choose one of the options (0-{len(choice_dict['options']-1)}) or q to exit")
        except:
            print(f"{user_input} is not accepted, please choose one of the options (0-{len(choice_dict['options'])-1}) or q to exit")
    print("="*20)
    return choice_dict["options"][choice]

def modify_default(choice_dict:dict[str,any],defaults:dict):
    """
    used to modify select choice
    Parameters:
        choice_dict (dict): object which holds prompt and key options
        defaults (dict): object which holds values to be modified
    """
    finished = False
    # add exit and save to options
    choice_dict["options"].append("exit")
    choice_dict["options"].append("save and exit")
    modified_default = defaults
    choice = ""
    while not finished:
        # get value to modify
        choice = get_choice(choice_dict)
        if choice == "exit":
            # exit without saving
            return defaults
        elif choice == "save and exit":
            return modified_default
        elif choice == "optimiser":
            print(f"modifying:",choice)
            # Determine which optimizers are available based on current defaults
            if "nn_save" in modified_default and modified_default["nn_save"].endswith(".pth"):
                # PyTorch optimizers
                optimiser_options = {
                    "prompt":"Choose between (PyTorch):",
                    "options":["sgd","adam","exit"]
                }
            else:
                # Custom implementation optimizers
                optimiser_options = {
                    "prompt":"Choose between (Custom):",
                    "options":["gradient_descent","adam","exit"]
                }
            optimiser_choice = get_choice(optimiser_options)
            if optimiser_choice == "exit":
                print("not saved")
            elif optimiser_choice in ["gradient_descent","adam","sgd"]:
                modified_default[choice] = optimiser_choice
                print(f"new value saved: {optimiser_choice}")
        else:
            # modify defaults
            print(f"modifying:",choice)
            print("current value:",modified_default[choice])
            new_val_type = type(modified_default[choice])
            print("new value must be of type",new_val_type)
            if new_val_type is list:
                # default to infinite amount of ints for hidden layers
                list_length = 0
                val_type = int
                if choice == "input":
                    # if its the predict input set to 24
                    list_length = 24
                    val_type = float
                new_val = handle_lists(val_type,list_length)
            else:
                new_val = input("new value > ")
            try:
                if new_val:
                    modified_default[choice] = new_val_type(new_val)
                else:
                    print("value empty, not saved")
            except:
                print("conversion failed, value not saved")

def display_defaults(defaults):
    keys = defaults.keys()
    for i,key in enumerate(keys):
        # print key name with idxs they can be modified by user
        print(f"  {i}) {key.replace('_',' ')}: {defaults[key]}")

def get_defaults(defaults):
    """
    helper function to get default configuration for GA and allow for user modification of values.
    """
    print("starting with these defaults:")
    keys = defaults.keys()
    display_defaults(defaults)
    print("="*20)
    user_in = input("would you like to change anything? (y/N): ")
    if user_in.upper() == "Y":
        choices = {
            "prompt":"Which would you like to change?",
            "options":list(keys)
        }
        defaults = modify_default(choices,defaults)
        print("continuing with modified:")
        display_defaults(defaults)
    else:
        print("continuing with defaults")
    return defaults


def handle_lists(val_type:any,list_length:int=0):
    """
    Handles inputting data into a list,
    This function will specificially be used for getting values for hidden layers and user input for predict.
    Parameters:
        list_length (int): length of list to ask values for, defaults to 0, if 0 then list can be infinitely long
        val_type (any): what type the list should be full of.
    Returns:
        list of size list_length
    """
    finished:bool = False
    counter:int = 0
    return_list = []
    infinite = list_length == 0
    if not infinite:
        print(f"enter {list_length} values:")
    while not finished:
        if infinite:
            print("keep entering values until you are happy, save and exit with q")
        user_input = input(f"{counter+1}) please input value of type ({val_type}) > ")
        # if list counter is 0 then, just keep accepting vars until done
        if not infinite:
            # count until counter is done
            try:
                user_input = val_type(user_input)
                return_list.append(user_input)
                counter += 1
                if counter == list_length:
                    finished = True
            except:
                print("failed to input to list - type conversion failed, please try again.")
        else:
            try:
                if user_input.upper() == "Q":
                    finished = True
                else:
                    user_input = val_type(user_input)
                    return_list.append(user_input)
                    counter += 1
            except:
                print("failed to input to list - type conversion failed, please try again.")
    return return_list
