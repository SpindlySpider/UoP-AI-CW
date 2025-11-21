def get_choice(choice_dict:dict[str,any]) -> int | bool:
    """
    Get user choice on what to run
    Parameters:
        options: list of choices
    Returns:
        integer representing a choice form the list or bool for quit
    """
    #TODO: implement this functionality
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
        else:
            # modify defaults
            print(f"modifying:",choice)
            print("current value:",modified_default[choice])
            new_val_type = type(modified_default[choice])
            print("new value must be of type",new_val_type)
            new_val = input("new value > ")
            try:
                modified_default[choice] = new_val_type(new_val)
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
    print("starting ga with these defaults:")
    keys = defaults.keys()
    display_defaults(defaults)
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
