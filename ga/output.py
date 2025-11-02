from custom_types import Gait, Individual
import fitness
# use functions here for outputting the final best GA to a string that matlab can take
# docs here: https://uk.mathworks.com/help/matlab/ref/readmatrix.html

def output(filename:str,soultion:Individual):
    '''
    Outputs the given solution to a file in a comma-separated format which can be read by MATLAB.
    Args:
        filename (str): The name of the file to output the solution to.
        solution (Individual): The solution to be outputted.
    '''
    with open(filename,"w") as file:
        for arr in soultion:
            out_string = ",".join(map(str,arr)) + "\n"
            file.writelines(f"{out_string}")

def output_gait(filename:str,individual:Individual,gait_length:int):
    '''
    Outputs the gait generated from the given individual to a file in a comma-separated format which can be read by MATLAB.
    Args:
        filename (str): The name of the file to output the gait to.
        individual (Individual): The individual to generate the gait from.
        gait_length (int): The length of the gait.
    '''
    gait:Gait = fitness.gen_gait(individual,gait_length)
    output(filename,gait)
