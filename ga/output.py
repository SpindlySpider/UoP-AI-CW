from ga.custom_types import Gait, Individual
import ga.fitness as fitness
# use functions here for outputting the final best GA to a string that matlab can take
# docs here: https://uk.mathworks.com/help/matlab/ref/readmatrix.html

def output(filename:str,soultion:Individual):
    with open(filename,"w") as file:
        for arr in soultion:
            out_string = ",".join(map(str,arr)) + "\n"
            file.writelines(f"{out_string}")

def output_gait(filename:str,individual:Individual,gait_length:int):
    gait:Gait = fitness.gen_gait(individual,gait_length)
    output(filename,gait)
