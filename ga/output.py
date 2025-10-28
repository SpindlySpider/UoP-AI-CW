from custom_types import Individual
# use functions here for outputting the final best GA to a string that matlab can take
# docs here: https://uk.mathworks.com/help/matlab/ref/readmatrix.html

'''
Outputs the given solution to a file in a comma-separated format which can be read by MATLAB.
Args:
    filename (str): The name of the file to output the solution to.
    solution (Individual): The solution to be outputted.
Returns:
    The file saved with the solution data.
'''
def output(filename,solution:Individual):
    with open(filename,"w") as file:
        for arr in solution:
            out_string = ",".join(map(str,arr)) + "\n"
            file.writelines(f"{out_string}")
