from custom_types import Individual
# use functions here for outputting the final best GA to a string that matlab can take
# docs here: https://uk.mathworks.com/help/matlab/ref/readmatrix.html

def output(filename,soultion:Individual):
    with open(filename,"w") as file:
        for arr in soultion:
            out_string = ",".join(map(str,arr)) + "\n"
            file.writelines(f"{out_string}")
