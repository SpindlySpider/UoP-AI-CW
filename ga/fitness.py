from custom_types import Chromosome, Individual
import math

class fitness():
    # used to determine fitness of a individual
    def __init__(self):
        # sine configuration
        # magnitude for sine, this should be max and min angles of legs,
        # for right now this only affects coxa, so this would be max and min rotation of coxa, this should be in radians
        self.sin_mag: float = 6
        # period for sine
        self.sin_period:float = 0.5

    def leg_angles(self,individual:Individual) -> float:
        # this function will look at joints and use sine as target
        # we can use sin wave over each joint

        fitness_val = 0

        coxa_fit_val = 0
        # get fitness of coxa joints
        for limb_index in range(1,24,3):
            # extract coxa rotation for L1,L2,... etc
            # each entry is a different timestamp of that limbs rotation
            coxa_list = [individual[row][limb_index-1] for row in range(len(individual))]
            # not getting correct amount 
            # print("indvidual:",individual)
            # print(" coxalist:",coxa_list)
            opposite = limb_index % 2 == 0
            coxa_fit_val += self.coxa_fitness(coxa_list,opposite)
        # print("coxa_fit",coxa_fit_val)

        # generate fitness for tibia and femur here
        # ...

        #  we can set how much each rotation fitness affects the overall fitness of this individual
        # e.g. fit = 0.5*coxa_fit_val + femur_fit*0.3 + tibia_fit*0.2 

        return coxa_fit_val

    def coxa_fitness(self, coxa_list:list[float], opposite: bool) -> float:
        # function used to determine the fitness of a individual coxa joint with a target of sine
        # oposite used to determine which direction the target should be (pos or neg)
        total_error = 0
        for timestamp, rotation in enumerate(coxa_list):
            # might need to multiply target by acceptable angles
            target = self.sin_mag*math.sin(self.sin_period*timestamp)
            target = -target if opposite else target
            total_error += (target-rotation)**2
        mse = (1/len(coxa_list)) * total_error
        fit_val = 1/(1+mse)
        return fit_val
