from custom_types import Chromosome, Individual
import math

class fitness():
    # used to determine fitness of a individual
    def __init__(self):
        # sine configuration
        # magnitude for sine, this should be max and min angles of legs,
        # for right now this only affects coxa, so this would be max and min rotation of coxa, this should be in radians
        self.sin_mag: float = 17.5
        # period for sine
        self.sin_period:float = 0.5

    def leg_angles(self,individual:Individual) -> float:
        # this function will look at joints and use sine as target
        coxa_fit_val = 0
        # get fitness of coxa joints
        for limb_index in range(0,24,3):
            # this will make coxa1 = [r1,r2,r3,...,max_gait], then coxa2 e.t.c
            coxa_list = [individual[row][limb_index] for row in range(len(individual))]
            opposite = (limb_index+1) % 2 == 0
            coxa_fit_val += self.coxa_fitness(coxa_list,opposite)

        # generate fitness for tibia and femur here
        # ...

        #  we can set how much each rotation fitness affects the overall fitness of this individual
        # e.g. fit = 0.5*coxa_fit_val + femur_fit*0.3 + tibia_fit*0.2 

        fit_val = 1/(1+coxa_fit_val)
        return fit_val

    def coxa_fitness(self, coxa_list:list[float], opposite: bool) -> float:
        # function used to determine the fitness of a individual coxa joint with a target of sine
        # oposite used to determine which direction the target should be (pos or neg)
        total_error = 0
        for timestamp, rotation in enumerate(coxa_list):
            # might need to multiply target by acceptable angles
            timestamp = -timestamp if opposite else timestamp
            target = self.sin_mag*math.sin(self.sin_period*timestamp)
            total_error += (target-rotation)**2
            # total_error += target-rotation
        mse_fit = (1/len(coxa_list)) * total_error
        return mse_fit
