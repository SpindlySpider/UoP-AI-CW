import enum
from custom_types import Chromosome, Individual
import numpy as np

class fitness():
    # used to determine fitness of a individual
    def __init__(self):
        # sine configuration
        # magnitude for sine
        self.sin_mag = 1
        # period for sine
        self.sin_period = 0.5

    def error_accumulation(self,):
        # ranking individuals based on how much error they accumulate through their gait.
        # e.g. legs going into body, legs over lapping
        pass

    def leg_angles(self,individual:Individual) -> float:
        # rank individuals based on two leg pair angles,
        # where one leg moves clockwise and the other anti clockwise 
        # and the pair of legs behind are flipped
        # aim of this function is to make realistic movement

        # we can use sin wave over each joint, 
        # every even limb (e.g. L2,L4) will be fitted against -1 instead of 1 sin wave

        # get number of rows
        fitness_val = 0
        # could do somthing with gait length to make leg walk cycle more realistic
        # used to track which direction the legs should aim for

        # use sin wave as target and calculate mean squared error as fitness
        # we can use 1 / (1+mse) to get best soultion of 1 and any suboptional solution to be 0

        # mse - sum
        # target val will be t = A sin(B(x))
        # where A is magnitude and B is period
        # if leg number is even invert the limb index.

        # we need to sum each coxa fitness


        for limb_index, chromosome in enumerate(individual):
            # each one is a frame

            # keep track of leg pair, which legs direction legs should be going
            for coxa_index in range(0,12,3):
                # we can coxa index by 2 to get if angle should be positve or negative
                pass

        return 0.0

    def coxa_fitness(self, coxa_list:list[float], opposite: bool) -> float:
        # function used to determine the fitness of a individual coxa joint with a target of sine
        # oposite used to determine which direction the target should be (pos or neg)
        total_error = 0
        for timestamp, rotation in enumerate(coxa_list):
            # might need to multiply target by acceptable angles
            target = self.sin_mag*np.sin(self.sin_period*timestamp)
            target = -target if opposite else target
            total_error += (target-rotation)**2
        mse = (1/len(coxa_list)) * total_error 
        fit_val = 1/(1+mse)
        return fit_val



    def pose(self):
        # rank individuals based on keyframe pose
        # every x keyframes check how close individual to reaching rotation of keyframe.
        # the closer the better fitness
        pass

