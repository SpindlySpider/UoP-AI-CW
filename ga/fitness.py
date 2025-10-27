from custom_types import Chromosome, Individual
import target_sol

class fitness():
    # used to determine fitness of a individual
    def __init__(self,gait_length:int):
        # we generate the target solution initially to reduce computation later.
        self.target_individual = target_sol.produce_target(gait_length)
        self.gait_length = gait_length

    def get_fitness(self,individual:Individual) -> float:
        # get the fitness of this individual by comparison to target
        # using mean squared error for error: https://en.wikipedia.org/wiki/Mean_squared_error

        # fitness dictionary used to track the error of each joint
        fit_dict:dict[str,float] = {"coxa":0,"femur":0,"tibia":0}
        # used to get current joint in loop
        joint_names = ["coxa","femur","tibia"]

        for frame_idx in range(self.gait_length):
            for chromosome_idx in range(24):
                joint = joint_names[chromosome_idx % 3]
                t = self.target_individual[frame_idx][chromosome_idx]
                p = individual[frame_idx][chromosome_idx]
                err = (t - p)**2
                fit_dict[joint] += err

        for joint in joint_names:
            j = fit_dict[joint]
            # MSE needs sum of (t-p)^2 to be * by 1/n
            # n = joint num of that joint * gait, e.g. 8 * 300
            # j = (1/(8*self.gait_length)) * j
            j = (j/(8*self.gait_length))
            # make the value relative to 1 instead of large number of error
            j = 1/(1+j)
            fit_dict[joint] = j
        # weight the comparison here. e.g. coxa worth more e.t.c 
        # normalize 
        # total = fit_dict["coxa"] + fit_dict["femur"] + fit_dict["tibia"]
        # fit_val = (0.34*(fit_dict["coxa"]/total)) + (0.33*(fit_dict["femur"]/total)) + (0.33*(fit_dict["tibia"]/total))
        # ^ stuff above commented out because I couldnt get a good result from it
        fit_val = fit_dict["coxa"] + fit_dict["femur"] + fit_dict["tibia"]
        return fit_val
