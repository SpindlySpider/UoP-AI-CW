from custom_types import Chromosome, Individual, Gait
import math
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
        gait = gen_gait(individual,self.gait_length)
        print(gait[0])
        print(len(gait[0]))
        print(gait[0][0],gait[0][3],gait[0][6],gait[0][9])
        print(gait[0][12],gait[0][15],gait[0][18],gait[0][21])
        print(len(gait[0]))

        for frame_idx in range(self.gait_length):
            for chromosome_idx in range(24):
                joint = joint_names[chromosome_idx % 3]
                t = self.target_individual[frame_idx][chromosome_idx]
                p = gait[frame_idx][chromosome_idx]
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

def gen_gait(individual:Individual,gait_length:int) -> Gait:
    gait:Gait = []
    # need to gen the length of the gait using sin params in f
    for idx in range(gait_length):
        gait.append([])
        # limb_counter = 0
        prev_limb = []
        for c_idx,chromosome in enumerate(individual):
            mag, period, offset, neg = chromosome
            sin_val:float = (period*idx)+offset
            sin_val:float = (-sin_val) if neg else sin_val
            predict:float = mag*math.sin(sin_val)
            predict = predict if predict > -50 else -50
            # if c_idx // 3 != limb_counter and c_idx % 3 == 2 and c_idx % 3 > 0 and len(prev_limb) >= 3:
            # if limb_counter  > 1 and c_idx % 3 == 0 and limb_counter != c_idx // 3:
            # if len(gait[idx]) // 3 > limb_counter and c_idx // 3 > 1:
            # wait if gets to 2 limbs then pop stack
            prev_limb.append(predict)
            gait[idx].append(predict)
            # limb_counter += 1
            if len(prev_limb) == 6:
                # repeat prev two
                gait[idx] = gait[idx] + prev_limb[:3] + prev_limb[3:]
                prev_limb = []
        # gait[idx] = gait[idx] + prev_limb[:3]
        # print("prev",len(prev_limb))
        # prev_limb = prev_limb[3:]
        # gait[idx] = gait[idx] + prev_limb[:3]

    return gait
