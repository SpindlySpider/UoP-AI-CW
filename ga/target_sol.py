import math

from numpy import random
import ga.output as output
import matplotlib.pyplot as plt
from ga.custom_types import Individual, Period

def produce_target(gait_length:int, period:Period, coxa_amplitude:float, tibia_femur_v_shift:float, tibia_femur_amplitude:float) -> Individual:
    '''
    Produce target gait based on sine wave parameters
    Args:
        gait_length (int): The length of the gait
        period (float): The period of the sine wave
        coxa_amplitude (float): The amplitude of the coxa joint movement
        tibia_femur_v_shift (float): The vertical shift of the tibia-femur joint
        tibia_femur_amplitude (float): The amplitude of the tibia-femur joint movement
    '''
    best: Individual = []

    # offset to sync up tibia-femur with coxa rotation
    period_offset:float = 2

    # generate frames based on the gait length
    for idx in range(gait_length):
        # joint's target for the frame current frame
        frame:list[float] = []
        # generate for 24 joints
        for joint in range(24):
            # determine which leg and joint
            leg_num:int = joint // 3
            # determine if the joint is on the left or right side
            even_limb:int = leg_num % 2 == 0 
            # determine if coxa joint is right or left
            coxa_right:bool = even_limb
            # left side joints need to be inverted
            if joint < 11:
                # if the joint is on the left side invert right
                coxa_right = not even_limb
            # determine if coxa joint
            if joint % 3 ==0:
                # coxa joint rotation
                target:float = (coxa_amplitude*(math.sin(period*idx)))
                # invert direction if right side
                target = (-target) if coxa_right else target
                # append to frame
                frame.append(target)
            # determine if tibia-femur joint
            else:
                # match period so that gait syncs up
                sin_val:float = (period*idx)+period_offset

                # determine if joint is on the left side
                if even_limb:
                    # invert direction of sin wave if it is an even limb
                    # this is so it syncs up with coxa rotation
                    sin_val = (-sin_val)

                # compute target value for tibia-femur joint
                target:float = (tibia_femur_amplitude*(math.sin(sin_val))) - tibia_femur_v_shift

                if target <= -50:
                    target = -50
                frame.append(target)
        best.append(frame)
    return best

def random_sol(gait_length:int) -> Individual:
    '''
    Generate a random solution for based on the target gait parameter
    Args:
        gait_length (int): The length of the gait
    '''
    period:Period = round(random.uniform(0.05,1),3)
    coxa_amplitude: float = round(random.uniform(5,23),3)
    tibia_femur_v_shift: float = round(random.uniform(40,50),3)
    tibia_femur_amplitude: float = round(random.uniform(5,30),3)
    optimal_solution: Individual = produce_target(gait_length,period,coxa_amplitude,tibia_femur_v_shift,tibia_femur_amplitude)
    return optimal_solution

def generate_graph(individual):
    '''
    Generate a graph from the individual's joint angles over time
    Args:
        individual (Individual): The individual's joint angles
    '''
    # get frames for plotting
    x:list[int] = [f for f in range(len(individual))]

    # extract joint angles for plotting
    coxa_left:list[float] = []
    femur_tibia_left:list[float] = []
    coxa_right:list[float] = []
    femur_tibia_left_right:list[float] = []
    # limits the number of frames displayed on graph
    frame_limit:int = 50

    # extract joint angles for each frame
    for frame in range(len(x)):
        # append joint angles to respective lists
        coxa_left.append(individual[frame][0])
        coxa_right.append(individual[frame][3])
        femur_tibia_left.append(individual[frame][1])
        femur_tibia_left_right.append(individual[frame][4])
    # plot the joint angles
    plt.plot(x[:frame_limit],coxa_left[:frame_limit], label="coxa left")
    plt.plot(x[:frame_limit],femur_tibia_left[:frame_limit], label="tibia and femur left")
    plt.plot(x[:frame_limit],coxa_right[:frame_limit], label="coxa right")
    plt.plot(x[:frame_limit],femur_tibia_left_right[:frame_limit], label="tibia and femur right")
    plt.legend()
    # save the graph
    plt.savefig("output.png")
    plt.close()

if __name__ == "__main__":
    solution = optimal_solution = random_sol(300)
    generate_graph(optimal_solution)
    output.output("sol.txt",optimal_solution)
