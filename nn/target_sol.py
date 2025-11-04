import math

from numpy import random
import output
import matplotlib.pyplot as plt
from custom_types import  Individual,Gait

def produce_target(gait_length:int,period,coxa_mag,tf_v_shift,tf_mag) -> Gait:
    best: Gait = []

    period_offset:float = 2
    for idx in range(gait_length):
        frame:list[float] = []
        for joint in range(24):
            leg_num:int = joint // 3
            even_limb:int = leg_num % 2 == 0 
            coxa_opposite:bool = even_limb
            if joint < 11:
                # if the joint is on the left side invert opposite
                coxa_opposite = not even_limb
            if joint % 3 ==0:
                # coxa joint rotate
                target:float = (coxa_mag*(math.sin(period*idx)))
                target = (-target) if coxa_opposite else target
                frame.append(target)
            else:
                # value obtained through experimentation.
                # match period so that gait syncs up
                sin_val:float = (period*idx)+period_offset

                if even_limb:
                    # invert direction of sin wave if it is an even limb
                    # this is so it syncs up with coxa rotation
                    sin_val = (-sin_val)

                target:float = (tf_mag*(math.sin(sin_val))) - tf_v_shift

                if target <= -50:
                    target = -50
                frame.append(target)
        best.append(frame)
    return best

def random_sol(gait_length:int) -> Individual:
    period = round(random.uniform(0.05,1),3)
    c_mag = round(random.uniform(5,23),3)
    tf_v_shift = round(random.uniform(40,50),3)
    tf_mag = round(random.uniform(10,30),3)
    optimal_solution = produce_target(gait_length,period,c_mag,tf_v_shift,tf_mag)
    return optimal_solution

def generate_graph(individual):
    # sorry for messy code
    # get frames for plotting
    x:list[int] = [f for f in range(len(individual))]

    coxa_left:list[float] = []
    femur_tibia_left:list[float] = []
    coxa_opposite:list[float] = []
    femur_tibia_left_opposite:list[float] = []
    # limits the number of frames displayed on graph
    frame_limit:int = 50
    for frame in range(len(x)):
        coxa_left.append(individual[frame][0])
        coxa_opposite.append(individual[frame][3])
        femur_tibia_left.append(individual[frame][1])
        femur_tibia_left_opposite.append(individual[frame][4])
    plt.plot(x[:frame_limit],coxa_left[:frame_limit], label="coxa left")
    plt.plot(x[:frame_limit],femur_tibia_left[:frame_limit], label="tibia and femur")
    plt.legend()
    plt.savefig("output.png")
    plt.close()
    plt.plot(x[:frame_limit],coxa_opposite[:frame_limit], label="coxa left")
    plt.plot(x[:frame_limit],femur_tibia_left_opposite[:frame_limit], label="tibia and femur")
    plt.legend()
    plt.savefig("output_opposite.png")

if __name__ == "__main__":

    # optimal_solution = produce_target(300,0.2,20,45,25)
    optimal_solution = random_sol(300)
    generate_graph(optimal_solution)
    output.output("sol.txt",optimal_solution)
