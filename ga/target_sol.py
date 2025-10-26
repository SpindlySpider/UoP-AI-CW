import math
import output
import matplotlib.pyplot as plt
def produce_target(gait_length:int):
    best = []
    for idx in range(gait_length):
        frame = []
        for joint in range(24):
            opposite = (joint) % 2 == 0
            leg_num = joint // 3
            if joint < 11:
                # if the joint is on the left side invert
                opposite = not opposite
            if joint % 3 ==0:
                # coxa joint rotate
                target = (20*(math.sin(0.4*idx)))
                target = (-target) if opposite else target
                frame.append(target)
            else:
                # we could use a matplotlib to show timings.
                # want this on the troth of the coxa rotation
                # off set the movement so that when the leg is moving forward it is up e.g. rotation 0
                even_limb = leg_num % 2 == 0 

                # value obtained through experimentation.
                period_offset = 2

                sin_val = (0.4*idx)+period_offset

                if even_limb:
                    sin_val = (-sin_val)

                target = (25*(math.sin(sin_val))) - 45
                # clamp values to -50
                if target <= -50:
                    target = -50

                frame.append(target)
        best.append(frame)
    return best

def generate_graph(individual):
    # need to track directions of coxa for opposite and non
    x = [f for f in range(len(individual))]
    # then need to line up tibia and femur rotations
    # ---
    # for now we only track left side, since right is inverted
    # sorry for messy code just draft
    coxa_left = []
    femur_tibia_left = []
    coxa_oposite = []
    femur_tibia_left_oposite = []
    frame_limit = 50
    for frame in range(len(x)):
        coxa_left.append(individual[frame][0])
        coxa_oposite.append(individual[frame][3])
        femur_tibia_left.append(individual[frame][1])
        femur_tibia_left_oposite.append(individual[frame][4])
    plt.plot(x[:frame_limit],coxa_left[:frame_limit], label="coxa left")
    plt.plot(x[:frame_limit],femur_tibia_left[:frame_limit], label="tibia and femur")
    plt.legend()
    plt.savefig("output.png")
    plt.close()
    plt.plot(x[:frame_limit],coxa_oposite[:frame_limit], label="coxa left")
    plt.plot(x[:frame_limit],femur_tibia_left_oposite[:frame_limit], label="tibia and femur")
    plt.savefig("output_opposite.png")


optimal_solution = produce_target(300)
print(optimal_solution)
generate_graph(optimal_solution)
output.output("sol.txt",optimal_solution)
