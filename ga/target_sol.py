import math
import output
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
                target = (23*(math.sin(0.4*idx)))
                target = (-target) if opposite else target
                frame.append(target)
            else:
                # we could use a matplotlib to show timings.
                # want this on the troth of the coxa rotation
                # off set the movement so that when the leg is moving forward it is up e.g. rotation 0
                even_limb = leg_num % 2 == 0 

                # values obtained through experimentation.
                period_offset = 15 if even_limb else 2

                sin_val = (0.4*idx)+period_offset

                if even_limb:
                    sin_val = (-sin_val)

                target = (100*(math.sin(sin_val))) - 100
                # clamp values to -50
                if target <= -50:
                    target = -50

                frame.append(target)
        best.append(frame)
    print(best)
    output.output("sol.txt",best)


produce_target(300)
