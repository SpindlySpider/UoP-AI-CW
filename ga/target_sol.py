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
                # idx = (-idx) if opposite else idx
                target = (23*(math.sin(0.4*idx)))
                target = (-target) if opposite else target
                frame.append(target)
            else:
                even_limb = leg_num % 2 == 0 

                if not opposite:
                    even_limb = not even_limb

                period_offset = 2 if even_limb else 0
                # period_offset = 0 if even_limb else 2

                # if not opposite:
                    # period_offset = 0

                sin_val = (0.4*idx)+period_offset

                if opposite:
                    sin_val = (-sin_val)

                target = (100*(math.sin(sin_val))) - 100
                # target = -(target)
                if target <= -50:
                    target = -50
                # target = (-target) if opposite else target
                # target = target if opposite else (-target)

                frame.append(target)
                # need to check if for leg down we want it to stay at 50 and then go down to 0 when up
                # frame.append(-50)
        best.append(frame)
    print(best)
    output.output("sol.txt",best)


produce_target(300)
