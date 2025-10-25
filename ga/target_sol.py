import math
import output
def produce_target(gait):
    best = []
    for idx in range(gait):
        frame = []
        for joint in range(24):
            opposite = (joint) % 2 == 0
            if joint < 11:
                opposite = not opposite
            if joint % 3 ==0:
                # idx = (-idx) if opposite else idx
                target = (30*(math.sin(0.5*idx)))
                target = (-target) if opposite else target
                frame.append(target)
            else:
                frame.append(0)
            pass
        best.append(frame)
    print(best)
    output.output("sol.txt",best)


produce_target(300)
