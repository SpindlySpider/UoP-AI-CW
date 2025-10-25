import math
import output
def produce_target(gait_length:int):
    best = []
    for idx in range(gait_length):
        frame = []
        for joint in range(24):
            opposite = (joint) % 2 == 0
            if joint < 11:
                opposite = not opposite
            if joint % 3 ==0:
                # idx = (-idx) if opposite else idx
                target = (23*(math.sin(0.4*idx)))
                target = (-target) if opposite else target
                frame.append(target)
            # elif joint % 2 == 0:
                # joint is femur
                # frame.append(-10)
            else:
                # idx = idx if opposite else (-idx)
                # target = ((50)*(math.sin(0.4*idx)))
                # target = (100*(math.sin(0.4*(-idx)))) -100
                # 0.2 for peroid looks good, but doesnt do it enough sometimes
                target = (100*(math.sin(0.4*idx+2))) -100
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
