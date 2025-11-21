import nn.serialize
import random as rd
import numpy as np

# NOTE: can somone make this into an actual function / class lol

# input = [0.0,-50,-50,-0.0,-19.220740530193638,-19.220740530193638,0.0,-50,-50,-0.0,-19.220740530193638,-19.220740530193638,-0.0,-50,-50,0.0,-19.220740530193638,-19.220740530193638,-0.0,-50,-50,0.0,-19.220740530193638,-19.220740530193638]
# input = [-50 for _ in range(24)]
input = [rd.randint(-100,100) for _ in range(24)]

gait_length = 300

gait = []

nn = serialize.load()


gait.append(input)
for i in range(gait_length):
    gait.append([])
    input = np.array(input)

    # normalize input
    input = (input + 50 )/80

    predict = nn.feed_forward(input)

    # un normalize predict
    predict = predict*80 - 50
    # print(predict[0][0])

    # add to gait

    # print(nn.feed_forward(predict))
    gait[i+1] = predict[0]
    input = predict[0]
    # input = None
    # input = np.copy(predict[0])


print(gait)
#TODO: take from actual output.py in ga - lazy cbb to import
def output(filename:str,soultion):
    with open(filename,"w") as file:
        for arr in soultion:
            out_string = ",".join(map(str,arr)) + "\n"
            file.writelines(f"{out_string}")



output("results.txt",gait)
