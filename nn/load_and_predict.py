import serialize
import numpy as np

# NOTE: can somone make this into an actual function / class lol

input = [0.0,-50,-50,-0.0,-19.220740530193638,-19.220740530193638,0.0,-50,-50,-0.0,-19.220740530193638,-19.220740530193638,-0.0,-50,-50,0.0,-19.220740530193638,-19.220740530193638,-0.0,-50,-50,0.0,-19.220740530193638,-19.220740530193638]

gait_length = 300

gait = []

nn = serialize.load()

for i in range(gait_length):
    gait.append([])
    predict = None

    # normalize input
    for j in range(len(input)):
        input[j] = (input[j] +50 )/80

    predict = nn.feed_forward(input)

    # un normalize predict
    for j in range(len(predict[0])):
        predict[0][j] = (predict[0][j]*80) - 50

    # add to gait

    # print(nn.feed_forward(predict))
    gait[i] = np.copy(predict[0])
    input = None
    input = np.copy(predict[0])

    # print(predict.flat,type(predict.flat))


print(gait)
#TODO: take from actual output.py in ga - lazy cbb to import
def output(filename:str,soultion):
    with open(filename,"w") as file:
        for arr in soultion:
            out_string = ",".join(map(str,arr)) + "\n"
            file.writelines(f"{out_string}")



output("test.txt",gait)
