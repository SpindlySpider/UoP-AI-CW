import random as rd
import numpy as np
from nn.torch_model import TorchNet
from nn.serialize import load_torch


# generate random initial input
input_vec = [rd.randint(-100, 100) for _ in range(24)]
gait_length = 300

# create model and load weights
model = TorchNet(input_size=24, hidden_sizes=[24], output_size=24, activation='tanh')
model = load_torch(model, "nn.pth")

gait = []
gait.append(input_vec)

current = np.array(input_vec, dtype=np.float32)
for i in range(gait_length):
    # normalize (if your model expects normalized inputs)
    x = (current + 50.0) / 80.0
    x = np.expand_dims(x, axis=0)  # batch dim
    preds = model(torch_tensor := __import__('torch').tensor(x, dtype=__import__('torch').float32))
    preds = preds.detach().cpu().numpy()[0]
    # denormalize
    preds = preds * 80.0 - 50.0
    gait.append(preds.tolist())
    current = preds


def output(filename: str, solution):
    with open(filename, "w") as file:
        for arr in solution:
            out_string = ",".join(map(str, arr)) + "\n"
            file.writelines(f"{out_string}")


output("results.txt", gait)
