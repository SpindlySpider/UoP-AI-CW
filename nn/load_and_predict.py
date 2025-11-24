import sys
from pathlib import Path
import random as rd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from ga.custom_types import Gait

try:
    # Try absolute imports (when run from project root or via main.py)

    from nn.neural_network import Neural_network
    import nn.serialise as serialize
except ImportError:
    # Fall back to relative imports (when run directly from nn/ directory)
    sys.path.insert(0, str(Path(__file__).parent.parent / 'ga'))
    from neural_network import Neural_network
    import serialise as serialize

# Normalization constants matching input_data.py training normalization
# Maps [-80, 30] to [0, 1] using (x + 80) / 110
minimum_angle = -80
maximum_angle = 30
angle_diff = maximum_angle - minimum_angle  # = 110

def predict(nn:Neural_network,input:list[float]) -> list[float]:
    """
    Takes input and predicts next frame.
    Parameters:
        nn (Neural_network): neural network to use to predict.
        input (list[float]): list of 24 floats representing joint angles in degrees
    Returns:
        list of 24 values, next predicted frame for all joints.
    """
    input = normalize(input)
    predict = nn.feed_forward(input)
    return denormalize(predict)


# Normalization/denormalization matching input_data.py
# Training uses: (x + 80) / 110 for normalization
# So denormalization is: (x * 110) - 80
normalize = lambda x : (x - minimum_angle) / angle_diff  # (x + 80) / 110
denormalize = lambda x : (x * angle_diff) + minimum_angle  # (x * 110) - 80

def predict_gait(nn:Neural_network, input:list[float],gait_length:int = 300) -> Gait:
    """
    Recursively predict entire gait from one input.
    Parameters:
        nn (Neural_network): neural network to use to predict.
        input (list[float]): list of 24 floats representing joint angles in degrees
        gait_length (int): length of gait to produce (how many predictions will it do)
    Returns:
        list of 24 values, next predicted frame for all joints.
    """
    gait:Gait = []
    gait.append(np.array(input))
    for i in range(gait_length):
        # predict prev frame, starting from input
        prediction = predict(nn,gait[i])
        # reshape for output
        gait.append(prediction.reshape(prediction.shape[1]))
    return gait

def load_and_predict(input:list[float],nn_path:str = "nn.pickle",output_file_name:str = "results.txt",gait_length:int = 300):
    """
    Helper function to easily load nn and output results.
    Parameters:
        nn_path (str): path of neural network file to load. default name is nn.pickle
        input (list[float]): list of 24 floats representing joint angles in degrees
        output_file_name (str): Name of file to output to, will output to CWD
        gait_length (int): length of gait to produce (how many predictions will it do)
    """
    # load nn
    print(f"predicting next {gait_length} poses")
    nn = serialize.load(nn_path)
    gait = predict_gait(nn,input,gait_length)
    # save predicted gait to file in root directory
    with open(output_file_name, 'w') as f:
        for frame in gait:
            f.write(','.join(map(str, frame)) + '\n')
    print(f"predicted gait saved to {output_file_name}")


if __name__ == "__main__":
    # generate random input
    input = [rd.randint(-100,100) for _ in range(24)]

    load_and_predict(input)