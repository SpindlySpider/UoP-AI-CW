import sys
from pathlib import Path
import random as rd
import numpy as np
import torch

# Add parent directory to access ga module
sys.path.insert(0, str(Path(__file__).parent.parent))
from ga.custom_types import Gait
from ga.output import output

from pytorch_nn.torch_model import TorchNet
from pytorch_nn import serialise

# Normalization constants matching input_data.py training normalization
# Maps [-50, 30] to [0, 1] using (x + 50) / 80
minimum_angle = -50
maximum_angle = 30
angle_diff = maximum_angle - minimum_angle  # = 80

def predict(model:TorchNet, input:list[float]) -> np.ndarray:
    """
    Takes input and predicts next frame.
    Parameters:
        model (TorchNet): PyTorch neural network to use to predict.
        input (list[float]): list of 24 floats representing joint angles in degrees
    Returns:
        numpy array of 24 values, next predicted frame for all joints.
    """
    input_normalized = normalize(input)
    
    # Convert to tensor and add batch dimension
    device = next(model.parameters()).device
    input_tensor = torch.from_numpy(np.array(input_normalized, dtype=np.float32)).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy()
    
    return denormalize(prediction)


# Normalization/denormalization matching input_data.py
# Training uses: (x + 50) / 80 for normalization
# So denormalization is: (x * 80) - 50
normalize = lambda x : (np.array(x) - minimum_angle) / angle_diff  # (x + 50) / 80
denormalize = lambda x : (x * angle_diff) + minimum_angle  # (x * 80) - 50

def predict_gait(model:TorchNet, input:list[float], gait_length:int = 100) -> Gait:
    """
    Recursively predict entire gait from one input.
    Parameters:
        model (TorchNet): PyTorch neural network to use to predict.
        input (list[float]): list of 24 floats representing joint angles in degrees
        gait_length (int): length of gait to produce (how many predictions will it do)
    Returns:
        Gait: list of numpy arrays, each containing 24 joint angles
    """
    gait:Gait = []
    gait.append(np.array(input))
    for i in range(gait_length):
        # predict next frame, starting from input
        prediction = predict(model, gait[i])
        # reshape for output
        gait.append(prediction.reshape(prediction.shape[1]))
    return gait

def load_and_predict(input:list[float], nn_path:str = "nn_pytorch.pth", output_file_name:str = "pytorch_predict_results.txt", gait_length:int = 100):
    """
    Helper function to easily load PyTorch model and output results.
    Parameters:
        input (list[float]): list of 24 floats representing joint angles in degrees
        nn_path (str): path of neural network file to load. default name is nn_pytorch.pth
        output_file_name (str): Name of file to output to, will output to pytorch_nn folder by default
        gait_length (int): length of gait to produce (how many predictions will it do)
    """
    # Ensure the path is relative to pytorch_nn/ folder if just a filename
    if not Path(nn_path).is_absolute() and not str(nn_path).startswith('.'):
        nn_path = str(Path(__file__).parent / nn_path)
    
    if not Path(output_file_name).is_absolute() and not str(output_file_name).startswith('.'):
        output_file_name = str(Path(__file__).parent / output_file_name)
    
    # load model - instantiate with same architecture as trained
    print(f"predicting next {gait_length} poses")
    model = TorchNet(input_size=24, hidden_sizes=[128, 64, 32], output_size=24, activation='sigmoid')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model = serialise.load_torch(model, nn_path)
    
    gait = predict_gait(model, input, gait_length)
    # save predicted gait to file
    output(output_file_name, gait)
    print(f"predicted gait saved to {output_file_name}")


if __name__ == "__main__":
    # generate random input
    input = [rd.randint(-100, 100) for _ in range(24)]

    load_and_predict(input)
