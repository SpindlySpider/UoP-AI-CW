"""
Module used to serilize NN class load and dump
"""
import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_nn.torch_model import TorchNet


def save_torch(model: torch.nn.Module, out: str = "nn.pth"):
    """
    Save PyTorch model state_dict.
    Matches original save() function structure.
    
    Parameters:
        model (torch.nn.Module): PyTorch model to save.
        out (str): Name of file to save to, defaults to "nn.pth"
    """
    torch.save(model.state_dict(), out)


def load_torch(model: torch.nn.Module, file_name: str = "nn.pth") -> torch.nn.Module:
    """
    Load state_dict into provided model instance and return it.
    Matches original load() function structure.
    
    Parameters:
        model (torch.nn.Module): Model instance to load weights into.
        file_name (str): Name of file to load, defaults to "nn.pth"
    Returns:
        PyTorch model with loaded weights.
    """
    model.load_state_dict(torch.load(file_name, weights_only=True))
    model.eval()
    return model
