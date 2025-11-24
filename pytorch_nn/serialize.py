"""
Module used to serilize NN class load and dump
"""
import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))



def save_torch(model: torch.nn.Module, out: str = "nn.pth"):
    """
    Save PyTorch model state_dict.
    Matches original save() function structure.
    
    Parameters:
        model (torch.nn.Module): PyTorch model to save.
        out (str): Name of file to save to, defaults to "nn.pth"
    """
    # Ensure the path is relative to pytorch_nn/ folder if just a filename
    if not Path(out).is_absolute() and not str(out).startswith('.'):
        out = str(Path(__file__).parent / out)
    
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
    # Ensure the path is relative to pytorch_nn/ folder if just a filename
    if not Path(file_name).is_absolute() and not str(file_name).startswith('.'):
        file_name = str(Path(__file__).parent / file_name)
    
    model.load_state_dict(torch.load(file_name, weights_only=True))
    model.eval()
    return model
