"""
Module used to serilize NN class load and dump
"""
#from nn.neural_network import Neural_network
import pickle
import torch
import os

# Avoid importing the deprecated pure-Python Neural_network which raises an ImportError.
# Prefer the PyTorch-based TorchNet.
try:
    from pytorch_nn.torch_model import TorchNet
except Exception:
    # allow running the script directly (script's folder will be on sys.path)
    from torch_model import TorchNet

def save_torch(model, out='nn.pth'):
    """Save a torch.nn.Module state_dict (or fallback to saving object)."""
    if hasattr(model, 'state_dict'):
        torch.save(model.state_dict(), out)
    else:
        # fallback: save the object directly (less preferred)
        torch.save(model, out)

def load_torch(model_cls=None, path='nn.pth', map_location=None):
    """Load state and optionally instantiate model_cls and load state_dict."""
    state = torch.load(path, map_location=map_location)
    if model_cls is None:
        return state
    model = model_cls()
    model.load_state_dict(state)
    return model


def save_torch(model:torch.nn.Module, out: str = "nn.pth"):
    """
    Save PyTorch model state_dict.
    """
    torch.save(model.state_dict(), out)


def load_torch(model:torch.nn.Module, file_name: str = "nn.pth") -> torch.nn.Module:
    """
    Load state_dict into provided model instance and return it.
    """
    model.load_state_dict(torch.load(file_name))
    model.eval()
    return model
