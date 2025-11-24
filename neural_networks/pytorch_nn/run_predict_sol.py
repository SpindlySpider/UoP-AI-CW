import sys
import os
import numpy as np
import torch

try:
    from pytorch_nn.torch_model import TorchNet
except Exception:
    # allow running the script directly (script's folder will be on sys.path)
    from torch_model import TorchNet

def run_predictions(input_path, model_path='nn.pth', out_path='sol_pred.txt', normalize=True, denormalize=True):
    from pathlib import Path as PathLib
    
    # Ensure the model path is relative to pytorch_nn/ folder if just a filename
    if not os.path.isabs(model_path) and not model_path.startswith('.'):
        script_dir = PathLib(__file__).parent
        model_path = str(script_dir / model_path)
    
    # Ensure the output path is relative to pytorch_nn/ folder if just a filename
    if not os.path.isabs(out_path) and not out_path.startswith('.'):
        script_dir = PathLib(__file__).parent
        out_path = str(script_dir / out_path)
    
    X = np.loadtxt(input_path, delimiter=',')
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != 24:
        raise ValueError(f"Expected 24 columns per row, found {X.shape[1]}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # instantiate model matching the checkpoint architecture
    model = TorchNet(input_size=24, hidden_sizes=[128, 64, 32], output_size=24, activation='sigmoid')
    model.to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Train first: python -m pytorch_nn.main")

    state = torch.load(model_path, map_location=device)
    # try strict load first; if it fails, fall back to non-strict (warn user)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
        print("Warning: loaded state_dict with strict=False — some parameters did not match exactly.")
    model.eval()

    x = X.astype(np.float32)
    if normalize:
        x = (x + 50.0) / 80.0

    xt = torch.from_numpy(x).to(device)
    with torch.no_grad():
        y = model(xt).cpu().numpy()

    if denormalize:
        y = (y * 80.0) - 50.0

    np.savetxt(out_path, y, delimiter=',')
    print(f"Saved predictions to {os.path.abspath(out_path)}")

if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv:
        print("Usage: python run_predict_sol.py <path/to/sol.txt> [model.pth] [out.txt]")
        sys.exit(0)

    # Default input is 'sol.txt' in current working directory if not provided
    if len(sys.argv) >= 2:
        inp = sys.argv[1]
    else:
        inp = 'sol.txt'

    mp = sys.argv[2] if len(sys.argv) > 2 else 'nn.pth'

    # Default output is 'sol_pred.txt' in same directory as input file
    if len(sys.argv) > 3:
        out = sys.argv[3]
    else:
        inp_dir = os.path.dirname(inp) or '.'
        out = os.path.join(inp_dir, 'sol_pred.txt')

    try:
        run_predictions(inp, mp, out)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)