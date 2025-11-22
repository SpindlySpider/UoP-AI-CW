import torch, numpy as np, os
from nn.torch_model import TorchNet

# load sample input
X = np.loadtxt('nn/sol.txt', delimiter=',')
if X.ndim==1: X = X.reshape(1,-1)
x = X.astype('float32')
# apply inference normalization used by run_predict_sol
x = (x + 50.0) / 80.0
xt = torch.from_numpy(x)

device = torch.device('cpu')
model = TorchNet(input_size=24, hidden_sizes=[128,64,32], output_size=24, activation='sigmoid').to(device)
state = torch.load('nn.pth', map_location=device)
# try non-strict load to avoid raising
try:
    model.load_state_dict(state)
except Exception:
    model.load_state_dict(state, strict=False)
model.eval()
with torch.no_grad():
    out = model(xt).cpu().numpy()
print("raw out stats:", out.shape, out.mean(), out.std(), out.min(), out.max())
# also show first row
print("raw out first row (first 12):", out[0,:12])
# denormalize like run_predict_sol
denorm = (out * 80.0) - 50.0
print("denorm stats:", denorm.mean(), denorm.std(), denorm.min(), denorm.max())
print("denorm first row (first 12):", denorm[0,:12])