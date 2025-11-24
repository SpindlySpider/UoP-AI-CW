import torch
import torch.nn as nn
from typing import Optional, List
import numpy as np
import pytorch_nn.graph_results as graph_results


def train_torch(
    model: nn.Module,
    input_list: np.ndarray,
    target_list: np.ndarray,
    epochs: int = 100,
    batch_size: int = 1,
    lr: float = 0.01,
    optimizer_name: str = 'sgd',
    device: Optional[torch.device] = None,
):
    """
    Train neural network to match original nn implementation exactly.
    Uses manual shuffling per epoch, MSE loss calculation, and gradient descent optimizer.
    
    Parameters:
        model: PyTorch model to train
        input_list: Input data (already normalized)
        target_list: Target data (already normalized)
        epochs: Number of training epochs
        batch_size: Size of batch per epoch
        lr: Learning rate
        optimizer_name: Optimizer to use ('sgd' or 'adam')
        device: Device to run on
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device)
    
    # MSE loss function (reduction='mean' matches the averaging in original)
    loss_fn = nn.MSELoss(reduction='mean')
    
    # Use SGD to match original nn implementation (gradient_descent)
    # PyTorch SGD with momentum=0 is equivalent to basic gradient descent
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_per_epoch: List[float] = []
    
    for epoch in range(epochs):
        model.train()
        
        # Shuffle data at beginning of each epoch (matches original nn)
        indices = np.random.permutation(len(input_list))
        shuffled_inputs = input_list[indices]
        shuffled_targets = target_list[indices]
        
        # Total loss for epoch
        mse_loss = 0.0
        batch_count = 0
        
        # Iterate over batches manually (matches original range(0, len-batch_size, batch_size))
        for b in range(0, len(shuffled_inputs) - batch_size, batch_size):
            # Get batch data
            inputs = torch.tensor(shuffled_inputs[b:b+batch_size], dtype=torch.float32).to(device)
            targets = torch.tensor(shuffled_targets[b:b+batch_size], dtype=torch.float32).to(device)
            
            # Feed forward
            preds = model(inputs)
            
            # Calculate loss (MSE)
            loss = loss_fn(preds, targets)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Optimizer step (gradient descent)
            optimizer.step()
            
            # Accumulate loss for epoch
            mse_loss += loss.item()
            batch_count += 1
        
        # Average loss for epoch (matches original: loss / (len(input_list) // batch_size))
        mean_loss = mse_loss / batch_count
        loss_per_epoch.append(mean_loss)
        
        # Print in same format as original: "mean loss {loss} | epoch: {epoch}"
        print(f"mean loss {mean_loss} | epoch: {epoch}")
    
    # Plot losses using existing helper
    try:
        graph_results.plot_loss_graph(loss_per_epoch, epochs, batch_size)
    except Exception as e:
        print(f"Could not save loss graph: {e}")

    return model, loss_per_epoch


def test_torch(model: nn.Module, input_list: np.ndarray, target_list: np.ndarray, device: Optional[torch.device] = None):
    """
    Test trained neural network on input data to find MSE.
    Matches the original test_NN function.
    
    Parameters:
        model: PyTorch model to test
        input_list: Input data (already normalized)
        target_list: Target data (already normalized)
        device: Device to run on
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device)
    model.eval()
    
    # Collect predictions
    predicts = []
    
    # Feed forward each input
    with torch.no_grad():
        for input_data in input_list:
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(input_tensor)
            predicts.append(pred.cpu().numpy())
    
    # Convert predictions to numpy array
    predicts = np.array(predicts)
    # Reshape predicts as they retain batch size
    predicts = predicts.reshape(predicts.shape[0], -1)
    
    # Calculate MSE manually to match original
    error_list = (target_list - predicts) ** 2
    error = np.average(error_list)
    
    print(f"tested nn on {len(input_list)} dataset |  MSE loss is: {error}")
    return error


def predict(model: nn.Module, inputs: np.ndarray, device: Optional[torch.device] = None, denormalize: bool = True) -> np.ndarray:
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device)
    model.eval()
    x = torch.tensor(inputs, dtype=torch.float32)
    if denormalize:
        x = (x + 50.0) / 80.0
    with torch.no_grad():
        preds = model(x.to(device))
    preds = preds.cpu().numpy()
    if denormalize:
        preds = preds * 80.0 - 50.0
    return preds
