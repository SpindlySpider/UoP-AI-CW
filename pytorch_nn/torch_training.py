import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List
import numpy as np
import pytorch_nn.graph_results as graph_results


def train_torch(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 500,
    lr: float = 0.01,
    optimizer_name: str = 'sgd',
    device: Optional[torch.device] = None,
    val_loader: Optional[DataLoader] = None,
):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) if optimizer_name == 'sgd' else torch.optim.Adam(model.parameters(), lr=lr)

    loss_per_epoch: List[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        mean_loss = epoch_loss / max(batch_count, 1)
        loss_per_epoch.append(mean_loss)
        print(f"Epoch {epoch+1}/{epochs} - loss: {mean_loss}")

        # optional validation
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_batches = 0
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    preds = model(xb)
                    val_loss += loss_fn(preds, yb).item()
                    val_batches += 1
                if val_batches:
                    print(f"  val loss: {val_loss/val_batches}")

    # plot losses using existing helper if available
    try:
        graph_results.plot_loss_graph(loss_per_epoch, epochs, train_loader.batch_size)
    except Exception:
        pass

    return model, loss_per_epoch


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
