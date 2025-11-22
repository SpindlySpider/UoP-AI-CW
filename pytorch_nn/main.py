import pytorch_nn.input_data as input_data
from pytorch_nn.torch_model import TorchNet
from pytorch_nn.dataset import make_dataloader
from pytorch_nn.torch_training import train_torch
from pytorch_nn.serialize import save_torch
from pytorch_nn.run_predict_sol import run_predictions


def main():
    # define hidden layer sizes
    hidden_layers = [128, 64, 32]

    # get training data (input_data already normalizes to (x+50)/80)
    data = input_data.generate_train_test_data(0.95, 40, 700)
    train_in, train_out = data["training"]
    test_in, test_out = data["test"]

    # build dataloaders (pass normalize=False because data is already normalized)
    train_loader = make_dataloader(train_in, train_out, batch_size=16, normalize=False)
    val_loader = make_dataloader(test_in, test_out, batch_size=16, normalize=False)

    # model
    model = TorchNet(input_size=24, hidden_sizes=hidden_layers, output_size=24, activation='sigmoid')

    # train
    model, losses = train_torch(model, train_loader, epochs=100, lr=0.01, val_loader=val_loader)

    # save
    save_torch(model, out="nn.pth")
    
    run_predictions("sol.txt", model_path="nn.pth", out_path="sol_pred.txt")


    


if __name__ == "__main__":
    main()
