import matplotlib.pyplot as plt


def plot_loss_graph( avg_loss, epochs, batch_size):
    """`Plots and saves the average loss over epochs graph.
    Args:
        avg_loss (list): List of average loss values per epoch.
        epochs (int): Number of epochs.
        batch_size (int): Size of each batch.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(epochs), avg_loss, color='orchid', linewidth = 2, label='Average loss')
    plt.title("Mean loss over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Mean loss")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    filename = f"Pytorch_ loss-over-{epochs}-epoch-batch-size-{batch_size}.png"
    plt.savefig(filename)
    print("saved average loss over epochs to:",filename)
