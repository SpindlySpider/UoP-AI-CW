import matplotlib.pyplot as plt


def plot_loss_graph( avg_loss, epochs, batch_size):
    plt.figure(figsize=(12, 6))
    plt.plot(range(epochs), avg_loss, color='orchid', linewidth = 2, label='Average loss')
    plt.title("Mean loss over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Mean loss")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"loss-over-{epochs}-epoch-batch-size-{batch_size}.png")
