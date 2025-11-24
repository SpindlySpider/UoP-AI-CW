import matplotlib.pyplot as plt


def plot_loss_graph( avg_loss, epochs, batch_size):
    """
    Plots the average loss over epochs and saves the figure as a PNG file.
    Parameters:
        avg_loss (list or array): Average loss values for each epoch.
        epochs (int): Number of epochs.
        batch_size (int): Batch size used during training.
    """
    # Plotting the average loss over epochs
    plt.figure(figsize=(12, 6))
    # Plot the average loss
    plt.plot(range(epochs), avg_loss, color='orchid', linewidth = 2, label='Average loss')
    # Add titles and labels
    plt.title("Mean loss over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Mean loss")
    # Add grid and legend
    plt.grid(True)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Add legend
    plt.legend()
    # Save the figure as a PNG file
    filename = f"loss-over-{epochs}-epoch-batch-size-{batch_size}.png"
    plt.savefig(filename)
    print("saved average loss over epochs to:",filename)
