import matplotlib.pyplot as plt


def plot_fitness_graph( avg_loss, epochs):
    plt.figure(figsize=(12, 6))
    plt.plot(range(epochs), avg_loss, color='orchid', linewidth = 2, label='Average loss')
    plt.title("Best vs Average Fitness Score Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.legend()
    plt.savefig(f"loss-over-{epochs}-epoch.png")
