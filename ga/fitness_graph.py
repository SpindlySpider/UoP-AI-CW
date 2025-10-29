import matplotlib.pyplot as plt


def plot_fitness_graph(fitness_values, avg_fitness_values, generations):
    plt.figure(figsize=(12, 6))
    plt.plot(range(generations), fitness_values, color='cornflowerblue', linewidth=2, label='Best Fitness')
    plt.plot(range(generations), avg_fitness_values, color='orchid', linewidth = 2, label='Average Fitness')
    plt.title("Best vs Average Fitness Score Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()





