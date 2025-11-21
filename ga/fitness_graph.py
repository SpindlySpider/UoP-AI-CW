import matplotlib.pyplot as plt


def plot_fitness_graph(fitness_values, avg_fitness_values, generations):
    '''
    Plots the fitness graph showing best and average fitness scores over generations.
    Parameters:
    fitness_values (list): List of best fitness scores for each generation.
    avg_fitness_values (list): List of average fitness scores for each generation.
    generations (int): Total number of generations.
    '''
    # Create a new figure for the plot
    plt.figure(figsize=(12, 6))
    # Plot the best fitness values
    plt.plot(range(generations), fitness_values, color='cornflowerblue', linewidth=2, label='Best Fitness')
    # Plot the average fitness values
    plt.plot(range(generations), avg_fitness_values, color='orchid', linewidth = 2, label='Average Fitness')
    # Add title to the plot
    plt.title("Best vs Average Fitness Score Over Generations")
    # label x-axis
    plt.xlabel("Generation")
    # label y-axis
    plt.ylabel("Fitness Score")
    # Show grid
    plt.grid(True)
    # Show legend
    plt.legend()
    # Show tight layout
    plt.tight_layout()
    # Show the plot
    plt.show()





