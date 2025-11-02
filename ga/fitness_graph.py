import matplotlib.pyplot as plt


def plot_fitness_graph(fitness_values, avg_fitness_values, generations):
    '''
    Plots the fitness graph showing best and average fitness scores over generations.
    Parameters:
    fitness_values (list): List of best fitness scores for each generation.
    avg_fitness_values (list): List of average fitness scores for each generation.
    generations (int): Total number of generations.
    '''
    plt.figure(figsize=(12, 6))
    plt.plot(range(generations), fitness_values, color='cornflowerblue', linewidth=2, label='Best Fitness')
    plt.plot(range(generations), avg_fitness_values, color='orchid', linewidth = 2, label='Average Fitness')
    plt.title("Best vs Average Fitness Score Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()





