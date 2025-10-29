from custom_types import Population
import random

def roulette(population: Population, fitness: list[float]) -> Population:
    """
    Selects individuals using a cumulative probability method based on fitness values.

    This method normalizes the fitness values to create a cumulative distribution.
    For each selection, a random number is drawn and the first individual whose
    cumulative probability exceeds this number is selected.

    Parameters
    ----------
    population : Population
        The current population of individuals.
    fitness : list[float]
        List of fitness values corresponding to each individual.

    Returns
    -------
    Population
        A new list of selected individuals of the same size as the input population.
    """
    total_fit = sum(fitness)
    if total_fit == 0:
        return random.sample(population, len(population))

    cumulative_sum = []
    running_total = 0.0
    selected_parents: Population = []

    # Build cumulative distribution
    for fit_value in fitness:
        normalized_fit = fit_value / total_fit
        running_total += normalized_fit
        cumulative_sum.append(running_total)

    # Select individuals based on cumulative probability
    for _ in range(len(population)):
        selection = random.random()
        for i, cumulative_value in enumerate(cumulative_sum):
            if selection < cumulative_value:
                # Select the individual just before exceeding the random number
                individual_index = i - 1 if i > 0 else 0
                selected_parents.append(population[individual_index])
                break

    return selected_parents


def tournament(population: Population, fitness: list[float], num_selected: int) -> Population:
    """
    Selects individuals using tournament selection.

    For each parent to select, 'num_selected' individuals are randomly sampled
    from the population, and the one with the highest fitness is chosen.

    Parameters
    ----------
    population : Population
        The current population of individuals.
    fitness : list[float]
        List of fitness values corresponding to each individual.
    num_selected : int
        Number of individuals to compare in each tournament.

    Returns
    -------
    Population
        A new list of selected individuals of the same size as the input population.
    """
    selected_parents: Population = []
    pop_size: int = len(population)

    for _ in range(pop_size):
        # Randomly pick individuals for the tournament
        selected_idx = [random.randint(0, pop_size - 1) for _ in range(num_selected)]

        # Choose the one with the highest fitness
        best_idx = max(selected_idx, key=lambda i: fitness[i])
        selected_parents.append(population[best_idx])

    return selected_parents
