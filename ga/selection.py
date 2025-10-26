from custom_types import Population
import random
import numpy

def roulette(population: Population, fitness: list[float]) -> Population:
    # create list same size as pop
    total_fit = sum(fitness)
    cumulative_sum = []
    selected_parents: Population = []
    running_total = 0
    # normalize and create list of cumulative values
    for fit_value in fitness:
        normalized_fit = fit_value/total_fit
        running_total += normalized_fit
        cumulative_sum.append(running_total)
    for _ in range(len(population)):
        # select individuals
        selection = random.random()
        for i in range(len(cumulative_sum)):
            # find where selection is bigger and -1 index
            if (selection < cumulative_sum[i]):
                individual = i - 1 if i > 0 else 0
                selected_parents.append(population[individual])
                break
    return selected_parents


def eliteism(population: Population, fitness: list[float],number:int) -> Population:
    elite_idx = numpy.argsort(fitness)[-number:]
    elites = [population[i] for i in elite_idx]
    return elites


def tournament(population: Population,fitness: list[float]) -> Population:
    selected_parents: Population = []
    pop_size:int = len(population)
    for _ in range(pop_size):
        p1,p2 = (random.randint(0,pop_size-1) for _ in range(2))
        if fitness[p1] > fitness[p2]:
            selected_parents.append(population[p1])
        else:
            selected_parents.append(population[p2])
    return selected_parents
