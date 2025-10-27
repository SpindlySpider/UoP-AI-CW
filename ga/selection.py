from custom_types import Population
import random
import numpy

def roulette(population: Population, fitness: list[float]) -> Population:
    # create list same size as pop
    # need to come up with a better way, I think selection is too fine, since there is like 300 individuals or whatever
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


def tournament(population: Population,fitness: list[float],num_selected:int) -> Population:
    # num selected is how many to compare in a tournament
    selected_parents: Population = []
    pop_size:int = len(population)
    # increase size to compare like 5 individuals at a time
    for _ in range(pop_size):
        selected_idx = [random.randint(0,pop_size-1) for _ in range(num_selected)]
        # need to get the highest value in these idxs
        best = {"idx":0,"val":0}
        for i in selected_idx:
            if fitness[i] >= best["val"]:
                best = {"idx":i,"val":fitness[i]}
        selected_parents.append(population[best["idx"]])
    return selected_parents
