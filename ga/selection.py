from custom_types import Chromosome, Individual
import random

def roulette():
    pass

def tournament(population: list[Individual],fitness: list[float]) -> Individual:
    selected_parents = []
    pop_size = len(population)
    for _ in range(pop_size):
        p1,p2 = (random.randint(0,pop_size) for _ in range(2))
        if fitness[p1] > fitness[p2]:
            selected_parents.append(population[p1])
            continue
        selected_parents.append(population[p2])
    return selected_parents
