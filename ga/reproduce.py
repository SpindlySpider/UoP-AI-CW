from custom_types import Chromosome, Individual,Population
import random

def crossover(parents:Population,gait_length:int) -> Population:
    pop_size: int = len(parents)
    offspring: Population = []
    if pop_size % 2 != 0:
        # ensure even num of parents
        offspring.append(parents.pop())
        pop_size -= 1
    for i in range(0,pop_size-1,2):
        cross_point:int = random.randint(0,gait_length)
        p1,p2 = parents[i], parents[i+1]
        offspring.append(p1[:cross_point] + p2[cross_point:])
        offspring.append(p2[:cross_point] + p1[cross_point:])
    return offspring


def mutate(population:Population,mut_rate:float) -> Population:
    for in_idx,individual in enumerate(population):
        for c_idx,chromosome in enumerate(individual):
            for g_idx, gene in enumerate(chromosome):
                if random.random() <= mut_rate:
                    # mutate gene and reassign
                    gene = random.uniform(-90,90)
                    population[in_idx][c_idx][g_idx] = gene
    return population
