from custom_types import Chromosome, Individual,Population
import random

def crossover(parents:Population,gait_length:int,crossover_rate:float) -> Population:
    pop_size: int = len(parents)
    offspring: Population = []
    if pop_size % 2 != 0:
        # ensure even num of parents
        offspring.append(parents.pop())
        pop_size -= 1
    for i in range(0, pop_size-1,2):
        p1,p2 = parents[i], parents[i+1]
        if random.random() <= crossover_rate:
            cross_point:int = random.randint(0,gait_length)
            offspring.append(p1[:cross_point] + p2[cross_point:])
            offspring.append(p2[:cross_point] + p1[cross_point:])
        else:
            # dont cross over put parents in new pop
            offspring.append(p1)
            offspring.append(p2)
    return offspring


def mutate(population:Population,mut_rate:float) -> Population:
    # mutate should generate a new frame, not a individual float
    # for in_idx,individual in enumerate(population):
        # for c_idx,chromosome in enumerate(individual):
            # if random.random() <= mut_rate:
                # chromosome = [random.uniform(-45,45) for _ in range(len(chromosome))]
                # population[in_idx][c_idx] = chromosome
    for in_idx,individual in enumerate(population):
        for c_idx,chromosome in enumerate(individual):
            for g_idx, gene in enumerate(chromosome):
                if random.random() <= mut_rate:
                    # mutate gene and reassign
                    gene = random.uniform(-45,45)
                    population[in_idx][c_idx][g_idx] = gene
    return population
