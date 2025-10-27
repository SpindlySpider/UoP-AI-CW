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

def uniform_crossover(parents:Population,gait_length:int,crossover_rate:float) -> Population:
    # explanation found here: 
    # - https://en.wikipedia.org/wiki/Crossover_(evolutionary_algorithm)
    # - https://www.geeksforgeeks.org/machine-learning/crossover-in-genetic-algorithm/
    pop_size: int = len(parents)
    offspring: Population = []
    gene_chance = 0.5
    if pop_size % 2 != 0:
        # ensure even num of parents
        offspring.append(parents.pop())
        pop_size -= 1
    for i in range(0, pop_size-1,2):
        p1,p2 = parents[i], parents[i+1]
        o1,o2 = [],[]
        if random.random() <= crossover_rate:
            for chromosome_idx in range(gait_length):
                c1,c2 = [],[]
                for gene_idx in range(24):
                    # for each gene choose if it should come from p1 or p2
                    # generate a "frame"
                    if random.random() >= gene_chance:
                        # swap over gene
                        c1.append(p2[chromosome_idx][gene_idx])
                        c2.append(p1[chromosome_idx][gene_idx])
                    else:
                        c1.append(p1[chromosome_idx][gene_idx])
                        c2.append(p2[chromosome_idx][gene_idx])
                # append f1 and f2 to offspring
                o1.append(c1)
                o2.append(c2)
            offspring.append(o1)
            offspring.append(o2)
        else:
            # dont cross over put parents in new pop
            offspring.append(p1)
            offspring.append(p2)

    return offspring   # uniform crossover test as single point not working well.


def mutate(population:Population,mut_rate:float) -> Population:
    # mutate should generate a new frame, not a individual float
    for in_idx,individual in enumerate(population):
        for c_idx,chromosome in enumerate(individual):
            for g_idx, gene in enumerate(chromosome):
                if random.random() <= mut_rate:
                    # population[in_idx][c_idx][g_idx] = gene + round(random.uniform(-2,2),2)
                    # population[in_idx][c_idx][g_idx] = round(random.uniform(gene-5,gene+5),2)
                    # population[in_idx][c_idx][g_idx] = round(random.uniform(-90,90),2)
                    new_gene = round(random.uniform(gene-0.5,gene+0.5),9) 
                    # new_gene = round(random.uniform(-50,25),2) 
                    # clamp between -50 & 30
                    if new_gene >= 25:
                        new_gene = 25
                    if new_gene <= -50:
                        new_gene = -50
                    population[in_idx][c_idx][g_idx] =  new_gene
    return population
