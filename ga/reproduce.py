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
            for joint_idx in range(24):
                c1,c2 = [],[]
                for gene_idx in range(4):
                    # for each gene choose if it should come from p1 or p2
                    # generate a "frame"
                    if random.random() >= gene_chance:
                        # swap over gene
                        c1.append(p2[joint_idx][gene_idx])
                        c2.append(p1[joint_idx][gene_idx])
                    else:
                        c1.append(p1[joint_idx][gene_idx])
                        c2.append(p2[joint_idx][gene_idx])
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
            m,p,o,n = chromosome
            val_arr = [m,p,o,n]
            for g_idx in range(4):
                # will either be mag, p, offset, n
                if random.random() <= mut_rate:
                    if g_idx < 3:
                        val_arr[g_idx] = round(random.uniform(val_arr[g_idx]-0.1,val_arr[g_idx]+0.1),9)
                    else:
                        # bool
                        val_arr[g_idx] = not val_arr[g_idx]
                    population[in_idx][c_idx] = tuple(val_arr)


    return population
