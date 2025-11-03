from custom_types import Population
import random

def crossover(parents:Population,gait_length:int,crossover_rate:float) -> Population:
    '''
    Performs single-point crossover on a population of individuals.
    Args:
        parents (Population): The population of individuals to perform crossover on.
        gait_length (int): The number of frames in the gait cycle.
        crossover_rate (float): The probability of performing crossover on a pair of parents.
    Returns:
        Population: The new population of individuals after crossover.
    '''
    # size of the population
    pop_size: int = len(parents)
    # list to hold new offspring
    offspring: Population = []
    # check if odd number of parents
    if pop_size % 2 != 0:
        # ensure even num of parents
        offspring.append(parents.pop())
        pop_size -= 1
    
    # perform crossover in pairs
    for i in range(0, pop_size-1,2):
        # get two adjacent individuals that will be parents
        p1,p2 = parents[i], parents[i+1]
        # check if crossover should occur
        if random.random() <= crossover_rate:
            # perform single point crossover by selecting a random cross point and swapping genes
            cross_point:int = random.randint(0,gait_length)
            offspring.append(p1[:cross_point] + p2[cross_point:])
            offspring.append(p2[:cross_point] + p1[cross_point:])
        else:
            # dont cross over put parents in new pop
            offspring.append(p1)
            offspring.append(p2)
    return offspring


def uniform_crossover(parents:Population,gait_length:int,crossover_rate:float) -> Population:
    '''
    Performs uniform crossover on a population of individuals. 
    Args:
        parents (Population): The population of individuals to perform crossover on.
        gait_length (int): The number of frames in the gait cycle.
        crossover_rate (float): The probability of performing crossover on a pair of parents.
    Returns:
        Population: The new population of individuals after crossover.
    '''
    # explanation found here: 
    # - https://en.wikipedia.org/wiki/Crossover_(evolutionary_algorithm)
    # - https://www.geeksforgeeks.org/machine-learning/crossover-in-genetic-algorithm/
    # size of the population
    pop_size: int = len(parents)
    # list to hold new offspring
    offspring: Population = []
    gene_chance = 0.5
    # check if odd number of parents
    if pop_size % 2 != 0:
        # ensure even num of parents
        offspring.append(parents.pop())
        pop_size -= 1

    # perform crossover in pairs
    for i in range(0, pop_size-1,2):
        # get two adjacent individuals that will be parents
        p1,p2 = parents[i], parents[i+1]
        # store offspring
        o1,o2 = [],[]
        # check if crossover should occur
        if random.random() <= crossover_rate:
            # perform uniform crossover by swapping genes based on gene chance
            for joint_idx in range(12):
                # store chromosomes for offspring
                c1,c2 = [],[]
                for gene_idx in range(5):
                    # for each gene choose if it should come from p1 or p2
                    #check if the gene should be swapped
                    if random.random() >= gene_chance:
                        # swap over gene
                        c1.append(p2[joint_idx][gene_idx])
                        c2.append(p1[joint_idx][gene_idx])
                    else:
                        c1.append(p1[joint_idx][gene_idx])
                        c2.append(p2[joint_idx][gene_idx])
                # append chromosome to offspring
                o1.append(c1)
                o2.append(c2)
            # append offspring to new population
            offspring.append(o1)
            offspring.append(o2)
        else:
            # dont cross over put parents in new pop
            offspring.append(p1)
            offspring.append(p2)

    return offspring   # uniform crossover test as single point not working well.


def mutate(population:Population,mut_rate:float) -> Population:
    '''
    Performs mutation on a population of individuals.
    Args:
        population (Population): The population of individuals to mutate.
        mut_rate (float): The probability of mutating each gene.
    Returns:
        Population: The new population of individuals after mutation.
    '''
 
    # for each individual in the population, check each gene in each chromosome
    for in_idx,individual in enumerate(population):
        # for each gene in each chromosome in each individual, check if it should be mutated
        for c_idx,chromosome in enumerate(individual):
            # unpack chromosome into separate genes
            amplitude,period,horizontal_offset,negative,vertical_offset = chromosome
            # put genes into a list for easier mutation
            val_arr = [amplitude,period,horizontal_offset,negative,vertical_offset]
            # check each gene for mutation
            for g_idx in range(4):
                # will either be amplitude, period, offset, negative
                # check if gene should be mutated
                if random.random() <= mut_rate:
                    # mutate gene: amplitude and vertical offset
                    if g_idx == 0 or g_idx == 4:
                        new_gene = round(random.uniform(val_arr[g_idx]-10,val_arr[g_idx]+10),9)
                        new_gene = new_gene if new_gene <= 50 else 50
                        new_gene = new_gene if new_gene >= -50 else -50
                        val_arr[g_idx] = new_gene
                    # mutate gene: horizontal offset and period 
                    elif g_idx < 3:
                        new_gene = round(random.uniform(val_arr[g_idx]-0.5,val_arr[g_idx]+0.5),9)
                        new_gene = new_gene if new_gene <= 10 else 10
                        new_gene = new_gene if new_gene >= -10 else -10
                        val_arr[g_idx] = new_gene
                    # mutate gene: negative
                    else:
                        val_arr[g_idx] = not val_arr[g_idx]
                    population[in_idx][c_idx] = tuple(val_arr)


    return population
