import random
from custom_types import Chromosome, Individual, Population

def gen_individual(gait_length:int) -> Individual:
    individual: Individual = []
    for _ in range(gait_length):
        chromosome: Chromosome = []
        for _ in range(24):
            chromosome.append(round(random.uniform(-90,90),2))
        individual.append(chromosome)
    return individual

def gen_population(max_pop:int,gait_length:int) -> Population:
    population:Population = []
    for _ in range(max_pop):
        population.append(gen_individual(gait_length))
    return population
