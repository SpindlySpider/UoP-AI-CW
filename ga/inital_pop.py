import random
from custom_types import Chromosome, Individual, Population

def gen_individual(gait_length:int) -> Individual:
    individual: Individual = []
    for _ in range(gait_length):
        chromosome: Chromosome = []
        for _ in range(24):
            # seeding values for search spaces (target sol is between -50 and 25)
            chromosome.append(round(random.uniform(-50,25),11))
        individual.append(chromosome)
    return individual

def gen_population(max_pop:int,gait_length:int) -> Population:
    population:Population = []
    for _ in range(max_pop):
        population.append(gen_individual(gait_length))
    return population
