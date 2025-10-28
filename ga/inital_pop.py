import random
from custom_types import Chromosome, Individual, Population, Period, H_offset, Magnitude, Negative

def gen_individual(gait_length:int) -> Individual:
    individual: Individual = []
    for _ in range(24):
        # seeding values for search spaces
        # generate values for each sine wave for each joint
        period: Period = random.uniform(-2,2)
        h_offset: H_offset = random.uniform(-2,2)
        magnitude: Magnitude = random.uniform(-180,180)
        negative: Negative = bool(random.randint(0,1))
        chromosome: Chromosome = (magnitude,period,h_offset,negative)
        individual.append(chromosome)
    return individual

def gen_population(max_pop:int,gait_length:int) -> Population:
    population:Population = []
    for _ in range(max_pop):
        population.append(gen_individual(gait_length))
    return population
