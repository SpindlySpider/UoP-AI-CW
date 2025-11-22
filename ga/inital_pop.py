import random
from ga.custom_types import Chromosome, Individual, Population, Period, H_offset, Magnitude, Negative, V_offset

def gen_individual(gait_length:int) -> Individual:
    individual: Individual = []
    for _ in range(12):
        # 8 because legs 0,2 will move the same, legs 1,3 move the same, then repeat for other side
        # seeding values for search spaces
        # generate values for each sine wave for each joint
        period: Period = random.uniform(-5,5)
        h_offset: H_offset = random.uniform(-5,5)
        magnitude: Magnitude = random.uniform(-55,30)
        negative: Negative = bool(random.randint(0,1))
        v_offset: V_offset = random.uniform(-50,50)
        chromosome: Chromosome = (magnitude,period,h_offset,negative,v_offset)
        individual.append(chromosome)
    return individual

def gen_population(max_pop:int,gait_length:int) -> Population:
    population:Population = []
    for _ in range(max_pop):
        population.append(gen_individual(gait_length))
    return population
