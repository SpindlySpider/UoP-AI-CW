import random
from custom_types import Chromosome, Individual, Population

class Population_obj():
    # object used to track and hold population of GA
    def __init__(self, gait_length:int,max_pop:int):
        # init population list - optional for init
        self.population: Population = []

        # max gait = number of entries of poses for each individual
        self.max_gait: int = gait_length
        self.max_pop: int = max_pop

    def gen_individual(self) -> Individual:
        # set seed so we can regen results
        # random.seed(self.seed)
        individual: Individual = []
        for _ in range(self.max_gait):
            chromosome: Chromosome = []
            for _ in range(24):
                chromosome.append(round(random.uniform(-90,90),2))
            individual.append(chromosome)
        return individual

    def gen_population(self) -> Population:
        for _ in range(self.max_pop):
            self.population.append(self.gen_individual())
        return self.population
