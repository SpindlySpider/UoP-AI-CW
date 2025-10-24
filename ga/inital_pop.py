import random
from custom_types import Chromosome, Individual, Population


class Population_obj():
    # object used to track and hold population of GA
    def __init__(self, seed):
        # population seed so we can regenerate results.
        self.seed: int = seed
        # init population list - optional for init
        self.population: Population = []
        # constraint on min and max angles of joints
        self.angle_constraint: dict[str, tuple[float, float]] = {
            "coxa": (-90, 90),
            "femur": (-90, 90),
            "tibia": (-90, 90)
        }

        # max gait = number of entries of poses for each individual
        self.max_gait: int = 300
        self.max_pop: int = 500

    def gen_individual(self) -> Individual:
        # set seed so we can regen results
        # random.seed(self.seed)
        individual: Individual = []
        for _ in range(self.max_gait):
            chromosome: Chromosome = []
            for idx in range(24):
                if idx %3 == 0:
                    chromosome.append(random.random())
                else:
                    chromosome.append(0)
            individual.append(chromosome)
        return individual

    def gen_population(self) -> Population:
        for _ in range(self.max_pop):
            self.population.append(self.gen_individual())
        return self.population
