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
        self.max_pop: int = 50

    def gen_individual(self) -> Individual:
        # set seed so we can regen results
        # random.seed(self.seed)
        individual: Individual = []
        joint_names: list[str] = ["tibia", "coxa", "femur"]
        for _ in range(self.max_gait):
            chromosome: Chromosome = []
            for gene in range(1, 24):
                # joint: str = joint_names[gene % 3]
                # append random value between constrained angle for each joint
                # min, max = self.angle_constraint[joint]
                # chromosome.append(random.uniform(min,max))
                chromosome.append(random.random())
            individual.append(chromosome)
        return individual

    def gen_population(self) -> Population:
        for _ in range(self.max_pop):
            self.population.append(self.gen_individual())
        return self.population
