import random
from typing import Optional

type Chromosome = list[float]
type Individual = list[Chromosome]


class population():
    # object used to track and hold population of GA
    def __init__(self, seed):
        # population seed so we can regenerate results.
        self.seed = seed
        # init population list - optional for init
        self.population: list[Optional[Individual]] = []
        # constraint on min and max angles of joints
        self.angle_constraint: dict[str:Optional[tuple(float, float)]] = {
            "coxa": (-90, 90), "femur": (-90, 90), "tibia": (-90, 90)}

        # max gait = number of entries of poses for each individual
        self.max_gait: int = 10
        self.max_pop: int = 1

    def gen_individual(self) -> Individual:
        # set seed so we can regen results
        random.seed(self.seed)
        individual: Individual = []
        joint_names = ["tiba", "coxa", "femur"]
        for _ in range(self.max_gait):
            chromosome: Chromosome = []
            for gene in range(1, 24,):
                joint: str = joint_names[gene % 3]
                # append random value between constrained angle for each joint
                chromosome.append(random.uniform(self.angle_constraint[joint]))
            individual.append(chromosome)
        return individual

    def gen_population(self) -> list[Individual]:
        for individual in range(self.max_gait):
            self.population.append(self.gen_indvidual())
        return self.population
