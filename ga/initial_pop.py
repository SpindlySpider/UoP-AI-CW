import sys
from pathlib import Path
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ga.custom_types import Chromosome, Individual, Population, Period, H_offset, Amplitude, Negative, V_offset
except ImportError:
    from custom_types import Chromosome, Individual, Population, Period, H_offset, Amplitude, Negative, V_offset

def gen_individual() -> Individual:
    """
    Generate a single individual (a full gait pattern) for the genetic algorithm population.

    Each individual consists of 12 chromosomes, each representing one joint's sine wave parameters
    (since paired legs share the same parameters, only 12 are needed for 24 total joints).

    Parameters:
        gait_length (int): The number of time steps representing a full gait cycle (not directly used
                           here but retained for compatibility and potential future scaling).

    Returns:
        Individual: A list of 12 chromosomes, where each chromosome is a tuple:
                    (amplitude, period, h_offset, negative, v_offset)
    """
    individual: Individual = []

    for _ in range(12):
        # Each chromosome encodes a sine wave controlling joint motion
        # 8 legs × 3 joints = 24 joints, but symmetric legs share parameters → 12 unique sets.

        # Define search space boundaries for each parameter.
        period: Period = random.uniform(-5, 5)          # Frequency of joint oscillation
        h_offset: H_offset = random.uniform(-5, 5)      # Phase shift (horizontal offset)
        amplitude: Amplitude = random.uniform(-55, 30)  # Amplitude of joint movement
        negative: Negative = bool(random.randint(0, 1)) # Whether to invert sine wave motion
        v_offset: V_offset = random.uniform(-50, 50)    # Vertical offset (baseline joint position)

        # A chromosome is defined as a tuple of sine wave parameters.
        chromosome: Chromosome = (amplitude, period, h_offset, negative, v_offset)
        individual.append(chromosome)

    return individual


def gen_population(max_pop: int) -> Population:
    """
    Generate the initial population for the genetic algorithm.

    Parameters:
        max_pop (int): The total number of individuals to create in the initial population.
        gait_length (int): The number of time steps in one gait cycle (passed to gen_individual).

    Returns:
        Population: A list of individuals, where each individual is a list of chromosomes.
    """
    population: Population = []

    for _ in range(max_pop):
        population.append(gen_individual())

    return population
