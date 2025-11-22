import sys
from pathlib import Path
import math

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ga.custom_types import Individual, Gait
    import ga.target_sol as target_sol
except ImportError:
    from custom_types import Individual, Gait
    import target_sol as target_sol

class Fitness:
    """
    A class to evaluate the fitness of an individual gait using a target gait as reference.

    The fitness is calculated based on the Mean Squared Error (MSE) between the generated
    gait (from the individual's chromosomes) and a target gait solution. Lower error means
    a higher fitness score.

    Attributes:
        target_individual (Gait): A reference gait used for comparison.
        gait_length (int): The number of time steps representing one gait cycle.
    """

    def __init__(self, gait_length: int):
        """
        Initialize the fitness evaluator.

        Parameters:
            gait_length (int): The number of time steps in the gait cycle.

        Notes:
            The target gait is generated once during initialization to avoid recomputation.
        """
        self.target_individual: Gait = target_sol.random_sol(gait_length)
        self.gait_length: int = gait_length

    def get_fitness(self, individual: Individual) -> float:
        """
        Compute the fitness of a given individual by comparing it to the target gait.

        The comparison uses Mean Squared Error (MSE) for each joint (coxa, femur, tibia),
        normalized and inverted so that higher fitness corresponds to lower error.

        Parameters:
            individual (Individual): The individual whose gait is to be evaluated.

        Returns:
            float: The total fitness value for the individual. Higher is better.
        """
        # Track cumulative error per joint type
        fit_dict: dict[str, float] = {"coxa": 0, "femur": 0, "tibia": 0}
        joint_names: list[str] = ["coxa", "femur", "tibia"]

        # Generate gait (predicted joint movements) for this individual
        gait: Gait = gen_gait(individual, self.gait_length)

        # Limit comparison to the first 50 time steps for efficiency
        length = self.gait_length if self.gait_length < 50 else 50

        for chromosome_idx in range(length):
            # Evaluate left side joints (indices 0–5)
            for gene_idx in range(6):
                joint: str = joint_names[gene_idx % 3]
                target_val: float = self.target_individual[chromosome_idx][gene_idx]
                pred_val: float = gait[chromosome_idx][gene_idx]
                err: float = (target_val - pred_val) ** 2
                fit_dict[joint] += err

            # Evaluate right side joints (indices 13–18, mirror pattern)
            for gene_idx in range(13, 19):
                joint: str = joint_names[gene_idx % 3]
                target_val: float = self.target_individual[chromosome_idx][gene_idx]
                pred_val: float = gait[chromosome_idx][gene_idx]
                err: float = (target_val - pred_val) ** 2
                fit_dict[joint] += err

        # Normalize errors and invert (1 / (1 + MSE)) for fitness
        for joint in joint_names:
            j: float = fit_dict[joint]
            j = (j / (4 * self.gait_length))  # Average per joint
            j = 1 / (1 + j)                   # Invert to make higher = better
            fit_dict[joint] = j

        # Combine fitness across all joint types equally
        fit_val: float = fit_dict["coxa"] + fit_dict["femur"] + fit_dict["tibia"]

        return fit_val


def gen_gait(individual: Individual, gait_length: int) -> Gait:
    """
    Generate a gait sequence from an individual's chromosomes.

    Each chromosome defines a sine wave controlling a joint's motion.
    The gait sequence is built by evaluating these sine functions over time.

    Parameters:
        individual (Individual): The list of chromosomes defining the gait.
        gait_length (int): The number of time steps to simulate.

    Returns:
        Gait: A list of lists containing predicted joint angles over time.
    """
    gait: Gait = []

    for idx in range(gait_length):
        gait.append([])
        prev_limb: list[float] = []

        for chromosome in individual:
            amplitude, period, offset, neg, v_offset = chromosome

            # Compute sine value for current timestep
            sin_val: float = (period * idx) + offset
            sin_val = -sin_val if neg else sin_val
            predict: float = (amplitude * math.sin(sin_val)) + v_offset

            # Clamp joint angle to minimum threshold
            predict = predict if predict > -50 else -50

            prev_limb.append(predict)
            gait[idx].append(predict)

            # Duplicate values for symmetric legs (6 joints mirrored)
            if len(prev_limb) == 6:
                gait[idx] = gait[idx] + prev_limb[:3] + prev_limb[3:]
                prev_limb = []

    return gait
