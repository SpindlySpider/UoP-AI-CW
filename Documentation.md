# Chromosome Optimization using Genetic Algorithms
# *Initial Population Design Decisions*

This document explains the design choices and trade-offs made in the implementation of the initial population generation code for the spider gait optimisation task.

## 1. Overview

The goal of this stage is to generate a diverse, valid initial population of potential solutions (gaits) for the genetic algorithm (GA).
Each spider has 8 legs, each with 3 joints (coxa, femur, tibia), resulting in 24 degrees of freedom.
A chromosome encodes the parameters controlling the motion of a single joint, while an individual represents a full gait pattern composed of multiple chromosomes.

## 2. Chromosome Encoding

Each chromosome is represented as a 5-tuple:
(magnitude, period, h_offset, negative, v_offset)

| Parameter | Description | Range | Design Rationale |
|------------|-------------|--------|------------------|
| **magnitude** | Amplitude of the sine wave controlling joint movement | (-55, 30) | Allows both subtle and large joint swings, enabling diverse gait styles |
| **period** | Controls the frequency of oscillation | (-5, 5) | Enables both slow and fast movements; sign reversal allows mirrored motion |
| **h_offset** | Phase shift of the sine wave | (-5, 5) | Introduces coordination or timing offsets between limbs |
| **negative** | Boolean flag to invert the sine function | {True, False} | Adds simple directional diversity without increasing dimensionality |
| **v_offset** | Baseline (mean) joint angle | (-50, 50) | Adjusts resting positions to simulate different standing postures |


### Reasoning
A sine-wave representation was selected as it gives smooth, periodic motion that is naturally matched to walking or crawling behaviour.
This representation encodes gait dynamics and supports realistic and continuous motion without sudden angle changes.

### Trade-offs
Compact encoding enables faster optimisation and smooth motion.
Naturally models periodic gaits.
~
Restricts possible gaits to those expressible as sinusoidal motions.
May limit irregular or complex leg trajectories.

## 3. Symmetry Reduction (12 Chromosomes Instead of 24)

The spider’s body exhibits bilateral symmetry which means opposite legs often move in mirrored patterns.
Rather than defining 24 independent joint controllers, only 12 unique chromosomes are generated, each shared between mirrored leg pairs.

### Reasoning
Reducing the dimensionality from 24 to 12 parameters decreases computation time and search complexity by half while preserving realistic movement patterns.

### Trade-offs
Simplifies the genetic search space and speeds convergence.
Encourages biologically plausible symmetric gait patterns.
~
Reduces the diversity of possible solutions, potentially excluding asymmetric but efficient gaits.

## 4. Randomised Parameter Initialization

Each of the sine wave parameters is plotted randomly within set boundaries.

### Reasoning
Random initialization guarantees that there is good diversity in the population initially. This diversity, especially in initial generations, is important so that the GA can explore a broad range of motion strategies before it can specialize towards optimal gaits.

### Trade-offs
Encourages exploration and avoids premature convergence. ~ Randomly generated gaits can be physically impossible to execute, particularly for the initial runs.

## 5. Population Generation Functions

(gen_individual(gait_length: int) -> Individual)
Produces one individual (one entire gait).
gait_length is not used here directly, but kept in case of future support for extensions (for instance, time-scaling gait or dynamic testing).
(gen_population(max_pop: int, gait_length: int) -> Population)
Produces the whole population by recursively calling gen_individual.

### Design Rationale
Modularity: Separating the two functions improves readability, testing, and reusability.
Type Annotations: Using aliases (Chromosome, Individual, Population) enforces consistency across the codebase.

### Trade-offs
Easy to extend (e.g., add seeding, mutations, or gait visualisation).
Improves maintainability and code clarity.
~
Slight overhead from repeated function calls (negligible at this scale).

## 6. Parameter Range Selection

The chosen numeric ranges reflect a balance between biological plausibility and search diversity:
- Wide enough to cover diverse gait possibilities.
- Narrow enough to avoid extreme, unrealistic joint configurations.
- These ranges can later be tuned empirically once simulation feedback is available.

## Diversity vs. Realism Trade-off

The design intentionally favours diversity in the early stages of evolution, accepting that some individuals will perform poorly or unrealistically.
Over successive generations, the GA’s selection and mutation processes refine these individuals into physically coherent gait patterns.

## Future Improvements

Introduce elitism or seeded individuals to guide early evolution.
Implement adaptive parameter bounds based on observed fitness.
Allow controlled asymmetry for advanced gaits by introducing mirroring offsets.

<br>

# *Fitness Function Design Decisions*
This section explains the design reasoning, choice of parameters, and trade-offs made during the implementation of the fitness evaluation of the genetic algorithm.

## 1. Overview

The fitness function quantifies how well an individual gait matches a desired motion pattern.
A higher fitness value indicates a more successful gait configuration.
The comparison is performed against a target gait, generated once during initialization for computational efficiency.

## 2. Fitness Evaluation Method
The fitness value is based on the Mean Squared Error (MSE) between predicted joint angles and the target gait over time.

$$
\text{MSE} = \frac{1}{n}\sum (t - p)^2
$$

Each joint type: coxa, femur, and tibia, is evaluated separately, and their errors are averaged and inverted using:

$$
\text{fitness} = \frac{1}{1 + \text{MSE}}
$$

This ensures that lower error -> higher fitness.

## 3. Design Rationale
| Design Aspect | Decision | Rationale | Trade-offs |
|----------------|-----------|------------|-------------|
| **Fitness metric** | Mean Squared Error (MSE) | Penalises larger deviations more heavily, encouraging stable, accurate motion | May over-penalise outliers in joint movement |
| **Error inversion** | `1 / (1 + MSE)` | Converts error into a bounded fitness (0–1), making it suitable for GA selection | Compresses large error ranges, reducing sensitivity for poor individuals |
| **Target gait** | Pre-generated once | Reduces repeated computation and ensures consistent evaluation | Fixed reference may bias evolution toward one gait type |
| **Symmetry assumption** | Only unique joints (12) evaluated; mirrored joints skipped | Reduces redundant computation and enforces biologically plausible symmetry | Limits discovery of asymmetric gait patterns |
| **Normalization** | Average per joint over gait length | Ensures fairness between individuals with different cycle lengths | Requires consistent gait_length across population |
| **Weighting** | Equal across coxa, femur, tibia | Simplicity and balanced joint contribution | May ignore relative biomechanical importance of joints |

## 4. gen_gait() Design

The gen_gait function reconstructs the full gait motion by evaluating each chromosome’s sine-wave parameters over time.

| Parameter | Description | Rationale |
|------------|-------------|------------|
| **mag** | Amplitude of motion | Controls the range of leg swing |
| **period** | Frequency of oscillation | Controls walking speed |
| **offset** | Phase shift | Synchronizes legs for coordinated gait |
| **neg** | Boolean inversion | Enables mirrored motion between left and right sides |
| **v_offset** | Baseline angle | Adjusts neutral joint position |

The generated gait is stored as a list of joint angles for each timestep. The limbs are symmetrical and are duplicated from one side to another for reducing computational overhead.

## 5. Efficiency Considerations

Only the first 50 timesteps are evaluated to save time during early evolution cycles.
Gait generation and comparison use basic arithmetic and trigonometric functions, computationally inexpensive and easily vectorizable later.
Fitness evaluation scales linearly with population_size × gait_length, ensuring predictable performance.

## 6. Normalization and Scaling

Each joint’s cumulative error is divided by (4 × gait_length) before inversion.
This normalization prevents penalizing individuals with more extended gait sequences unfairly and maintains all values of fitness within the same numeric range.

## 7. Future Improvements

Introduce weighted joint importance, giving higher value to load-bearing joints.
Evaluate energy efficiency or stability metrics alongside motion accuracy.
Allow multiple target gaits for multi-objective optimization (speed, stability, symmetry).
Replace fixed 50-step truncation with adaptive evaluation length based on movement periodicity.


<br>

# *Selection Step Design Decisions*

## Overview
Selection chooses individuals from the population to become parents. The code implements two methods: **cumulative probability selection** and **tournament selection**.

---

## 1. Cumulative Probability Selection

### Method
- The fitness values are normalized and accumulated into a cumulative distribution.
- A random number is drawn for every selection.
- The first individual whose cumulative probability reaches and surpasses the random number will be selected.

### Design Rationale

| Decision | Justification |
|-----------|---------------|
| Normalize fitness values | Ensures probabilities sum to 1 and higher-fitness individuals are more likely to be selected |
| Cumulative distribution | Efficient mapping from random number to selected individual |
| Random sampling | Introduces stochasticity to maintain diversity |
| Subtracting 1 from index | Matches the original implementation's selection behavior |

---

## 2. Tournament Selection

### Method
- Randomly select `num_selected` individuals per tournament.
- Choose the individual with the highest fitness from the tournament.
- Repeat until a full set of parents is selected.

### Design Rationale

| Decision | Justification |
|-----------|---------------|
| Random subset for each tournament | Reduces computational cost while maintaining selection pressure |
| Choosing highest-fitness participant | Directly favors stronger solutions |
| Adjustable `num_selected` | Allows control of selection pressure versus diversity |
