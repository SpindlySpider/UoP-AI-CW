# Chromosome Optimization using Genetic Algorithms

## Table of Contents
1. [Glossary](#glossary)
2. [Overview](#overview)
3. [Solution and Approach](#solution-and-approach)
   - [Initialization](#initialization)
   - [Fitness Function Design](#fitness-function-design)
   - [Selection](#selection)
   - [Reproduction](#reproduction)
   - [Termination](#termination)
4. [Design Decisions and Trade-offs](#design-decisions-and-trade-offs)
5. [Code Structure](#code-structure)
6. [Usage Instructions](#usage-instructions)
7. [Technologies and Libraries](#technologies-and-libraries)
8. [Testing and Validation](#testing-and-validation)
9. [Future Improvements](#future-improvements)

---

## Glossary
| Term | Definition |
|------|-------------|
| **GA (Genetic Algorithm)** | A search heuristic inspired by natural selection, used to optimize solutions. |
| **Gait** | A pattern of limb movement during locomotion. |
| **Chromosome** | Encoded representation of a solution (here, a gait). |
| **Gene** | Individual parameter within a chromosome controlling a specific joint motion. |
| **Fitness Function** | Function used to evaluate how well a solution performs the desired task. |

---

## Overview

The goal of this project is to evolve a **complete gait pattern** — a coordinated walking motion — for a simplified **3D spider model** using a **Genetic Algorithm (GA)**.

### Spider Model
- **8 legs**, each with **3 joints**: coxa, femur, and tibia.  
- Total **24 degrees of freedom**.
- A gait is represented as an **n × 24 matrix**, where *n* is the number of time steps per full walking cycle.

### Genetic Algorithm Process
The GA explores the space of possible walking patterns using the following components:

| Stage | Description |
|--------|-------------|
| **Initialization** | Randomly generates an initial population of gait candidates. |
| **Selection** | Chooses fitter individuals based on performance metrics (stability, speed, efficiency). |
| **Reproduction** | Creates new individuals via crossover and mutation. |
| **Termination** | Ends when improvements plateau or a maximum generation count is reached. |

The focus throughout this implementation is to balance:
- **Biological realism**
- **Computational efficiency**
- **Ease of implementation**

The final objective is a **stable, coordinated, and efficient gait**.

---

## Solution and Approach

### Initialization

Each **individual** in the population represents a complete gait.  
Each gait (chromosome) consists of **8 legs × 3 joints = 24 genes**.

#### Gene Encoding
Each joint’s motion is represented by a **sine-wave function** characterized by five parameters:

| Parameter | Description | Range | Design Rationale |
|------------|-------------|--------|------------------|
| **amplitude** | Amplitude of sine wave controlling joint motion | (-55, 30) | Enables both subtle and large joint swings |
| **period** | Frequency of oscillation | (-5, 5) | Allows fast or slow movement cycles |
| **h_offset** | Phase shift of the sine wave | (-5, 5) | Coordinates timing differences between limbs |
| **negative** | Boolean flag inverting the sine wave | {True, False} | Adds diversity without extra dimensions |
| **v_offset** | Baseline joint angle | (-50, 50) | Adjusts resting joint positions |

#### Representation Rationale
- **Sine-wave encoding** produces smooth, periodic motion aligned with natural walking.
- Ensures **continuous, non-abrupt movement**.
- Compact encoding enables **faster optimization**.

**Trade-off:** Restricts solutions to periodic gaits; may exclude more complex or irregular movements.

---

### Fitness Function Design

#### 1. Overview
The **fitness function** measures how well a gait replicates a desired motion pattern.  
A higher fitness value indicates better gait performance.

#### 2. Evaluation Method
The **Mean Squared Error (MSE)** between predicted joint angles and a pre-generated *target gait* is used:

$$
\text{MSE} = \frac{1}{n} \sum (t - p)^2
$$

Fitness is computed as:

$$
\text{fitness} = \frac{1}{1 + \text{MSE}}
$$

Lower error → higher fitness.

#### 3. Design Rationale

| Aspect | Decision | Rationale | Trade-offs |
|---------|-----------|------------|-------------|
| **Fitness Metric** | Mean Squared Error (MSE) | Penalises large deviations, promoting accuracy | Over-penalises outliers |
| **Error Inversion** | `1 / (1 + MSE)` | Normalises to 0–1 range, suitable for GA | Compresses large error values |
| **Target Gait** | Pre-generated once | Improves efficiency, ensures consistency | May bias evolution |
| **Symmetry** | Evaluate only unique 12 joints | Enforces biological realism, reduces cost | Prevents asymmetric gait discovery |
| **Normalization** | Averaged per joint | Fairness between individuals | Requires fixed gait length |
| **Equal Joint Weighting** | All joints contribute equally | Simplifies implementation | Ignores biomechanical differences |

---

### Selection

**Tournament selection** was chosen for its simplicity and control over selection pressure.

#### Method
1. Randomly select a subset of individuals (`num_selected`).
2. Choose the highest-fitness individual from that group.
3. Repeat until enough parents are selected.

| Decision | Justification |
|-----------|---------------|
| Random subset | Maintains diversity, reduces computation |
| Select fittest | Favors better individuals |
| Adjustable subset size | Controls balance between pressure and diversity |

---

### Reproduction

#### 1. Crossover
A **uniform crossover** is implemented:
- Swaps amplitude and vertical offset, horizontal offset and period, and negative flag values between two parents.
- Produces two offspring per crossover operation.

#### 2. Mutation
Each gene undergoes random variation within local bounds:

| Parameter | Mutation Range | Example |
|------------|----------------|----------|
| **Amplitude, v_offset** | ±10 | `random.uniform(val - 10, val + 10)` |
| **h_offset, period** | ±0.5 | `random.uniform(val - 0.5, val + 0.5)` |
| **negative** | Random boolean toggle | Promotes variation |

This ensures **diversity** and prevents **premature convergence**.

---

### Termination

The algorithm terminates after **600 generations**, balancing **runtime constraints** and **solution quality**.  
This limit was determined empirically based on available computational power.

---

## Design Decisions and Trade-offs

- Initial exploration of spider locomotion through video analysis suggested **sinusoidal patterns** matched biological motion.
- Early prototypes using a **300×24 matrix** for full gait encoding were **computationally intensive**.
- The current **parameterized sine-based encoding** offers a **lightweight yet expressive** representation.
- Simplified fitness computation and symmetry assumptions yield faster convergence with reasonable gait realism.

---
## Code Structure

```
project/
│
├── ga/   
│   ├── custom_types.py    # Defines the genes of the chromosome
│   ├── fitness_graph.py   # Generate a generation over fitness score and a genration over best individual graph 
│   ├── fitness.py         # Fitness function implementation                 
│   ├── initial_pop.py     # Generates a random initial population
│   ├── main.py            # To run the genetic algorithm
│   ├── output.py          # Generate text file for a target gait solution
│   ├── reproduce.py       # Mutation, Crossover ad Uniform Crossover implementation
│   ├── selection.py       # Roulette and Tournament Implementation
│   └── target_sol.py      # Generate a target solution along with graph for each of the joints in the spider
│
└── requirements.txt
```

## Usage Instructions

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)

Follow these steps to install and run the Genetic Algorithm project.

---

### 1. Install Python

Ensure that **Python 3.10+** is installed on your system.  
You can verify your version using:

```bash
python --version
```

If Python is not installed, download it from the [official Python website](https://www.python.org/downloads/).

---

### 2. Install Required Libraries

Use the following command to install all dependencies:

```bash
pip install numpy matplotlib
```

---

### 3. Run the Genetic Algorithm

Execute the main program to start the Genetic Algorithm and evolve gait patterns:

```bash
python main.py
```

---

### 4. Generate a Target Gait (Without GA)

To generate a **reference gait** without using the Genetic Algorithm:

```bash
python target_sol.py
```

The resulting gait data will be saved for later comparison and testing. The code generates a sol.txt file that contains a 300x24 matrix that can be imported into matlab.

---

## MATLAB Visualization

To visualize an evolved gait in **MATLAB**, use the following script:

```matlab
v = readmatrix('ga/sol.txt');
A = deg2rad(v);

for idx = 1:size(v,1)
    plot_spider_pose(A(idx,:));
    pause(0.001);
end
```

**Explanation:**
- `readmatrix()` loads the gait data from `sol.txt`.  
- `deg2rad()` converts joint angles to radians.  
- The loop visualizes each time step, animating the spider’s movement.

---

## Technologies and Libraries

| Library | Purpose |
|----------|----------|
| [NumPy](https://numpy.org/) | Numerical computation and matrix operations |
| [Matplotlib (pyplot)](https://matplotlib.org/) | Visualization and plotting of gait data |
| **random** | Randomized initialization and mutation processes |
| **math** | Trigonometric and mathematical calculations for gait motion |

---

## Testing and Validation

| Test Type | Description |
|------------|-------------|
| **Convergence Tracking** | Recorded and plotted fitness values across generations to monitor improvement |
| **Visual Verification** | Assessed gait smoothness and motion stability via MATLAB visualization |
| **Parameter Sensitivity** | Tested robustness by varying mutation rates and population sizes |

---

## Future Improvements

1. Introduce **weighted joint importance** to emphasize load-bearing joints.  
2. Incorporate **energy efficiency** and **stability** into a multi-objective fitness function.  
3. Allow **multiple target gaits** to evolve more diverse movement styles.  
4. Replace fixed generation limits with **adaptive convergence detection**.  
5. Combine the Genetic Algorithm with **reinforcement learning** for enhanced adaptability.

---

