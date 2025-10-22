# What To Do (Step-by-Step)
## Step 1: Encode the Chromosome
- Represent each individual in the population as a 1×24 vector of real numbers (angles in radians).
- Each gene can be in a reasonable range, e.g. [-π/4, π/4] or whatever joint limits make sense.
- Chromosomes must be encoded so crossover and mutation can operate cleanly (e.g., numpy arrays in Python or MATLAB vectors).
Mark weight: 15 marks
You must clearly explain your encoding logic.

## Step 2: Design the Fitness Function
-This is the most important part. The GA uses the fitness function to decide which poses are “better”.

Possible fitness objectives:
- Symmetry – left and right legs should mirror each other.
- Stability – center of mass should be over the legs.
- Balance – not tipping over (e.g., height of body stable).
- Target position – maybe one leg or body moves toward a goal.
- Energy efficiency or smoothness – fewer extreme joint angles.
Mark weight: 15 marks
You must explain what the fitness measures and why it makes sense for a spider pose.

## Step 3: Implement Genetic Operators

You must implement the core GA steps:
### Initialization
Create an initial population (e.g., 50–200 chromosomes).
Each chromosome = 24 random angles within valid limits.

### Selection
Pick parents for reproduction.
Possible methods:
- Roulette Wheel (probability ∝ fitness)
- Tournament Selection (best of random subset)
- Rank Selection

### Crossover (Recombination)
Mix genes between two parents.
Methods:
- 1-point crossover
- 2-point crossover
- Uniform crossover (swap each gene with a probability)

### Mutation
Slightly change random genes (add small random noise).
Prevents premature convergence.
Mutation rate: typically 0.5–5%.

### Elitism (optional)
Keep the best chromosome(s) from one generation to the next unchanged.

### Termination condition:
e.g., number of generations (100–500), or convergence (fitness stops improving).

## Step 5: Performance Visualization
You must show that your GA works using plots or animations, for example:
- Fitness vs. generations (line chart of best and average fitness)
- 3D pose visualization (using the provided spider model)
- Animations showing improvement over time
- Console logs showing progress
Mark weight: 10 marks

## Step 6: Explanation and Justification
For each design choice (encoding, fitness, operators), explain why you chose it and what alternatives you considered.
You can either:
- Write these as inline comments in your code, or
- Write short .txt explanation files (one per component).
Mark weight: 10 marks

## Step 7: Code Quality and Execution
Your code should:
- Run without errors
- Be modular and readable
- Have comments or docstrings
- Be easy to run (e.g., with a clear entry point like main.m or main.py)
Mark weight: 10 marks