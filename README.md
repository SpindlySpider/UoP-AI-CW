# Spider Gait Generation & Prediction - Main CLI Documentation

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [System Architecture](#system-architecture)
4. [CLI Features](#cli-features)
5. [Installation & Usage Guide](#installation-&-usage-guide)
6. [Component Integration](#component-integration)
7. [Configuration Options](#configuration-options)
8. [File Outputs](#file-outputs)

---

## Overview

The **main.py** file serves as the command-line interface (CLI) for the Spider Generation & Prediction System, providing an access generating and predicting spider gait using the following:

**Genetic Algorithm (GA)** - Evolves optimal gait patterns through comparison to target solution.

**Neural Network (NN)** - Learns to predict sequential gait frames from training data:
    - From Scratch Implementation - Pure Python implementation.
    - PyTorch Implementation - GPU-accelerated deep learning framework.


---

## Project Structure

```
├── .git/                          # Git version control
├── .gitignore                     # Git ignore rules

├── main.py                        # Main CLI entry point
├── utils.py                       # CLI utility functions
├── requirements.txt               # Python dependencies
├── README.md                      # This documentation
│
├── ga_results.txt                 # Genetic algorithm output
├── nn.pickle                      # Trained NumPy neural network
├── nn_results.txt                 # NumPy NN training results
├── nn_predict_results.txt         # NumPy NN prediction output
├── nn_pytorch.pth                 # Trained PyTorch neural network
│
├── ga/                            # Genetic Algorithm module
│   ├── images/                    # GA visualisation
│   ├── GA_documentation.md        # GA documentation
│   ├── custom_types.py            # Type definitions
│   ├── fitness.py                 # Fitness evaluation
│   ├── fitness_graph.py           # Fitness visualisation
│   ├── initial_pop.py             # Population initialisation
│   ├── main.py                    # GA entry point
│   ├── output.py                  # Result output handling
│   ├── reproduce.py               # Crossover & mutation
│   ├── selection.py               # Selection operators
│   ├── target_sol.py              # Target solution generator
│   ├── results/                   # Resulting GAs after being run
│
├── nn/                            # Custom NumPy Neural Network
│   ├── doc-images/                # Documentation images
│   ├── NN_documentations.md       # NN documentation
│   ├── activation_functions.py    # Activation implementations
│   ├── error_funcs.py             # Loss functions
│   ├── graph_results.py           # Training visualisation
│   ├── input_data.py              # Data generation
│   ├── load_and_predict.py        # Inference & prediction
│   ├── main.py                    # NN training entry point
│   ├── neural_network.py          # Network architecture
│   ├── optimiser.py               # Gradient descent & Adam
│   ├── serialise.py               # Model save/load (pickle)
│   └── training.py                # Training loop
│
│── pytorch_nn/                    # PyTorch Neural Network
│   ├── graph_results.py           # Training visualisation
│   ├── load_and_predict.py        # PyTorch inference
│   ├── main.py                    # PyTorch training entry point
│   ├── serialise.py               # Model save/load (.pth)
│   ├── torch_model.py             # PyTorch model (nn.Module)
│   └── torch_training.py          # PyTorch training loop
│ 
├── .gitignore                     # Specifies files Git should ignore
├── README.md                      # This documentation
├── ga_results.txt                 # Genetic algorithm output
├── main.py                        # Main CLI entry point
├── nn.pickle                      # Trained NumPy neural network
├── nn_predict_results.txt         # NumPy NN prediction output
├── requirements.txt               # Python dependencies
└── utils.py                       # CLI utility functions
```

---

## System Architecture

```
main.py (CLI Entry Point)
├── Genetic Algorithm Module (ga/)
│   └── [See GA_documentation.md for details]
├── Neural Network Module (nn/)
│   └── [See NN_documentation.md for details]
└── Utilities Module (utils.py)
    ├── get_choice()        # Interactive menu system
    ├── get_defaults()      # Configuration management
    ├── display_defaults()  # Display configuration
    ├── modify_default()    # Parameter customisation
    └── handle_lists()      # Input validation
```

### Components

For further details about the GA and NN documentation please see:

[genetic algorithm](./ga/GA_documentation.md) docs found at `ga/GA_documentation.md`.

[neural network](./nn/NN_documentations.md) docs found at `nn/NN_documentation.md`.


---

## CLI Features

### 1. Interactive Menu System

The CLI provides a user-friendly, menu-driven interface with hierarchical navigation:

```
Main Menu
├── Genetic Algorithm
├── Neural Network
│   ├── Implementation Choice
│   │   ├── From Scratch Implementation
│   │   └── PyTorch
│   ├── Train
│   └── Predict
│       ├── Random Input
│       └── Manual Input
└── Exit
```

#### Menu Navigation
Users are able to select different options using numbers (e.g. 0,1,2). This menu navigation also includes input validation, with type checking when modifying default values. 

### 2. Configuration Management

Both the GA and NN allow users to customise parameters form inside the CLI. There are a few quality of life features such as:
- displaying the current configuration before running the GA or NN
- allowing for saving / discarding modifications.

#### Example Workflow:
```
starting with these defaults:
  0) hidden layers: [128, 64, 32]
  1) learning rate: 0.01
  2) nn save: nn.pickle
  ...
Would you like to change anything? (y/N): y
```

### 3. Input Handling

The CLI provides flexible input methods for different use cases:

#### List Input
- **Fixed-Length Lists**: For neural network predictions (24 joint angles)
- **Variable-Length Lists**: For hidden layer configuration
- **Type Validation**: Ensures correct data types (float/int)

#### Input Modes for prediction
- **Random Generation**: Generates a size 24 array of random float values to test the neural networks predictions.
- **Manual Entry**: Allows user to enter 24 different float values.

---

## Installation & Usage Guide

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)

Follow these steps to install and run this project.

---

### 1. Install Python

Ensure that **Python 3.10+** is installed on your system.  
You can verify your version using:

```bash
python --version
```

If Python is not installed, download it from the [official Python website](https://www.python.org/downloads/).

---
### 2. Creating an virtual environment
from the root of the repo e.g. `some-path/UoP-AI-CW/`.
create virtual environment.
```bash
python3 -m venv .venv
```
activate virtual environment.
```bash
source .venv/bin/activate
```
---

### 3. Install Required Libraries

Use the following command to install all dependencies:

```bash
pip install -r requirements.txt
```
---

### 4. Starting the CLI
from the root of the repo e.g. `some-path/UoP-AI-CW/` run:
```bash
python main.py
```
to start the CLI application and access genetic algorithm and neural network features.

### Workflow Examples

#### Example 1: Running Genetic Algorithm with Defaults

1. Start the application
2. Select `0` for Genetic Algorithm
3. When prompted about defaults, enter `N` to use default parameters
4. The GA will execute and generate an optimised gait pattern

**Output**: Creates `results.txt` containing a 1000×24 matrix of joint angles, the large size is for the neural network to ensure there is enough data.

#### Example 2: Training a Neural Network with Custom Parameters

1. Start the application
2. Select `1` for Neural Network
3. Select `0` for a Neural Network without PyTorch or `1` with PyTorch
4. Select `0` for Train
5. When prompted, enter `y` to modify defaults
6. Select parameters to modify (e.g., `0` for hidden layers)
7. Enter new values (e.g., `[256, 128, 64]`)
8. Select `save and exit` to confirm changes
9. Training begins with a custom configuration

**Output**: Creates `nn.pickle` containing the trained model without PyTorch or `nn_pytorch.pth` otherwise

#### Example 3: Making Predictions with Trained Neural Network

1. Start the application
2. Select `1` for Neural Network
3. Select `0` for a Neural Network without PyTorch or `1` with PyTorch
4. Select `1` for Predict
5. Choose input method:
   - **Random**: Select `0` - generates 24 random joint angles
   - **Manual**: Select `1` - enter 24 values sequentially
6. Optionally modify prediction parameters (gait length, output file)
7. Neural network generates sequential predictions

**Output**: Creates `nn_predict_results.txt` containing predicted gait sequence using a Non PyTorch trained network or `pytorch_predict_results.txt` otherwise

---

## Component Integration

### Genetic Algorithm Integration

The CLI interfaces with the GA module through `ga.main`:

```python
import ga.main as ga

# Get GA configuration parameters
defaults = get_defaults(ga.defaults)

# Execute genetic algorithm with configuration
ga.main(**defaults)
```

#### GA Default Parameters
```python
{
    "gait_length":gait_length,
    "population_size": population_size,
    "mutation_rate": mutation_rate,
    "crossover_rate": crossover_rate,
    "output_file": output_file,
    "fitness_score_target": fitness_score_target
}
```

**For detailed GA implementation**, see [GA_documentation.md](./ga/GA_documentation.md)

---

### Neural Network Integration

The CLI provides two operational modes for the neural network:

#### Training Mode

```python
import nn.main as nn

import pytorch_nn.main as pytorch_nn

# Get NN training configuration
defaults = get_defaults(nn.defaults)

# Set the correct modules based on choice
if nn_choice == "pytorch": # PyTorch implementation
    nn_module = pytorch_nn
else:  # from scratch implementation
    nn_module = nn

# Train neural network
nn_module.main(**defaults)
```

**Training Parameters**:
```python

{
    "hidden_layers": list[int],      # e.g., [128, 64, 32]
    "learning_rate": float,          # e.g., 0.01
    "nn_save": str,                  # e.g., "nn.pickle"
    "training_data_ratio": float,    # e.g., 0.95
    "training_batch_size": int,      # e.g., 1
    "epochs": int                    # e.g., 100

    #pytorch implementation
    "optimiser": str                 # e.g., adam
    #from scratch implementation
    "optimiser": opt                 # e.g., adam
}
```

#### Prediction Mode

```python
import nn.load_and_predict as predict

import pytorch_nn.load_and_predict as pytorch_predict

if nn_choice == "pytorch": # PyTorch implementation
    predict_module = pytorch_predict
    model_extension = "nn_pytorch.pth"
    predict_file_name = "pytorch_predict_results.txt"
else:  # from scratch implementation
    predict_module = predict
    model_extension = "nn.pickle"
    predict_file_name = "nn_predict_results.txt"
        

# Generate or collect input
input = [random.randint(-100, 100) for _ in range(24)]  # Random
# OR
input = handle_lists(float, 24)  # Manual

# Get prediction configuration
predict_defaults = {
    "nn_path": "./nn.pickle",
    "output_file_name": "./nn_predict_results.txt",
    "input": input,
    "gait_length": 100
}

defaults = get_defaults(predict_defaults)

# unpacks default values to named params of load_and_predict
predict_module.load_and_predict(**defaults)
```

**For detailed NN implementation**, see [NN_documentations.md](./nn/NN_documentations.md)

---

## Configuration Options

### Genetic Algorithm Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `population_size` | int | Number of individuals in each generation | 100 |
| `gait_compare_length` | int | Number of time steps / frames for each individual, this can be a low number as each individual is not the final time step size. A lower number here reduces computational overhead during execution. | 50 |
| `final_gait_length` | int | The number time steps / frame for the outputted result. The default value of 1000 is chosen to generate more training data for the neural network. | 1000 |
| `num_generations` | int | Maximum evolutionary iterations | 100 |
| `mutation_rate` | float | Probability of gene mutation | 0.1 |
| `crossover_rate` | float | Probability of parent crossover occuring. If parent pairs are not selected for crossover they move into the next generation. | 0.7 |
| `tournament_size` | int | Individuals selected for tournament | 5 |
| `output_file` | str | Results file path | "ga_results.txt" |

### Neural Network Training Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `hidden_layers` | list[int] | Neurons per hidden layer | [128, 64, 32] |
| `learning_rate` | float | Training step size | 0.01 |
| `nn_save` | str | Model save location | "nn.pickle" |
| `training_data_ratio` | float | Train/test split ratio | 0.95 |
| `training_batch_size` | int | Samples per training batch | 1 |
| `epochs` | int | Training iterations | 100 |
| `optimiser` | str | Optimiser algorithm | "gradient_descent" |

### Neural Network Prediction Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `nn_path` | str | Path to trained model | "./nn.pickle" |
| `output_file_name` | str | Prediction results file | "./nn_predict_results.txt" |
| `input` | list[float] | Initial 24 joint angles | (random/manual) |
| `gait_length` | int | Number of frames to predict | 1000 |

---

## File Outputs

### Genetic Algorithm Output

The genetic algorithm by default will output a gait from the best individual to a file called `ga_results_{unix time}.txt`.
Where unix time is replaced with the current time since epoch, this was chosen to prevent overwriting results.
This file will include 24 values per line, which correspond to joint angles.
Additionally the file will be 1000 lines long by default to allow for diverse training data for the neural network, this can be changed.

Finally the resulting output `ga_results_{unix time}.txt` can be imported into matlab and the gait can be viewed by adding these lines to the existing `spider.m` file.

```matlab
% In MATLAB:
%  the file path of read matrix should be the resulting ga output, this is an example

v = readmatrix('./ga/results/ga_results_1764094990.txt')

A = deg2rad(v)

for idx = 1:size(v,1)
    plot_spider_pose(A(idx,:))
    pause(0.0001)
end
```

### Neural Network Outputs

#### Training

The neural network from scratch by default will load `nn.pickle`. This file is a pickled (serialised) neural network object.
This neural network includes: learned weights and biases and specific layer sizes e.g. `[128,64,32]`

#### Prediction

The neural network will output a file called `nn_predict_results.txt` by default.
This file has a similar format to the GA output, except by default is 101 lines instead of 1000.
The file represents the result of starting from an initial pose, predicting and then feeding those predictions back into the neural network 100 times.

---

## Error Handling

The CLI includes robust error handling:

### Input Validation
- **Type Checking**: Ensures correct data types for all parameters
- **Range Validation**: Verifies values are within acceptable bounds
- **Retry Mechanism**: Allows users to correct invalid inputs without restarting

### File Operations
- **Path Validation**: Checks file accessibility before operations
- **Graceful Failures**: Provides clear error messages if files cannot be read/written
- **Default Fallbacks**: Uses sensible defaults when custom paths are invalid

### Execution Safety
- **Exception Handling**: Catches and reports module-level errors
- **Clean Exit**: Properly terminates on user request or critical failure

---

## Best Practices

### For Genetic Algorithm Runs
1. **Start with defaults** to establish baseline performance
2. **Increase population size** for better exploration (computational cost increases)
3. **Adjust mutation rate** if convergence is too fast or too slow
4. **Monitor fitness graphs** to assess convergence behavior

### For Neural Network Training
1. **Use adequate training data** (≥500 gait variations recommended)
2. **Tune learning rate** based on convergence speed (0.001-0.1 typical range)
3. **Experiment with architecture** (hidden layers) for optimal performance
4. **Validate with test set** to ensure generalisation

### For Neural Network Prediction
1. **Start with biologically plausible inputs** (-50° to 30° per joint)
2. **Use trained models** that showed good test performance
3. **Verify prediction continuity** by checking frame-to-frame transitions
4. **Compare with GA results** for consistency validation

---

## Integration Workflow

### Combined GA + NN Approach

For optimal results, use both systems in sequence:

1. **Phase 1: Genetic Algorithm**
   - Run GA to generate high-quality gait data
   - Save multiple results to allow give the NN diverse training data

2. **Phase 2: Learning**
   - Use evolved gaits as training data for neural network
   - Train NN to learn patterns from optimised gaits

3. **Phase 3: Neural Network Prediction**
   - Use a trained neural network to predict the next time steps.
   - Verify in matlab the results of the predicted gait.

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Module not found" | Missing dependencies | Run `pip install -r requirements.txt` |
| "File not found" | Incorrect path | Verify file paths are absolute or relative to CWD |
| "Training not converging" | Poor hyperparameters | Adjust learning rate or increase epochs |
| "GA fitness plateauing" | Premature convergence | Increase mutation rate or population size |

---

## Future Improvements

- **Batch Processing**: Run multiple configurations automatically
- **Visualisation**: Real-time fitness/loss plotting during execution
- **Comparison Mode**: Automatically compare GA vs NN outputs
- **Resume Training**: Save and restore training checkpoints
- **Export Formats**: Support for multiple output formats (CSV, JSON, etc.)

---

## References

- **Genetic Algorithm Implementation**: See [GA_documentation.md](./ga/GA_documentation.md)
- **Neural Network Implementation**: See [NN_documentations.md](./nn/NN_documentations.md)
