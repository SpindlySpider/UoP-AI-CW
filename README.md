# Spider Gait Generation System - Main CLI Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [CLI Features](#cli-features)
4. [Usage Guide](#usage-guide)
5. [Component Integration](#component-integration)
6. [Neural Network Implementations](#neural-network-implementations)
7. [Configuration Options](#configuration-options)
8. [File Outputs](#file-outputs)

---

## Overview

The **main.py** file serves as the central command-line interface (CLI) for the Spider Gait Generation System, providing a unified access point to multiple AI approaches for generating and predicting spider locomotion patterns:

1. **Genetic Algorithm (GA)** - Evolves optimal gait patterns through evolutionary optimization
2. **Neural Network (NN)** - Learns to predict sequential gait frames from training data
   - **From Scratch Implementation** - Pure Python implementation for educational purposes
   - **PyTorch Implementation** - GPU-accelerated deep learning framework for production use

This modular CLI design allows users to seamlessly switch between evolutionary optimization and machine learning approaches, or use them in combination for comprehensive gait generation and analysis.

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
    ├── modify_default()    # Parameter customization
    └── handle_lists()      # Input validation
```

### Key Components

| Component | Purpose | Reference |
|-----------|---------|-----------|
| **Genetic Algorithm** | Evolves complete gait patterns using sine-wave chromosome encoding | [GA_documentation.md](./ga/GA_documentation.md) |
| **Neural Network** | Predicts sequential gait frames through supervised learning | [NN_documentation.md](./nn_without_pytorch/NN_documentations.md) |
| **CLI Utilities** | Provides interactive menu navigation and parameter configuration | `utils.py` |

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
- **Numbered Selection**: Users select options by entering corresponding numbers (0, 1, 2, etc.)
- **Input Validation**: Automatically validates user input and provides helpful error messages
- **Clear Feedback**: Each selection is confirmed with visual separators for clarity

### 2. Configuration Management

Both GA and NN modules support **customizable parameters** with an intuitive modification interface:

#### Features:
- **Display Defaults**: Shows all current parameter values before execution
- **Interactive Modification**: Allows selective parameter changes through guided prompts
- **Type Safety**: Enforces correct data types for each parameter
- **Save/Discard Options**: Users can save changes or revert to defaults

#### Example Workflow:
```
starting with these defaults:
  0) hidden layers: [128, 64, 32]
  1) learning rate: 0.01
  2) nn save: nn.pickle
  ...
====================
would you like to change anything? (y/N): y
```

### 3. Input Handling

The CLI provides flexible input methods for different use cases:

#### List Input
- **Fixed-Length Lists**: For neural network predictions (24 joint angles)
- **Variable-Length Lists**: For hidden layer configuration
- **Type Validation**: Ensures correct data types (float/int)

#### Input Modes
- **Random Generation**: Automatically generates valid random inputs
- **Manual Entry**: Guided step-by-step input for precise control

---

## Usage Guide

### Running the Application

```bash
python main.py
```

### Workflow Examples

#### Example 1: Running Genetic Algorithm with Defaults

1. Start the application
2. Select `0` for Genetic Algorithm
3. When prompted about defaults, enter `N` to use default parameters
4. The GA will execute and generate an optimized gait pattern

**Output**: Creates `results.txt` containing a 300×24 matrix of joint angles

#### Example 2: Training a Neural Network with Custom Parameters

1. Start the application
2. Select `1` for Neural Network
3. Select `0` for Train
4. When prompted, enter `y` to modify defaults
5. Select parameters to modify (e.g., `0` for hidden layers)
6. Enter new values (e.g., `[256, 128, 64]`)
7. Select `save and exit` to confirm changes
8. Training begins with custom configuration

**Output**: Creates `nn.pickle` containing the trained model

#### Example 3: Making Predictions with Trained Neural Network

1. Start the application
2. Select `1` for Neural Network
3. Select `1` for Predict
4. Choose input method:
   - **Random**: Select `0` - generates 24 random joint angles
   - **Manual**: Select `1` - enter 24 values sequentially
5. Optionally modify prediction parameters (gait length, output file)
6. Neural network generates sequential predictions

**Output**: Creates `predict_results.txt` containing predicted gait sequence

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
    "population_size": int,
    "gait_length": int,
    "num_generations": int,
    "mutation_rate": float,
    "crossover_rate": float,
    "tournament_size": int,
    "output_file": str
}
```

**For detailed GA implementation**, see [GA_documentation.md](./ga/GA_documentation.md)

---

### Neural Network Integration

The CLI provides two operational modes for the neural network:

#### Training Mode

```python
import nn.main as nn

# Get NN training configuration
defaults = get_defaults(nn.defaults)

# Train neural network
nn.main(**defaults)
```

**Training Parameters**:
```python
{
    "hidden_layers": list[int],      # e.g., [128, 64, 32]
    "learning_rate": float,          # e.g., 0.01
    "nn_save": str,                  # e.g., "nn.pickle"
    "training_data_ratio": float,    # e.g., 0.95
    "data_gait_length": int,         # e.g., 40
    "gait_variations": int,          # e.g., 700
    "training_batch_size": int,      # e.g., 1
    "epochs": int                    # e.g., 100
    "optimiser": str                 # e.g., adam
}
```

#### Prediction Mode

```python
import nn.load_and_predict as predict

# Generate or collect input
input = [random.randint(-100, 100) for _ in range(24)]  # Random
# OR
input = handle_lists(float, 24)  # Manual

# Get prediction configuration
predict_defaults = {
    "nn_path": "./nn.pickle",
    "output_file_name": "./predict_results.txt",
    "input": input,
    "gait_length": 100
}
defaults = get_defaults(predict_defaults)

# Generate predictions
predict.load_and_predict(**defaults)
```

**For detailed NN implementation**, see [NN_documentations.md](./nn_without_pytorch/NN_documentations.md)

---

## Neural Network Implementations

The system provides **two neural network implementations** with identical interfaces but different underlying technologies:

### From Scratch Implementation (`nn_without_pytorch/`)

**Purpose**: Educational reference and lightweight deployment

**Key Features**:
- Pure Python implementation using NumPy
- From scratch backpropagation and gradient descent
- No external deep learning frameworks required
- Ideal for understanding neural network mechanics

**Model Format**: `.pickle` (serialised Python objects)

**Optimizers**: `gradient_descent`, `adam`

### PyTorch Implementation (`pytorch_nn/`)

**Purpose**: Production use with GPU acceleration

**Key Features**:
- Built on PyTorch deep learning framework
- GPU acceleration via CUDA (when available)
- Automatic differentiation for backpropagation
- Optimised matrix operations
- Industry-standard architecture

**Model Format**: `.pth` (PyTorch state dict)

**Optimizers**: `sgd`, `adam`

### Implementation Comparison

| Feature | From Scratch | PyTorch |
|---------|--------------|---------|
| **Speed** | Moderate (CPU only) | Fast (GPU/CPU) |
| **Dependencies** | NumPy only | PyTorch framework |
| **Learning** | Great for education | Production ready |
| **GPU Support** | ❌ No | ✅ Yes |
| **Model Size** | Larger (full objects) | Smaller (weights only) |
| **Compatibility** | Python specific | Cross-platform |

### Identical API Interface

Both implementations share the **exact same interface**:

```python
# Training (both implementations)
main(
    hidden_layers=[128, 64, 32],
    learning_rate=0.01,
    epochs=100,
    ...
)

# Prediction (both implementations)
load_and_predict(
    input=[...],           # 24 joint angles
    nn_path="model_file",  # .pickle or .pth
    output_file_name="results.txt",
    gait_length=300
)
```

### Choosing an Implementation

**Use From Scratch Implementation when**:
- Learning about neural network internals
- No GPU available and dataset is small
- Avoiding heavy framework dependencies
- Teaching or demonstrating concepts

**Use PyTorch when**:
- Training large models with extensive data
- GPU acceleration is available
- Production deployment requirements
- Leveraging modern deep learning features

**Both produce identical results** with the same hyperparameters, ensuring consistency across implementations.

---

## Configuration Options

### Genetic Algorithm Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `population_size` | int | Number of individuals in each generation | 100 |
| `gait_length` | int | Number of time steps in gait cycle | 300 |
| `num_generations` | int | Maximum evolutionary iterations | 1000 |
| `mutation_rate` | float | Probability of gene mutation | 0.1 |
| `crossover_rate` | float | Probability of parent crossover | 0.7 |
| `tournament_size` | int | Individuals selected for tournament | 5 |
| `output_file` | str | Results file path | "results.txt" |

### Neural Network Training Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `hidden_layers` | list[int] | Neurons per hidden layer | [128, 64, 32] |
| `learning_rate` | float | Training step size | 0.01 |
| `nn_save` | str | Model save location | "nn.pickle" |
| `training_data_ratio` | float | Train/test split ratio | 0.95 |
| `data_gait_length` | int | Length of training sequences | 300 |
| `gait_variations` | int | Number of training examples | 700 |
| `training_batch_size` | int | Samples per training batch | 1 |
| `epochs` | int | Training iterations | 100 |
| `optimiser` | str | Optimiser algorithm | "gradient_descent" |

### Neural Network Prediction Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `nn_path` | str | Path to trained model | "./nn.pickle" |
| `output_file_name` | str | Prediction results file | "./predict_results.txt" |
| `input` | list[float] | Initial 24 joint angles | (random/manual) |
| `gait_length` | int | Number of frames to predict | 300 |

---

## File Outputs

### Genetic Algorithm Output

**File**: `results.txt` (default)

**Format**: 300×24 matrix (rows = time steps, columns = joint angles)

**Description**: Complete evolved gait pattern with all 24 joint angles across the full gait cycle

**MATLAB Compatible**: Can be directly imported into MATLAB for visualisation and analysis

```matlab
% In MATLAB:
gait_data = load('results.txt');
plot(gait_data(:,1));  % Plot first joint's motion
```

### Neural Network Training Output

**File**: `nn.pickle` (default)

**Format**: Serialized Neural Network object

**Description**: Trained model containing:
- Network architecture (layer sizes)
- Learned weights and biases
- Activation functions
- Training configuration

### Neural Network Prediction Output

**File**: `predict_results.txt` (default)

**Format**: (gait_length+1)×24 matrix

**Description**: Sequential gait predictions starting from input frame

**Usage**: Can be analysed to evaluate prediction quality and gait continuity

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
4. **Validate with test set** to ensure generalization

### For Neural Network Prediction
1. **Start with biologically plausible inputs** (-50° to 30° per joint)
2. **Use trained models** that showed good test performance
3. **Verify prediction continuity** by checking frame-to-frame transitions
4. **Compare with GA results** for consistency validation

---

## Integration Workflow

### Combined GA + NN Approach

For optimal results, use both systems in sequence:

1. **Phase 1: Evolution**
   - Run GA to generate high-quality gait data
   - Save multiple evolved solutions with different parameters

2. **Phase 2: Learning**
   - Use evolved gaits as training data for neural network
   - Train NN to learn patterns from optimized gaits

3. **Phase 3: Prediction**
   - Use trained NN for real-time gait generation
   - Generate variations by starting from different initial poses

This hybrid approach combines the **optimization strength of GA** with the **prediction speed of NN**.

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Module not found" | Missing dependencies | Run `pip install -r requirements.txt` |
| "Invalid input type" | Wrong data format | Check parameter types in configuration |
| "File not found" | Incorrect path | Verify file paths are absolute or relative to CWD |
| "Training not converging" | Poor hyperparameters | Adjust learning rate or increase epochs |
| "GA fitness plateauing" | Premature convergence | Increase mutation rate or population size |

---

## Future Enhancements

Potential CLI improvements:

- **Batch Processing**: Run multiple configurations automatically
- **Visualization**: Real-time fitness/loss plotting during execution
- **Comparison Mode**: Automatically compare GA vs NN outputs
- **Resume Training**: Save and restore training checkpoints
- **Export Formats**: Support for multiple output formats (CSV, JSON, etc.)

---

## References

- **Genetic Algorithm Implementation**: See [GA_documentation.md](./ga/GA_documentation.md)
- **Neural Network Implementation**: See [NN_documentations.md](./nn_without_pytorch/NN_documentations.md)
- **Utility Functions**: See `utils.py` for CLI helper functions

---

## Summary

The `main.py` CLI provides a **unified interface** for spider gait generation using two complementary AI approaches:

- **Genetic Algorithm**: Evolutionary optimization for discovering optimal gait patterns
- **Neural Network**: Supervised learning for rapid gait prediction

With its **interactive menu system**, **flexible configuration**, and **robust error handling**, the CLI enables both researchers and practitioners to efficiently generate, train, and predict spider gait patterns.
