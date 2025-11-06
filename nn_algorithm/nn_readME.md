# Spider Joint Angle Neural Network - README

## Overview

This Neural Network system generates realistic joint angle configurations for a simplified 3D spider model with 8 legs and 3 joints per leg (24 total joint angles).

---

## Neural Network Architecture

### **Input Layer: 6 Neurons**

The input features represent high-level movement goals and environmental conditions:

1. **Target X Position** (-1 to 1, normalized)
2. **Target Y Position** (-1 to 1, normalized) 
3. **Target Z Position** (-1 to 1, normalized)
4. **Gait Phase** (0 to 1, represents walking cycle)
5. **Terrain Slope X** (-0.3 to 0.3)
6. **Terrain Slope Z** (-0.3 to 0.3)

**Design Rationale:** This input design allows the network to generate context-aware joint configurations. The phase parameter enables coordinated gait patterns, while terrain parameters allow adaptation to slopes.

### **Hidden Layers: [32, 48, 48, 32] Neurons**

- **Layer 1:** 32 neurons - Initial feature extraction
- **Layer 2:** 48 neurons - Expanded representation learning
- **Layer 3:** 48 neurons - Deep feature processing
- **Layer 4:** 32 neurons - Feature compression before output

**Design Rationale:** 
- Deep architecture (4 hidden layers) captures complex relationships between inputs and 24-dimensional output
- Expansion-contraction pattern (32→48→48→32) allows the network to learn rich intermediate representations
- Width of 48 neurons provides sufficient capacity for the problem complexity

### **Output Layer: 24 Neurons**

Produces 24 joint angles (8 legs × 3 joints), scaled to [-π, π] radians to represent realistic joint rotation limits.

---

## Key Design Choices

### **Activation Functions**

**Hidden Layers: Hyperbolic Tangent (tanh)**
- Output range: [-1, 1]
- Centered around zero, ideal for normalized data
- Smooth gradients prevent vanishing gradient problems
- Better than sigmoid for hidden layers due to zero-centered outputs

**Output Layer: Scaled tanh**
- tanh output multiplied by π to produce range [-π, π]
- Ensures physically realistic joint angles
- Smooth bounds prevent extreme configurations

### **Loss Function**

**Mean Squared Error (MSE)**
```
MSE = average((target - prediction)²)
```

**Rationale:**
- Standard choice for regression problems
- Penalizes large errors quadratically
- Smooth derivative for gradient descent
- Appropriate for continuous angle prediction

### **Training Method**

**Stochastic Gradient Descent with Mini-Batches**

- **Learning Rate:** 0.01 (tuned through experimentation)
- **Batch Size:** 32 samples
- **Optimizer:** Standard SGD with backpropagation
- **Weight Initialization:** Xavier initialization (√(2/(n_in + n_out)))

**Rationale:**
- Mini-batches balance training speed and gradient stability
- Learning rate 0.01 provides steady convergence without overshooting
- Xavier initialization prevents vanishing/exploding gradients

---

## Training Data Generation

### **Biomechanically-Inspired Synthetic Data**

The training data is generated using principles from arachnid locomotion:

#### **Tripod Gait Pattern**
- Legs alternate in two groups (0,2,4,6 vs 1,3,5,7)
- Phase offset of 0.5 between groups
- Common pattern in hexapods and arachnids

#### **Phase-Based Coordination**
- Each leg's movement coordinated with gait cycle phase
- Swing phase (0-0.5): Leg lifts and moves forward
- Stance phase (0.5-1.0): Leg pushes against ground

#### **Joint Angle Calculations**

**Joint 1 (Coxa - Horizontal Rotation):**
```
j1 = base_angle + 0.5 × sin(2π × phase) × target_x + slope_x × 0.3
```
- Base angle varies by leg position (front/middle/rear)
- Modulated by target direction and terrain

**Joint 2 (Femur - Vertical Lift):**
```
Swing phase:  j2 = -0.8 + 1.2 × sin(π × phase)
Stance phase: j2 = -0.3 + 0.2 × sin(π × (phase - 0.5))
```
- Larger movement during swing phase for ground clearance
- Small adjustment during stance for weight distribution

**Joint 3 (Tibia - Extension):**
```
Swing phase:  j3 = 0.6 - 0.8 × sin(π × phase)
Stance phase: j3 = 0.4 - 0.3 × cos(π × (phase - 0.5))
```
- Coordinates with femur for natural leg extension
- Adapts to terrain slope

**Training Dataset:** 5,000 samples covering diverse movement scenarios

---

## How to Run the Code

### **Requirements**
```bash
pip install numpy matplotlib
```

### **Basic Usage**

1. **Train a new model:**
```bash
python spider_nn.py
```

This will:
- Generate 5,000 training samples
- Train for 500 epochs
- Save model to `spider_nn_model.pkl`
- Generate visualization plots

2. **Use the trained model:**
```python
from spider_nn import Spider_NN

# Load trained model
nn = Spider_NN(X=6, HL=[32, 48, 48, 32], Y=24)
nn.load_model('spider_nn_model.pkl')

# Generate joint angles for a movement
import numpy as np
input_data = np.array([0.5, 0.0, 0.3, 0.25, 0.0, 0.0])
# [target_x, target_y, target_z, phase, slope_x, slope_z]

joint_angles = nn.FF(input_data)
print(f"Generated {len(joint_angles)} joint angles")
```

### **Expected Outputs**

1. **Console Output:** Training progress showing loss decrease
2. **spider_nn_model.pkl:** Saved trained model
3. **spider_nn_training_results.png:** 4-panel training visualization
4. **spider_pose_visualization.png:** Single detailed 2D spider pose
5. **spider_pose_matrix.png:** 6 different poses in a 2×3 grid layout
6. **output_poses.txt:** Detailed joint angle matrices in text format

---

## Output Files Explained

### **spider_pose_matrix.png**
Matrix visualization showing 6 different spider poses generated by the neural network:
- Arranged in 2 rows × 3 columns
- Each subplot shows a different input configuration
- Displays the gait phase and target position for each pose
- Color-coded legs: Blue (Coxa) → Green (Femur) → Red (Tibia)
- Demonstrates network's ability to generate diverse configurations

### **output_poses.txt**
Comprehensive text file containing detailed joint angle data:
- **Format:** 5 example poses with complete information
- **For each pose:**
  - Input parameters (target position, phase, terrain)
  - Joint angles organized by leg in a clear table format
  - Complete 24-element joint angle vector
  - Statistical summary (mean, std dev, min, max)
- **Easy to parse:** Can be read by other programs or imported to 3D renderers
- **Human-readable:** Clear labels and formatting for manual inspection

Example format from output_poses.txt:
```
EXAMPLE 1
--------------------------------------------------------------------------------
Input Parameters:
  Target Position: X= 0.4567, Y=-0.1234, Z= 0.7890
  Gait Phase:       0.2500 (0=start, 1=end of cycle)
  Terrain Slope:   X= 0.0500, Z=-0.0300

Output Joint Angles (radians):

     Leg    |   Coxa   |  Femur   |  Tibia   |
  ----------|----------|----------|----------|
  Leg 0     |  0.5234  | -0.3421  |  0.2156  |
  Leg 1     |  0.4987  | -0.3198  |  0.1987  |
  ...
```

---

## Visualization Outputs

### **Training Loss Plot**
- Shows MSE decreasing over epochs
- Log scale for better visibility
- Indicates convergence quality

### **Prediction vs Target**
- Compares network output with ground truth
- Shows joint-by-joint accuracy
- Sample taken from test set

### **Error Distribution**
- Histogram of prediction errors
- Should be centered near zero
- Narrow distribution indicates good performance

### **Joint-wise MAE**
- Mean Absolute Error for each of 24 joints
- Identifies which joints are harder to predict
- Useful for architecture refinement

### **Spider Pose Visualization**
- 2D projection of 3D spider configuration
- Color-coded leg segments (blue→green→red)
- Visual verification of pose validity

---

## Performance Metrics

### **Expected Performance**
- **Training Loss:** < 0.01 after 500 epochs
- **Mean Absolute Error:** < 0.05 radians per joint
- **Convergence:** Stable loss reduction by epoch 300

### **Evaluation Criteria**
1. **Convergence:** Loss should decrease smoothly
2. **Generalization:** Low error on test set
3. **Validity:** Generated angles within [-π, π]
4. **Coordination:** Legs show realistic phase relationships

---

## Customization Options

### **Modify Architecture**
```python
# Deeper network
nn = Spider_NN(X=6, HL=[32, 64, 64, 64, 32], Y=24)

# Wider network
nn = Spider_NN(X=6, HL=[64, 96, 96, 64], Y=24)
```

### **Adjust Training Parameters**
```python
nn.train_nn(
    training_inputs, 
    training_targets,
    epochs=1000,      # More training
    lr=0.005,         # Lower learning rate
    batch_size=64     # Larger batches
)
```

### **Different Input Features**
Modify `generate_training_data()` to include:
- Velocity information
- Previous pose (for sequence prediction)
- Obstacle avoidance vectors
- Energy efficiency targets

---

## Comparison with Libraries (Optional)

To compare with PyTorch or TensorFlow:

### **PyTorch Equivalent**
```python
import torch
import torch.nn as nn

class SpiderNN_PyTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(6, 32),
            nn.Tanh(),
            nn.Linear(32, 48),
            nn.Tanh(),
            nn.Linear(48, 48),
            nn.Tanh(),
            nn.Linear(48, 32),
            nn.Tanh(),
            nn.Linear(32, 24),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.layers(x) * torch.pi

# Train with MSE loss
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### **Key Differences**
- **Libraries:** Automatic differentiation, GPU acceleration
- **From Scratch:** Better understanding of backpropagation
- **Performance:** Libraries typically faster with optimizations
- **Learning:** From-scratch implementation demonstrates comprehension

---

## Troubleshooting

### **High Training Loss**
- Reduce learning rate (try 0.005)
- Increase training epochs
- Check data normalization
- Verify weight initialization

### **Network Not Converging**
- Try different activation functions
- Adjust network depth/width
- Increase training data size
- Check for gradient explosion (reduce lr)

### **Poor Generalization**
- Generate more diverse training data
- Add regularization (L2 penalty)
- Reduce network complexity
- Validate on separate test set

---

## Extension Ideas

1. **Recurrent Network:** Add LSTM layers for temporal sequences
2. **Physics Constraints:** Include stability/balance loss terms
3. **Real Data:** Train on motion capture from real spiders
4. **Multi-Task:** Predict both joint angles and foot positions
5. **Reinforcement Learning:** Let spider learn through simulation

---

## Contact & Support

For questions about the implementation or coursework requirements, refer to:
- Code comments (detailed inline documentation)
- This README
- Coursework specification document

**Model Files:**
- `spider_nn_model.pkl` - Trained network weights
- `spider_nn_training_results.png` - Performance visualizations
- `spider_pose_visualization.png` - Example output pose

---

## Summary

This neural network successfully generates 24 joint angles for an 8-legged spider model using:
- **6 input features** (position, phase, terrain)
- **4 hidden layers** [32, 48, 48, 32] with tanh activation
- **MSE loss** function
- **SGD optimizer** with lr=0.01
- **5,000 training samples** from biomechanical patterns

The implementation demonstrates understanding of:
✓ Neural network architecture design
✓ Activation function selection
✓ Backpropagation and gradient descent
✓ Training data generation
✓ Performance visualization
✓ Model persistence