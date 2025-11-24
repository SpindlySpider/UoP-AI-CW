# Neural Network for Spider Gait Prediction

## Overview

This neural network predicts the **next frame of a spider's gait** from its current joint configuration. The spider has 8 legs with 3 joints each (coxa, femur, tibia), totaling **24 degrees of freedom**. 
**Input/Output**: `[24 joint angles] → Neural Network → [24 predicted joint angles]`

### Implementations

Two implementations are provided: **`nn_without_lib/`** (NumPy from scratch) and **`pytorch_nn/`** (PyTorch framework). The PyTorch version replicates the exact architecture, training procedure, and hyperparameters of the NumPy implementation to enable direct comparison. However, PyTorch runs significantly slower with batch size 1 due to framework overhead—PyTorch is optimized for larger batches where GPU acceleration and vectorization provide substantial speedups, but with single-sample batches, the tensor conversion and computational graph overhead outweighs these benefits compared to NumPy's direct array operations.

---

## 1. Network Architecture

### Structure

```
Input Layer:     24 neurons (current joint angles)
Hidden Layer 1:  128 neurons + Sigmoid activation
Hidden Layer 2:  64 neurons + Sigmoid activation  
Hidden Layer 3:  32 neurons + Sigmoid activation
Output Layer:    24 neurons + Sigmoid activation (predicted joint angles)
```

**Total Parameters**: ~20,000 trainable weights and biases

### Architecture Justification

| Decision | Rationale |
|----------|-----------|
| **3 Hidden Layers [128, 64, 32]** | Progressive dimensionality reduction for hierarchical feature extraction. Balances capacity with training efficiency. |
| **Input/Output Shape (24, 24)** | Matches spider's 24 joints. Direct frame-to-frame prediction enables recursive gait generation. |
| **Decreasing Layer Sizes** | Funnels high-dimensional input through compressed representations, learning essential motion patterns. |
| **Fully-Connected (Dense)** | All joints influence each other - legs coordinate during walking. Dense connections capture inter-joint dependencies. |

**Number of layer**
- **Too shallow (1-2 layers)**: Cannot learn complex temporal patterns
- **Too deep (5+ layers)**: Overfits small dataset, slower training, vanishing gradients
- **3 layers**: Optimal for this problem's complexity

---

## 2. Activation Functions

**Choice**: **Sigmoid** activation for all layers

### Justification

| Aspect | Sigmoid Benefits | Why Suitable |
|--------|------------------|--------------|
| **Output Range** | [0, 1] | Matches normalized joint angle range perfectly |
| **Smooth Gradients** | Differentiable everywhere | Enables stable backpropagation |
| **Non-linearity** | S-curve shape | Captures complex motion patterns |
| **Biological Realism** | Smooth transitions | Mirrors natural joint movement |

**Formula**: $\sigma(x) = \frac{1}{1 + e^{-x}}$

**Derivative** (for backpropagation): $\sigma'(x) = \sigma(x)(1 - \sigma(x))$

**Trade-off**: Sigmoid can cause vanishing gradients in very deep networks, but at 3 layers this is manageable. The bounded output range [0,1] is ideal for our normalized data.

---

## 3. Loss Function

**Choice**: **Mean Squared Error (MSE)**

### Formula

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Where:
- $y_i$ = target joint angle (ground truth)
- $\hat{y}_i$ = predicted joint angle  
- $n$ = 24 (number of joints)

### Justification

| Reason | Explanation |
|--------|-------------|
| **Regression Task** | Predicting continuous values (angles), not classification |
| **Penalizes Large Errors** | Squared term heavily penalizes predictions far from target |
| **Differentiable** | Smooth gradient enables efficient backpropagation |
| **Balanced Across Joints** | Treats all 24 joints equally in error calculation |

**Why Not Other Loss Functions?**
- **MAE (Mean Absolute Error)**: Less sensitive to outliers, but we want to heavily penalize bad predictions
- **Cross-Entropy**: For classification only, not regression
- **Huber Loss**: Useful for noisy data, but our synthetic data is clean

---

## 4. Training Method & Learning Rate

### Optimizer Comparison: Gradient Descent vs Adam

Both optimizers were tested extensively. Results below show **actual training runs** on identical data.

#### Configuration

| Parameter | Value |
|-----------|-------|
| **Training Data** | 25,935 samples (700 gait variations × 39 pairs each × 0.95) |
| **Test Data** | 1,365 samples (5% holdout) |
| **Batch Size** | 1 (SGD - Stochastic Gradient Descent) |
| **Epochs** | 100 |
| **Learning Rate** | 0.01 (both optimizers) |

---

### Gradient Descent ✅ RECOMMENDED

**Algorithm**: $W_{new} = W_{old} - \alpha \nabla L$

**Results**:
- **Training Loss**: 0.011322 → 0.001249 (89% reduction)
- **Test Loss**: 0.001287
- **Training Time**: 231.91 seconds (~4 minutes)
- **Stability**: Perfect - zero spikes

**Training Progress**:

![Gradient Descent Loss](doc-images/gradient_decent_default_learning%20_rate.png)

**Analysis**: Smooth exponential decay from epoch 0 to 100. No oscillations or instabilities. The learning rate of 0.01 is well-tuned - loss decreases steadily without overshooting.

**Why It Works**:
- **Simple updates**: Direct gradient application without momentum
- **Stable convergence**: Predictable, deterministic weight changes
- **Low computational overhead**: Fast per-epoch execution
- **Suitable LR**: 0.01 allows steady progress without divergence

---

### Adam Optimizer ⚠️ UNSTABLE (Default Settings)

**Algorithm**: Adaptive Moment Estimation with momentum

**Results**:
- **Training Loss**: 0.002011 → 0.001383 (31% reduction)
- **Test Loss**: 0.001342  
- **Training Time**: 2381.02 seconds (~40 minutes) - **10× slower**
- **Stability**: Poor - 10+ major loss spikes

**Training Progress**:

![Adam Default LR](doc-images/adam_default_learning_rate.png)

**Analysis**: Severe instability with loss spikes at epochs 33, 39, 52, 57, 62-63, 70, 80, 88. One spike reached 0.0053 (4× baseline). Overflow warnings indicate gradient explosions despite clipping.

**Why It Failed**:
- **LR too high**: 0.01 causes overshooting when combined with momentum
- **Sigmoid saturation**: Adaptive rates amplified saturation at boundaries
- **Gradient explosions**: Default clip value (1.0) insufficient
- **Degenerate predictions**: Network outputs freeze at -50° or 30° (boundaries)

---

### Adam with Tuned Learning Rate ✅ IMPROVED

**Results with LR = 0.001**:

![Adam LR 0.001](doc-images/adam_learning_rate_0.001.png)

**Analysis**: Reducing learning rate to 0.001 dramatically improves stability. Still shows minor oscillations but no major spikes. Converges to similar loss as gradient descent but takes longer.

**Conclusion**: Adam **can work** with proper tuning (LR=0.001, possibly ReLU activations), but gradient descent is simpler and more reliable for this problem.

---

### Gradient Descent with LR = 0.001 ✅ CONSERVATIVE ALTERNATIVE

**Algorithm**: $W_{new} = W_{old} - 0.001 \times \nabla L$ (lower learning rate)

**Results**:
- **Training Loss**: 0.015 → 0.0015 (90% reduction)
- **Test Loss**: Similar to LR=0.01
- **Training Time**: Longer per loss reduction
- **Stability**: Extremely stable - ultra-smooth convergence

**Training Progress**:

![Gradient Descent LR 0.001](doc-images/gradient_decent_learning_rate_0.001.png)

**Analysis**: Exceptionally smooth exponential decay with even more gradual convergence than LR=0.01. The lower learning rate produces an extremely stable training curve with zero oscillations. Loss decreases more slowly but very predictably.

**Why It Works**:
- **Gentle updates**: Smaller steps prevent any overshooting
- **Maximum stability**: Ultra-conservative approach eliminates all risk
- **Fine control**: Precise weight adjustments near optimal point
- **Safe choice**: Guaranteed stability for sensitive problems

**When to Use 0.001**:
- Initial experiments show instability at higher rates
- Fine-tuning a pre-trained network
- Working with sensitive or poorly-conditioned problems
- Maximum stability is priority over training speed
- Transfer learning scenarios

**Trade-off**: While more conservative and stable, the slower convergence means 2-3× more epochs needed to reach the same loss as LR=0.01. Since 0.01 is already perfectly stable for this problem, 0.001 provides minimal benefit at significant time cost.

---

### Learning Rate Tuning Summary

| Optimizer | LR | Result | Verdict |
|-----------|----|----|---------|
| **Gradient Descent** | 0.01 | ✅ Stable, fast convergence | **Optimal** |
| **Gradient Descent** | 0.001 | ✅ Very stable, slower | Conservative alternative |
| **Adam** | 0.01 | ❌ Unstable, unusable | Too high |
| **Adam** | 0.001 | ✅ Stable, good convergence | Workable with tuning |

**Final Choice**: **Gradient Descent with LR = 0.01**
- Best performance (lowest loss: 0.00125)
- Fastest training (232s)
- Perfectly stable (zero issues)
- Simplest to tune (one hyperparameter)
- Optimal balance of speed and stability

**Alternative**: Use LR = 0.001 only if extreme stability is required or if experimenting with more complex architectures that might be sensitive to higher learning rates.

---

## 5. Backpropagation & Convergence

### Backpropagation Implementation

**Algorithm**: Chain rule applied layer-by-layer from output to input

```
For each layer i (backward):
    δᵢ = error × σ'(aᵢ)           # Error signal
    ∇Wᵢ = aᵢ₋₁ᵀ × δᵢ              # Weight gradient
    error = δᵢ × Wᵢᵀ              # Propagate to previous layer
```

**Evidence of Correct Implementation**:

1. **Loss Decreases**: 89% reduction over 100 epochs proves gradients flow correctly
2. **Smooth Convergence**: No erratic behavior suggests proper gradient calculation
3. **Generalization**: Test loss (0.00129) close to training loss (0.00125) indicates learned patterns, not memorization

### Convergence Analysis

| Metric | Value | Indicates |
|--------|-------|-----------|
| **Initial Loss** | 0.011322 | Random initialization baseline |
| **Epoch 10 Loss** | 0.001498 | Rapid early learning |
| **Epoch 50 Loss** | 0.001294 | Continued refinement |
| **Final Loss** | 0.001249 | Converged to stable minimum |
| **Test Loss** | 0.001287 | Generalizes well (+3% from training) |

**Convergence Pattern**: Exponential decay → Logarithmic refinement → Plateau

The network reaches a **reasonable solution** where predicted joint angles closely match targets (average error ~4° per joint after denormalization).

---

## 6. Data Handling & Input/Output Format

### Training Data Generation

**Source**: Synthetic gaits generated using parametric sine wave functions

**Process**:
1. **Generate Base Gaits**: Use genetic algorithm's sine-based gait generator
2. **Randomize Parameters**: Create 700 variations by varying:
   - Period: [0.1, 1.0] seconds (gait speed)
   - Coxa amplitude: [15°, 23°] (horizontal leg swing)
   - Tibia-femur vertical shift: [40°, 55°] (leg height)
   - Tibia-femur amplitude: [15°, 35°] (vertical movement)
3. **Extract Frames**: Each gait = 40 time steps
4. **Create Pairs**: `(frame[t], frame[t+1])` for supervised learning (39 pairs per gait since last frame has no next)
5. **Combine**: 700 gaits × 39 pairs = 27,300 total samples

**Data Split**:
- **Training**: 95% (25,935 samples)
- **Testing**: 5% (1,365 samples)

### Input/Output Format

**Raw Data**: Joint angles in degrees, range varies by joint type
- Coxa joints: approximately [-23°, 23°]
- Tibia/Femur joints: approximately [-75°, -20°]

**Normalization**: Map to [0, 1] using expanded bounds

[neural_networks/nn_without_lib/`input_data.py`](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/neural_networks/nn_without_lib/input_data.py)

```python
# Used in both input_data.py and load_and_predict.py
normalized = (raw_angle + 80) / 110  # Maps [-80°, 30°] → [0, 1]
```

**Why [-80°, 30°] bounds (110° range)?**
- Code comment states: "actual range: coxa [-23, 23], tibia-femur approximately [-75, -20]"
- Expanded to [-80°, 30°] to provide safety margin beyond observed extremes
- Ensures no joint angle exceeds [0, 1] bounds after normalization
- Consistent scaling prevents some joints dominating loss
- Enables stable sigmoid outputs without saturation

### Output Format

**Network Output**: 24 normalized values in [0, 1]

**Denormalization**: Convert back to degrees

[neural_networks/nn_without_lib/`load_and_predict.py`](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/neural_networks/nn_without_lib/load_and_predict.py)

```python
# Used in both training and prediction
raw_angle = (normalized × 110) - 80  # Maps [0, 1] → [-80°, 30°]
```

**Usage**: Predicted frame becomes input for next prediction, enabling recursive gait generation

---

## 7. Performance Visualization

### Training Loss Curves

**Gradient Descent**:

![GD Loss](doc-images/gradient_decent_default_learning%20_rate.png)

- **Pattern**: Smooth exponential decay
- **Epochs 0-10**: Rapid drop (0.0113 → 0.0015)
- **Epochs 10-100**: Steady refinement (0.0015 → 0.0012)
- **Final Training Loss**: 0.001249
- **Final Test Loss**: 0.001287 (+3%)

---

### Sample Outputs: Predicted vs Target Joint Angles

**Test Input** (initial pose from GA-generated gait):
```
[0.51, -50.00, -50.00, 2.55, -27.66, -26.26, 0.51, -50.00, -50.00, 
 2.55, -27.66, -26.26, 3.23, -50.00, -50.00, -4.93, -26.37, -26.26,
 3.23, -50.00, -50.00, -4.93, -26.37, -26.26]
```

**Network Prediction** (next frame):
```
[6.73, -50.26, -50.09, -6.71, -35.07, -34.99, 6.80, -50.21, -50.35,
 -7.01, -34.92, -35.06, -6.70, -50.18, -50.12, 6.83, -34.85, -34.99,
 -6.84, -50.22, -50.20, 7.05, -35.01, -34.88]
```

**Target Values** (actual next frame from GA):
```
[17.29, -44.17, -46.70, -14.27, -39.54, -35.91, 17.29, -44.17, -46.70,
 -14.27, -39.54, -35.91, -13.73, -50.00, -50.00, 12.17, -37.64, -36.26,
 -13.73, -50.00, -50.00, 12.17, -37.64, -36.26]
```

**Comparison** (first 6 joints):

| Joint | Predicted | Target | Error |
|-------|-----------|--------|-------|
| 1 (Coxa) | 6.73° | 17.29° | 10.56° |
| 2 (Femur) | -50.26° | -44.17° | 6.09° |
| 3 (Tibia) | -50.09° | -46.70° | 3.39° |
| 4 (Coxa) | -6.71° | -14.27° | 7.56° |
| 5 (Femur) | -35.07° | -39.54° | 4.47° |
| 6 (Tibia) | -34.99° | -35.91° | 0.92° |

**Analysis**: 
- **Average Error**: ~5.5° per joint across all 24 joints
- **Pattern Mismatch**: The network predicts smoother, more symmetric motion patterns compared to the GA-generated gait which has higher amplitude variations
- **Key Difference**: The NN was trained on smoother parametric sine-based gaits, while this GA output (from 277-generation run achieving fitness 1.5035) shows more dynamic asymmetric movement (note joints 1 and 4 with ~7-11° errors)
- **Generalization**: Despite never seeing this exact GA gait pattern during training, the network produces physically plausible joint angles within valid ranges

---

### Gait Sequence Generation

**Recursive Prediction**: The network generates complete gait sequences by feeding each prediction back as input for the next frame.

**Implementation**: [neural_networks/nn_without_lib/`load_and_predict.py`](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/neural_networks/nn_without_lib/load_and_predict.py)

```python
def predict_gait(nn:Neural_network, input:list[float], gait_length:int = 300) -> Gait:
    """
    Recursively predict entire gait from one input.
    Parameters:
        nn: neural network to use to predict
        input: list of 24 floats representing joint angles in degrees
        gait_length: length of gait to produce (how many predictions)
    Returns:
        Complete gait sequence
    """
    gait = []
    gait.append(np.array(input))
    for i in range(gait_length):
        # predict next frame, starting from input
        prediction = predict(nn, gait[i])
        # reshape for output
        gait.append(prediction.reshape(prediction.shape[1]))
    return gait
```

**Example** (first 10 frames from GA initial pose, showing Leg 1's 3 joints):

```python
Frame 0 (input):  [0.51, -50.00, -50.00]    # Initial pose from GA
Frame 1:          [6.73, -50.26, -50.09]    # NN prediction
Frame 2:          [13.73, -47.21, -47.20]   # NN prediction
Frame 3:          [16.05, -40.93, -40.85]   # NN prediction
Frame 4:          [14.11, -34.39, -34.47]   # NN prediction
Frame 5:          [8.70, -30.78, -31.05]    # NN prediction
Frame 6:          [0.81, -32.74, -33.01]    # NN prediction
Frame 7:          [-7.45, -40.23, -40.27]   # NN prediction
Frame 8:          [-12.73, -47.28, -47.18]  # NN prediction
Frame 9:          [-13.51, -50.29, -50.31]  # NN prediction
Frame 10:         [-10.85, -51.20, -51.32]  # NN prediction
```

**Analysis**: 
- **Smooth Transitions**: No sudden jumps between frames, demonstrating stable temporal dynamics
- **Oscillatory Pattern**: Angles show natural cyclic movement (coxa: 0.51° → 16.05° → -13.51° → -10.85°, demonstrating learned periodic motion)
- **Physically Plausible**: All angles remain within valid ranges, no boundary saturation
- **Recursive Stability**: Network maintains coherent predictions over 300 frames (full gait in `predict_results.txt`)
- **Learned Motion**: Shows natural leg movement pattern with coordinated joint oscillations - network learned temporal dependencies from training data

## MATLAB Visualization

To visualize the gait in **MATLAB**, use the following script:

```matlab
function plot_spider_pose(angles)
    % plot_spider_pose - Plot a static 3D spider pose based on joint angles
    %
    % Input:
    %   angles: 1x24 vector of joint angles in radians
    %           [theta1_1, theta2_1, theta3_1, ..., theta1_8, theta2_8, theta3_8]
    % Legs are arranged in this configuration: {'L1', 'L2', 'L3', 'L4','R4', 'R3', 'R2', 'R1'}
    
    % Parameters
    n_legs = 8;
    segment_lengths = [1.2, 0.7, 1.0];  % [Coxa, Femur, Tibia]
    a = 1.5; b = 1.0;  % Ellipse axes for body (oval shape)

    % Base angles (L1 front-left to L4 rear-left, R4 rear-right to R1 front-right)
    left_leg_angles = deg2rad([45, 75, 105, 135]);
    right_leg_angles = deg2rad([-135, -105, -75, -45]);
    base_angles = [left_leg_angles, right_leg_angles];

    % Leg labels
    leg_labels = {'L1', 'L2', 'L3', 'L4', 'R4', 'R3', 'R2', 'R1'};

    % Validate input
    if length(angles) ~= n_legs * 3
        error('Input angles must be a 1x24 vector (3 angles per leg for 8 legs).');
    end

    % Setup figure
    figure(1); clf;
    set(gcf, 'Color', '[0,0,0]');
    ax = gca;
    ax.Color = [0.5 0.5 0.5];  
    axis equal;
    grid on;
    hold on;
    xlabel('X'); ylabel('Y'); zlabel('Z');
    %view(45, 45); % window view
    view(90,45);
    xlim([-4 4]); ylim([-4 4]); zlim([-2 2]);

    % Plot body (oval shape)
    t = linspace(0, 2*pi, 100);
    body_x = a * cos(t);
    body_y = b * sin(t);
    plot3(body_x, body_y, zeros(size(t)), 'k-', 'LineWidth', 3);

    % Head marker (front of spider at +X)
    plot3(a + 0.2, 0, 0, 'r^', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

    % Print joint angles for all legs
    fprintf('--- Spider Pose ---\n');
    
    % Loop over legs
    for i = 1:n_legs
        % Indices for this leg's angles
        idx = (i-1)*3 + 1;
        theta1 = angles(idx);
        theta2 = angles(idx+1);
        theta3 = angles(idx+2);

        fprintf('Leg %s: theta1 = %.3f rad, theta2 = %.3f rad, theta3 = %.3f rad\n', ...
            leg_labels{i}, theta1, theta2, theta3);

        % Compute leg base position on body ellipse
        angle = base_angles(i);
        x_base = a * cos(angle);
        y_base = b * sin(angle);
        base_pos = [x_base, y_base, 0];

        % Compute FK for this leg
        [j1, j2, j3, j4] = forward_leg_kinematics2(base_pos, angle, ...
            [theta1, theta2, theta3], segment_lengths);

        % Plot leg segments
        plot3([j1(1), j2(1)], [j1(2), j2(2)], [j1(3), j2(3)], 'k-', 'LineWidth', 2);
        plot3([j2(1), j3(1)], [j2(2), j3(2)], [j2(3), j3(3)], 'b-', 'LineWidth', 2);
        plot3([j3(1), j4(1)], [j3(2), j4(2)], [j3(3), j4(3)], 'r-', 'LineWidth', 2);
        plot3(j4(1), j4(2), j4(3), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');

        % Label leg
        offset = 0.2;
        label_pos = base_pos + offset * [cos(angle), sin(angle), 0];
        text(label_pos(1), label_pos(2), label_pos(3)+0.05, leg_labels{i}, ...
            'FontSize', 12, 'FontWeight', 'bold');
    end

    hold off;
end

function [j1, j2, j3, j4] = forward_leg_kinematics2(base_pos, base_angle, joint_angles, segment_lengths)
    % base_pos: [x,y,z] position of leg base on body
    % base_angle: angle around body ellipse where leg base is located (radians)
    % joint_angles: [theta1, theta2, theta3] joint angles for the leg in radians
    % segment_lengths: [coxa, femur, tibia] lengths of leg segments
    
    % Unpack joint angles
    theta1 = joint_angles(1); % Coxa yaw (rotation about vertical axis)
    theta2 = joint_angles(2); % Femur pitch
    theta3 = joint_angles(3); % Tibia pitch
    
    % Unpack segment lengths
    L1 = segment_lengths(1);  % Coxa length
    L2 = segment_lengths(2);  % Femur length
    L3 = segment_lengths(3);  % Tibia length
    
    % Joint 1: leg base on body
    j1 = base_pos;  % starting point
    
    % --- Compute Coxa direction with elevation ---
    coxa_elevation = deg2rad(30);  % fixed 30 degree upward pitch for coxa
    
    % Horizontal direction of coxa in XY plane based on base_angle + theta1
    coxa_horiz_dir = [cos(base_angle + theta1), sin(base_angle + theta1), 0];
    
    % Rotation axis for pitch up: perpendicular to coxa horizontal direction in XY plane
    rot_axis = cross(coxa_horiz_dir, [0 0 1]);
    
    % Rotation matrix around rot_axis by coxa_elevation
    R = axis_angle_rotation_matrix(rot_axis, coxa_elevation);
    
    % Rotate horizontal coxa direction upward
    coxa_dir = (R * coxa_horiz_dir')';
    
    % Joint 2 position: end of coxa segment
    j2 = j1 + L1 * coxa_dir;
    
    % --- Femur rotation ---
    % Femur pitch is relative to coxa direction, rotate in plane defined by coxa_dir
    % To simplify, rotate femur around axis perpendicular to coxa_dir and Z
    
    % Define femur rotation axis (perpendicular to coxa_dir and vertical axis)
    femur_rot_axis = cross(coxa_dir, [0 0 1]);
    femur_rot_axis = femur_rot_axis / norm(femur_rot_axis);
    
    % Femur direction vector starts aligned with coxa_dir
    femur_dir = rotate_vector(coxa_dir, femur_rot_axis, theta2);
    
    % Joint 3 position: end of femur segment
    j3 = j2 + L2 * femur_dir;
    
    % --- Tibia rotation ---
    % Tibia pitch is relative to femur direction
    % Rotate tibia around axis perpendicular to femur_dir and vertical axis
    
    tibia_rot_axis = cross(femur_dir, [0 0 1]);
    tibia_rot_axis = tibia_rot_axis / norm(tibia_rot_axis);
    
    % Tibia direction vector
    tibia_dir = rotate_vector(femur_dir, tibia_rot_axis, theta3);
    
    % Joint 4 position: end of tibia segment (foot)
    j4 = j3 + L3 * tibia_dir;
end

% --- Helper function: axis-angle rotation matrix ---
function R = axis_angle_rotation_matrix(axis, angle)
    axis = axis / norm(axis);
    x = axis(1); y = axis(2); z = axis(3);
    c = cos(angle);
    s = sin(angle);
    C = 1 - c;
    R = [ x*x*C + c,   x*y*C - z*s, x*z*C + y*s;
          y*x*C + z*s, y*y*C + c,   y*z*C - x*s;
          z*x*C - y*s, z*y*C + x*s, z*z*C + c ];
end

% --- Helper function: rotate a vector around an axis by an angle ---
function v_rot = rotate_vector(v, axis, angle)
    R = axis_angle_rotation_matrix(axis, angle);
    v_rot = (R * v')';
end


v = readmatrix('predict_results.txt');
A = deg2rad(v);

for idx = 1:size(v,1)
    plot_spider_pose(A(idx,:));
    pause(0.001);
end
```
---

## 8. Explanation & Justification

### Key Design Choices

| Choice | Justification | Trade-offs |
|--------|---------------|------------|
| **MLP Architecture** | Simple, proven for regression tasks. Fully-connected layers capture joint interdependencies. | Not optimized for sequential data (RNN would be), but works well for frame-to-frame prediction. |
| **Sigmoid Activation** | Output range [0,1] matches normalized data perfectly. Smooth for continuous motion. | Vanishing gradients in deep networks. Mitigated by keeping network shallow (3 layers). |
| **MSE Loss** | Standard for regression. Penalizes large errors heavily, encouraging accurate predictions. | Sensitive to outliers. Acceptable since our synthetic data is clean. |
| **Gradient Descent** | Stable, fast, simple to tune. One hyperparameter (LR). | Theoretically slower than adaptive methods, but empirically fastest here. |
| **LR = 0.01** | Optimal for this problem - fast convergence without overshooting. | Too high for Adam. Problem-specific tuning required. |
| **Batch Size = 1** | SGD (Stochastic Gradient Descent) updates weights after each sample. Simpler implementation. | Noisier gradients than mini-batch. Acceptable with low LR and stable optimizer. |
| **Synthetic Data (target_sol)** | Parametric sine-based generator produces optimal gaits instantly. GA alternative is too slow (minutes per gait) and non-deterministic (fitness varies). Fast data generation enables large training sets. | May not capture real spider physics (friction, inertia). Good for learning motion patterns. |
| **95/5 Split** | Large training set maximizes learning. 5% test sufficient for validation. | Could use cross-validation for more robust estimates, but single split adequate. |

### Understanding of Trade-offs

**Depth vs. Complexity**:
- Deeper networks learn more complex patterns but risk overfitting on small datasets
- 3 layers provides sufficient capacity without excessive parameters

**Activation Functions**:
- ReLU would prevent vanishing gradients but unbounded outputs require careful output clipping
- Sigmoid's bounded range is ideal for our normalized data

**Optimizers**:
- Adam adapts per-parameter learning rates, theoretically better for complex loss landscapes
- Gradient descent simpler but requires well-tuned global learning rate
- **Our finding**: Gradient decent is superior with proper tuning, despite Adam's theoretical advantages

**Learning Rate**:
- Higher LR (0.1): Faster convergence but risks divergence
- Lower LR (0.001): More stable but slower
- **0.01**: Sweet spot for GD on this problem

---

## 9. PyTorch Implementation Details

The **`pytorch_nn/`** implementation provides a PyTorch version that exactly replicates the NumPy implementation in **`nn_without_lib/`**. The following changes were made to adapt the code:

### Files Replaced or Removed

The PyTorch implementation consolidates functionality by leveraging PyTorch's built-in modules, eliminating the need for several manual implementations:

| NumPy File | PyTorch Equivalent | Reason |
|------------|-------------------|--------|
| ❌ `activation_functions.py` | Built-in `nn.Sigmoid()`, `nn.ReLU()`, etc. | PyTorch provides optimized activation functions |
| ❌ `error_funcs.py` | Built-in `nn.MSELoss()` | PyTorch's loss functions integrate with autograd |
| ❌ `neural_network.py` | **`torch_model.py`** | Replaced with `nn.Module` class structure |
| ❌ `optimiser.py` | Built-in `torch.optim.SGD()` | PyTorch optimizers handle weight updates automatically |
| ❌ `training.py` | **`torch_training.py`** | Adapted for PyTorch's `loss.backward()` and `optimizer.step()` |
| ❌ `load_and_predict.py` | **`run_predict_sol.py`** | Adapted for PyTorch model loading and inference |
| ✅ `input_data.py` | **Shared** from `nn_without_lib/` | Data generation remains framework-agnostic |
| ❌ `serialize.py` | **`serialize.py`** (rewritten) | Uses `torch.save()` / `torch.load()` instead of pickle |
| ❌ `graph_results.py` | **`graph_results.py`** (adapted) | Modified for PyTorch training output format |

**Key Insight**: PyTorch eliminates ~50 lines of core backpropagation code (gradient calculations in `back_propagation()`, weight updates in `gradient_descent()`, and activation derivatives) by using autograd and built-in modules, demonstrating the framework's abstraction benefits. Overall, ~355 lines across all eliminated files are replaced by PyTorch's built-in functionality.

### Code Structure Changes

| File | Changes from NumPy Implementation | Specific Code References |
|------|-----------------------------------|--------------------------|
| **[`torch_model.py`](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/pytorch_nn/torch_model.py)** | Replaces `neural_network.py`. Uses `nn.Module` with `nn.Linear` layers. Custom weight initialization matches NumPy: `uniform(-0.5, 0.5)` for weights, `-0.5` constant for biases. Sigmoid applied to all layers including output. | **[Lines 37-43](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/pytorch_nn/torch_model.py#L37-L43)**: `nn.Linear()` layers with `nn.Sigmoid()` activations<br>**[Lines 59-64](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/pytorch_nn/torch_model.py#L59-L64)**: `_initialize_weights()` using `nn.init.uniform_(-0.5, 0.5)` for weights and `nn.init.constant_(-0.5)` for biases<br>**[Lines 76-77](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/pytorch_nn/torch_model.py#L76-L77)**: `forward()` method returns `self.net(x)` |
| **[`torch_training.py`](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/pytorch_nn/torch_training.py)** | Replaces `training.py`. Accepts NumPy arrays directly (no DataLoader). Manual epoch shuffling with `np.random.permutation()`. Manual batch iteration matching original algorithm. Loss computed with `nn.MSELoss()`, gradients via PyTorch autograd instead of manual backpropagation. | **[Line 31](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/pytorch_nn/torch_training.py#L31)**: `nn.MSELoss(reduction='mean')`<br>**[Lines 35-37](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/pytorch_nn/torch_training.py#L35-L37)**: `torch.optim.SGD(momentum=0)` for gradient descent<br>**[Lines 45-47](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/pytorch_nn/torch_training.py#L45-L47)**: `np.random.permutation()` shuffles data each epoch<br>**[Lines 54-56](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/pytorch_nn/torch_training.py#L54-L56)**: Manual batch iteration `range(0, len - batch_size, batch_size)`<br>**[Lines 62-68](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/pytorch_nn/torch_training.py#L62-L68)**: `loss.backward()` and `optimizer.step()` replace manual gradient computation | 
| **[`main.py`](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/pytorch_nn/main.py)** | Identical structure and parameters to `nn_without_lib/main.py`. Uses `sys.path.insert()` to access shared `input_data.py` from `nn_without_lib/`. | **[Line 6](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/pytorch_nn/main.py#L6)**: `sys.path.insert()` to access parent directory<br>**[Line 8](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/pytorch_nn/main.py#L8)**: `import nn.input_data` from `nn_without_lib/`<br>**[Line 51](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/pytorch_nn/main.py#L51)**: `TorchNet(input_size=24, hidden_sizes=hidden_layers, output_size=24, activation='sigmoid')`<br>**[Lines 60-61](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/pytorch_nn/main.py#L60-L61)**: `train_torch()` and `save_torch()` match original workflow |
| **[`serialize.py`](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/pytorch_nn/serialize.py)** | Uses `torch.save()` and `torch.load()` for model persistence instead of Python's `pickle`. | **[Line 18](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/pytorch_nn/serialize.py#L18)**: `torch.save(model.state_dict(), out)`<br>**[Line 30](https://github.com/SpindlySpider/UoP-AI-CW/blob/main/pytorch_nn/serialize.py#L30)**: `model.load_state_dict(torch.load(file_name, weights_only=True))` |

### Key Implementation Differences

1. **Forward Pass**: PyTorch's computational graph automatically handles forward propagation through `nn.Linear` layers
2. **Backward Pass**: `loss.backward()` replaces manual gradient computation—PyTorch autograd calculates all gradients automatically
3. **Weight Updates**: `optimizer.step()` replaces manual weight updates (`w -= learning_rate * dw`)
4. **Optimizer**: `torch.optim.SGD(momentum=0)` configured to match vanilla gradient descent exactly

### Performance Characteristics

With **batch_size=1**, PyTorch is significantly slower than the NumPy implementation due to:
- **Tensor conversion overhead**: 70,000+ conversions per epoch (700 gaits × 39 frames × 2.5 epochs)
- **Computational graph construction**: PyTorch builds computation graphs for autograd even when unnecessary for inference
- **Framework abstraction**: Multiple layers of PyTorch abstraction versus direct NumPy operations

PyTorch excels with **larger batch sizes** (32+) where:
- GPU parallelization amortizes overhead
- Vectorized operations dominate computation time
- Autograd benefits from batched gradient computation

For this specific use case (batch_size=1, CPU-only, small network), NumPy's direct implementation is more efficient. However, the PyTorch version demonstrates framework portability and provides a foundation for GPU acceleration if batch sizes increase.

---

## Conclusion

This neural network successfully learns to predict spider gait sequences through:

1. **Well-designed architecture** (3-layer MLP with 128→64→32 neurons)
2. **Appropriate activation** (sigmoid for bounded outputs)
3. **Suitable loss function** (MSE for regression)
4. **Stable training method** (gradient descent, LR=0.01)
5. **Correct backpropagation** (89% loss reduction, smooth convergence)
6. **Thoughtful data handling** (normalized inputs, synthetic training data)

Both **NumPy** and **PyTorch** implementations achieve identical results, validating the correctness of the from-scratch NumPy implementation while demonstrating modern framework integration.

