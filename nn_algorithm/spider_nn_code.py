import numpy as np
from random import random
import matplotlib.pyplot as plt
import pickle
import json

"""
NEURAL NETWORK FOR SPIDER JOINT ANGLE CONFIGURATION
====================================================

ARCHITECTURE DESIGN RATIONALE:
------------------------------
Input Layer: 6 neurons
  - Target X, Y, Z position (3 values): Where the spider should move
  - Current phase (1 value): Gait cycle phase (0-1) for walking coordination
  - Terrain slope (2 values): X and Z slope for adaptation
  
Hidden Layers: [32, 48, 48, 32] neurons
  - Deep architecture to capture complex joint relationships
  - Wider middle layers (48 neurons) to learn rich representations
  - Gradual expansion then contraction pattern
  
Output Layer: 24 neurons
  - 8 legs × 3 joints per leg = 24 joint angles
  - Output range scaled to [-π, π] for joint rotation limits

ACTIVATION FUNCTIONS:
--------------------
- Hidden layers: Hyperbolic tangent (tanh)
  * Reason: Centers output around 0, suitable for normalized data
  * Provides smooth gradients and avoids vanishing gradient issues
  * Output range [-1, 1] matches our normalized joint angles
  
- Output layer: Tanh followed by scaling
  * Produces bounded outputs suitable for physical joint constraints
  * Prevents unrealistic joint configurations

LOSS FUNCTION:
--------------
- Mean Squared Error (MSE)
  * Appropriate for regression problems
  * Penalizes large errors more heavily
  * Easy to differentiate for backpropagation

LEARNING STRATEGY:
------------------
- Stochastic Gradient Descent with mini-batches
- Learning rate: 0.01 (tuned through experimentation)
- Training on synthetic data generated from biomechanically-inspired patterns
"""

class Spider_NN(object):
    """
    Enhanced Neural Network for Spider Joint Angle Generation
    """
    
    def __init__(self, X=6, HL=[32, 48, 48, 32], Y=24):
        """
        Initialize the neural network architecture
        
        Parameters:
        -----------
        X : int
            Number of input features (default: 6)
            - Target position (3), phase (1), terrain (2)
        HL : list
            Hidden layer configuration (default: [32, 48, 48, 32])
        Y : int
            Number of outputs (default: 24 joint angles)
        """
        self.X = X
        self.HL = HL
        self.Y = Y
        
        # Complete layer structure
        L = [X] + HL + [Y]
        
        # Initialize weights with Xavier initialization for better convergence
        W = []
        for i in range(len(L)-1):
            # Xavier initialization: scale by sqrt(1/n_in)
            scale = np.sqrt(2.0 / (L[i] + L[i+1]))
            w = np.random.randn(L[i], L[i+1]) * scale
            W.append(w)
        self.W = W
        
        # Initialize derivative storage
        Der = []
        for i in range(len(L)-1):
            d = np.zeros((L[i], L[i+1]))
            Der.append(d)
        self.Der = Der
        
        # Initialize output storage for each layer
        out = []
        for i in range(len(L)):
            o = np.zeros(L[i])
            out.append(o)
        self.out = out
        
        # Training history
        self.loss_history = []
        self.epoch_losses = []
        
    def tanh(self, x):
        """
        Hyperbolic tangent activation function
        Better than sigmoid for centered data
        """
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """
        Derivative of tanh for backpropagation
        """
        return 1.0 - x**2
    
    def FF(self, x):
        """
        Forward propagation through the network
        
        Parameters:
        -----------
        x : numpy array
            Input vector
            
        Returns:
        --------
        out : numpy array
            Output joint angles (scaled to [-π, π])
        """
        out = x
        self.out[0] = x
        
        for i, w in enumerate(self.W):
            Xnext = np.dot(out, w)
            out = self.tanh(Xnext)
            self.out[i+1] = out
        
        # Scale output from [-1, 1] to [-π, π] for realistic joint angles
        out_scaled = out * np.pi
        
        return out_scaled
    
    def BP(self, Er):
        """
        Backpropagation to calculate weight gradients
        
        Parameters:
        -----------
        Er : numpy array
            Output error (target - predicted)
        """
        # Scale error back to match tanh output range
        Er = Er / np.pi
        
        for i in reversed(range(len(self.Der))):
            out = self.out[i+1]
            
            # Calculate delta using tanh derivative
            D = Er * self.tanh_derivative(out)
            D_fixed = D.reshape(D.shape[0], -1).T
            
            this_out = self.out[i]
            this_out = this_out.reshape(this_out.shape[0], -1)
            
            # Calculate weight gradient
            self.Der[i] = np.dot(this_out, D_fixed)
            
            # Propagate error backwards
            Er = np.dot(D, self.W[i].T)
    
    def GD(self, lr=0.01):
        """
        Gradient Descent weight update
        
        Parameters:
        -----------
        lr : float
            Learning rate
        """
        for i in range(len(self.W)):
            W = self.W[i]
            Der = self.Der[i]
            W += Der * lr
    
    def msqe(self, t, output):
        """
        Mean Squared Error loss function
        """
        return np.average((t - output)**2)
    
    def train_nn(self, x, target, epochs, lr=0.01, batch_size=32, verbose=True):
        """
        Train the neural network
        
        Parameters:
        -----------
        x : numpy array
            Training inputs (N x 6)
        target : numpy array
            Target outputs (N x 24)
        epochs : int
            Number of training epochs
        lr : float
            Learning rate
        batch_size : int
            Mini-batch size for training
        verbose : bool
            Print training progress
        """
        n_samples = len(x)
        
        for epoch in range(epochs):
            epoch_error = 0
            
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            target_shuffled = target[indices]
            
            # Mini-batch training
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_x = x_shuffled[batch_start:batch_end]
                batch_target = target_shuffled[batch_start:batch_end]
                
                batch_error = 0
                for j in range(len(batch_x)):
                    input_data = batch_x[j]
                    t = batch_target[j]
                    
                    # Forward pass
                    output = self.FF(input_data)
                    
                    # Calculate error
                    e = t - output
                    
                    # Backward pass
                    self.BP(e)
                    
                    # Accumulate error
                    batch_error += self.msqe(t, output)
                
                # Update weights after batch
                self.GD(lr)
                epoch_error += batch_error
            
            # Record epoch loss
            avg_loss = epoch_error / n_samples
            self.epoch_losses.append(avg_loss)
            
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
        
        if verbose:
            print("Training completed!")
    
    def save_model(self, filename='spider_nn_model.pkl'):
        """
        Save the trained model to disk
        """
        model_data = {
            'W': self.W,
            'X': self.X,
            'HL': self.HL,
            'Y': self.Y,
            'loss_history': self.epoch_losses
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='spider_nn_model.pkl'):
        """
        Load a trained model from disk
        """
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.W = model_data['W']
        self.X = model_data['X']
        self.HL = model_data['HL']
        self.Y = model_data['Y']
        self.epoch_losses = model_data.get('loss_history', [])
        print(f"Model loaded from {filename}")


def generate_training_data(n_samples=5000):
    """
    Generate biomechanically-inspired training data for spider locomotion
    
    TRAINING DATA GENERATION STRATEGY:
    ----------------------------------
    This function creates synthetic training data based on:
    1. Tripod gait pattern (common in hexapods/arachnids)
    2. Phase-based coordination between legs
    3. Terrain adaptation
    4. Realistic joint angle ranges
    
    Returns:
    --------
    inputs : numpy array (N x 6)
        Input features: [target_x, target_y, target_z, phase, slope_x, slope_z]
    targets : numpy array (N x 24)
        Target joint angles for 8 legs × 3 joints
    """
    inputs = []
    targets = []
    
    for _ in range(n_samples):
        # Input features
        target_x = np.random.uniform(-1, 1)  # Normalized target position
        target_y = np.random.uniform(-0.3, 0.3)  # Vertical component
        target_z = np.random.uniform(-1, 1)
        phase = np.random.uniform(0, 1)  # Gait cycle phase
        slope_x = np.random.uniform(-0.3, 0.3)  # Terrain slope
        slope_z = np.random.uniform(-0.3, 0.3)
        
        input_vec = np.array([target_x, target_y, target_z, phase, slope_x, slope_z])
        
        # Generate joint angles based on biomechanical principles
        joint_angles = []
        
        for leg_idx in range(8):
            # Determine leg phase offset (tripod gait)
            # Legs 0,2,4,6 move together; Legs 1,3,5,7 move together
            leg_phase = (phase + (leg_idx % 2) * 0.5) % 1.0
            
            # Base angles for different leg positions (front, middle, rear)
            if leg_idx < 2:  # Front legs
                base_angle = 0.4
            elif leg_idx < 6:  # Middle legs
                base_angle = 0.0
            else:  # Rear legs
                base_angle = -0.4
            
            # Joint 1 (Coxa): Horizontal rotation
            # Varies with target direction and leg phase
            j1 = base_angle + 0.5 * np.sin(2 * np.pi * leg_phase) * target_x
            j1 += slope_x * 0.3  # Adapt to terrain
            
            # Joint 2 (Femur): Vertical lift
            # Larger movement during swing phase
            if leg_phase < 0.5:  # Swing phase
                j2 = -0.8 + 1.2 * np.sin(np.pi * leg_phase)
            else:  # Stance phase
                j2 = -0.3 + 0.2 * np.sin(np.pi * (leg_phase - 0.5))
            j2 += target_y * 0.5  # Adjust for vertical target
            
            # Joint 3 (Tibia): Extension
            # Coordinates with femur for ground contact
            if leg_phase < 0.5:
                j3 = 0.6 - 0.8 * np.sin(np.pi * leg_phase)
            else:
                j3 = 0.4 - 0.3 * np.cos(np.pi * (leg_phase - 0.5))
            j3 -= slope_z * 0.2  # Terrain adaptation
            
            joint_angles.extend([j1, j2, j3])
        
        target_vec = np.array(joint_angles)
        
        inputs.append(input_vec)
        targets.append(target_vec)
    
    return np.array(inputs), np.array(targets)


def visualize_training(nn, test_inputs, test_targets):
    """
    Visualize training progress and network performance
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Training Loss
    axes[0, 0].plot(nn.epoch_losses, linewidth=2, color='blue')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Mean Squared Error', fontsize=12)
    axes[0, 0].set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Sample Predictions vs Targets
    sample_idx = np.random.randint(0, len(test_inputs))
    prediction = nn.FF(test_inputs[sample_idx])
    target = test_targets[sample_idx]
    
    joints = np.arange(24)
    axes[0, 1].plot(joints, target, 'o-', label='Target', linewidth=2, markersize=6)
    axes[0, 1].plot(joints, prediction, 's-', label='Predicted', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Joint Index', fontsize=12)
    axes[0, 1].set_ylabel('Angle (radians)', fontsize=12)
    axes[0, 1].set_title('Sample Prediction vs Target', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Prediction Error Distribution
    all_predictions = np.array([nn.FF(inp) for inp in test_inputs[:100]])
    all_targets = test_targets[:100]
    errors = all_targets - all_predictions
    
    axes[1, 0].hist(errors.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Prediction Error (radians)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Joint-wise Mean Absolute Error
    mae_per_joint = np.mean(np.abs(errors), axis=0)
    axes[1, 1].bar(joints, mae_per_joint, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Joint Index', fontsize=12)
    axes[1, 1].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[1, 1].set_title('MAE per Joint', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('spider_nn_training_results.png', dpi=300, bbox_inches='tight')
    print("Training visualization saved to 'spider_nn_training_results.png'")
    plt.show()


def visualize_spider_pose(joint_angles, title="Spider Pose"):
    """
    Visualize spider joint angles as a simple 2D representation
    Each leg shows the three joint angles
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Reshape to (8 legs, 3 joints)
    joints = joint_angles.reshape(8, 3)
    
    # Define leg positions around spider body (8 legs)
    leg_base_angles = np.linspace(0, 2*np.pi, 9)[:-1]  # 8 legs evenly distributed
    
    for leg_idx in range(8):
        base_angle = leg_base_angles[leg_idx]
        base_x = np.cos(base_angle) * 0.3
        base_y = np.sin(base_angle) * 0.3
        
        # Get joint angles for this leg
        j1, j2, j3 = joints[leg_idx]
        
        # Calculate leg segment positions (simplified 2D projection)
        segment1_len = 0.3
        segment2_len = 0.3
        segment3_len = 0.2
        
        # Segment 1 (Coxa)
        angle1 = base_angle + j1
        x1 = base_x + segment1_len * np.cos(angle1)
        y1 = base_y + segment1_len * np.sin(angle1)
        
        # Segment 2 (Femur)
        angle2 = angle1 + j2
        x2 = x1 + segment2_len * np.cos(angle2)
        y2 = y1 + segment2_len * np.sin(angle2)
        
        # Segment 3 (Tibia)
        angle3 = angle2 + j3
        x3 = x2 + segment3_len * np.cos(angle3)
        y3 = y2 + segment3_len * np.sin(angle3)
        
        # Draw leg
        ax.plot([base_x, x1], [base_y, y1], 'b-', linewidth=3)
        ax.plot([x1, x2], [y1, y2], 'g-', linewidth=3)
        ax.plot([x2, x3], [y2, y3], 'r-', linewidth=3)
        
        # Mark joints
        ax.plot([base_x, x1, x2, x3], [base_y, y1, y2, y3], 'ko', markersize=6)
    
    # Draw body
    body = plt.Circle((0, 0), 0.3, color='gray', alpha=0.5)
    ax.add_patch(body)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('spider_pose_visualization.png', dpi=300, bbox_inches='tight')
    print("Pose visualization saved to 'spider_pose_visualization.png'")
    plt.show()


if __name__ == "__main__":
    print("="*70)
    print("SPIDER JOINT ANGLE NEURAL NETWORK - TRAINING SYSTEM")
    print("="*70)
    print()
    
    # Generate training data
    print("Generating training data...")
    training_inputs, training_targets = generate_training_data(n_samples=5000)
    print(f"Generated {len(training_inputs)} training samples")
    print(f"Input shape: {training_inputs.shape}")
    print(f"Target shape: {training_targets.shape}")
    print()
    
    # Generate test data
    print("Generating test data...")
    test_inputs, test_targets = generate_training_data(n_samples=500)
    print(f"Generated {len(test_inputs)} test samples")
    print()
    
    # Create and train network
    print("Initializing neural network...")
    print("Architecture: 6 -> [32, 48, 48, 32] -> 24")
    nn = Spider_NN(X=6, HL=[32, 48, 48, 32], Y=24)
    print()
    
    # Train
    print("Training neural network...")
    nn.train_nn(training_inputs, training_targets, epochs=500, lr=0.01, batch_size=32)
    print()
    
    # Save model
    nn.save_model('spider_nn_model.pkl')
    print()
    
    # Test network
    print("="*70)
    print("TESTING NEURAL NETWORK")
    print("="*70)
    
    test_input = test_inputs[0]
    test_target = test_targets[0]
    prediction = nn.FF(test_input)
    
    print("Test Input:")
    print(f"  Target position: ({test_input[0]:.3f}, {test_input[1]:.3f}, {test_input[2]:.3f})")
    print(f"  Phase: {test_input[3]:.3f}")
    print(f"  Terrain slope: ({test_input[4]:.3f}, {test_input[5]:.3f})")
    print()
    
    print("Target Joint Angles (first 6):")
    print(test_target[:6])
    print()
    
    print("Predicted Joint Angles (first 6):")
    print(prediction[:6])
    print()
    
    error = np.mean(np.abs(test_target - prediction))
    print(f"Mean Absolute Error: {error:.6f} radians")
    print()
    
    # Visualize results
    print("Generating visualizations...")
    visualize_training(nn, test_inputs, test_targets)
    visualize_spider_pose(prediction, title="Neural Network Generated Spider Pose")
    
    print()
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)