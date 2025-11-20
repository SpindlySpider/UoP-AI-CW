import numpy as np
import sys
import os
# Ensure the project root and the 'ga' folder are on sys.path so imports work when running this file as a script
_script_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_script_dir, '..'))        # add parent directory (project root)
sys.path.insert(0, os.path.join(_script_dir, '..', 'ga'))  # add ga directory if needed

# Import from ga package
sys.path.append(os.path.join(_script_dir, '..', 'ga'))
from target_sol import random_sol

# Import local modules
from multiplayer_nn import Full_NN

def verify_data_shape(data, expected_cols=24):
    """Verify that data is properly shaped as arrays of size 24"""
    print("\n" + "="*60)
    print("Data Shape Verification")
    print("="*60)
    
    if isinstance(data, list):
        data = np.array(data)
    
    print(f"Data type: {type(data)}")
    print(f"Data shape: {data.shape}")
    print(f"Expected shape: (num_frames, {expected_cols})")
    
    if len(data.shape) != 2:
        print(f"ERROR: Data should be 2D array, got {len(data.shape)}D")
        return False
    
    if data.shape[1] != expected_cols:
        print(f"ERROR: Each frame should have {expected_cols} joints, got {data.shape[1]}")
        return False
    
    print(f"✓ Data is correctly shaped with {data.shape[0]} frames of {expected_cols} joint angles each")
    
    # Show sample frames
    print(f"\nFirst frame (24 joint angles):")
    print(f"  {data[0]}")
    print(f"\nLast frame (24 joint angles):")
    print(f"  {data[-1]}")
    
    # Show statistics
    print(f"\nData statistics:")
    print(f"  Min value: {data.min():.3f}")
    print(f"  Max value: {data.max():.3f}")
    print(f"  Mean value: {data.mean():.3f}")
    print(f"  Std deviation: {data.std():.3f}")
    
    return True

def test_nn_predictions(nn, data, data_min, data_max, num_test_frames=10):
    """Test neural network predictions on sequential frames"""
    print("\n" + "="*60)
    print("Neural Network Prediction Testing")
    print("="*60)
    
    print(f"Testing predictions on {num_test_frames} frame transitions...")
    
    total_error = 0
    for i in range(num_test_frames):
        current_frame = data[i]
        next_frame_actual = data[i + 1]
        
        # Normalize current frame
        current_norm = 2 * (current_frame - data_min) / (data_max - data_min) - 1
        
        # Predict next frame
        next_frame_pred_norm = nn.FF(current_norm)
        
        # Denormalize prediction
        next_frame_pred = (next_frame_pred_norm + 1) * (data_max - data_min) / 2 + data_min
        
        # Calculate error
        frame_error = np.mean((next_frame_actual - next_frame_pred)**2)
        total_error += frame_error
        
        print(f"\nFrame {i} -> {i+1}:")
        print(f"  MSE: {frame_error:.3f}")
        print(f"  RMSE: {np.sqrt(frame_error):.3f}")
        print(f"  Actual : {next_frame_actual}")
        print(f"  Predicted : {next_frame_pred}")
        
        # Check if prediction is reasonable
        max_diff = np.max(np.abs(next_frame_actual - next_frame_pred))
        print(f"  Max absolute difference: {max_diff:.3f}")
    
    avg_error = total_error / num_test_frames
    print(f"\n{'='*60}")
    print(f"Average MSE: {avg_error:.3f}")
    print(f"Average RMSE: {np.sqrt(avg_error):.3f}")
    print(f"{'='*60}")

def save_weights(nn, filename):
    """Save neural network weights to a file"""
    np.savez(filename, *nn.W)
    print(f"\nWeights saved to {filename}")

def load_weights(nn, filename):
    """Load neural network weights from a file"""
    if os.path.exists(filename):
        data = np.load(filename)
        nn.W = [data[f'arr_{i}'] for i in range(len(data.files))]
        print(f"Weights loaded from {filename}")
        return True
    return False

if __name__ == "__main__":
    print("="*60)
    print("Neural Network Gait Predictor - GA Testing")
    print("="*60)
    
    # Generate random solutions for testing GA
    print("\nGenerating random solutions using random_sol for GA testing...")
    
    gait_length = 300
    print(f"Gait length: {gait_length}")
    
    # Generate multiple random solutions to test
    num_test_solutions = 5
    print(f"\nTesting with {num_test_solutions} random solutions...")
    
    for sol_num in range(num_test_solutions):
        print(f"\n{'='*60}")
        print(f"Random Solution #{sol_num + 1}")
        print(f"{'='*60}")
        
        # Generate random gait data
        gait_data = random_sol(gait_length)
        
        # Convert to numpy array
        data = np.array(gait_data)
        
        # Verify data shape
        if not verify_data_shape(data, expected_cols=24):
            print(f"\nWARNING: Solution #{sol_num + 1} shape verification failed!")
            continue
        
        # Normalize data
        data_min = data.min()
        data_max = data.max()
        data_normalized = 2 * (data - data_min) / (data_max - data_min) - 1
        
        print(f"\nNormalization range: [{data_min:.3f}, {data_max:.3f}]")
        print(f"Normalized data range: [{data_normalized.min():.3f}, {data_normalized.max():.3f}]")
        
        # Load pre-trained neural network
        print("\n" + "="*60)
        print("Loading Pre-trained Neural Network")
        print("="*60)
        print("Architecture: 24 -> [48, 24] -> 24")
        
        nn = Full_NN(24, [48, 24], 24)
        
        # Load existing weights
        weights_file = "nn_weights.npz"
        if load_weights(nn, weights_file):
            print("Using pre-trained weights for prediction")
            
            # Test neural network predictions on this random solution
            test_nn_predictions(nn, data, data_min, data_max, num_test_frames=5)
        else:
            print(f"ERROR: No pre-trained weights found at {weights_file}")
            print("Please train the network first using multiplayer_nn.py")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("GA Testing Complete")
    print("="*60)
    print(f"Successfully tested {num_test_solutions} random solutions")
