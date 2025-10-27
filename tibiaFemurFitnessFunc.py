import numpy as np

def fitness_outer_joints(chromosome):
    """
    Fitness function for outer two joints of each spider leg.
    Focuses on symmetry, smoothness, and coordination.
    """
    # Indices for outer 2 joints of each leg (0-based)
    outer_joint_indices = [1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23]
    outer_angles = np.array([chromosome[i] for i in outer_joint_indices])
    
    # Parameters
    theta_max = np.pi / 2         # max hinge movement (90 degrees)
    theta_min_ext, theta_max_ext = -np.pi/4, np.pi/4
    theta_mid = (theta_min_ext + theta_max_ext) / 2
    theta_range = theta_max_ext - theta_min_ext
    
    # 1. Symmetry score
    left_legs = outer_angles[:8]   # legs 1-4
    right_legs = outer_angles[8:]  # legs 5-8
    symmetry = 1 - np.mean(np.abs(left_legs - (-right_legs)) / np.pi)
    
    # 2. Smoothness score
    smoothness = 1 - np.mean((outer_angles / theta_max) ** 2)
    
    # 3. Extension coordination score
    extension = 1 - np.mean(np.abs(outer_angles - theta_mid) / (theta_range / 2))
    
    # Weighted combination
    w1, w2, w3 = 0.4, 0.3, 0.3
    fitness = max(0.0, w1 * symmetry + w2 * smoothness + w3 * extension)
    
    return fitness