import numpy as np

# Read the file
def read_matrix_from_file(filename, rows, cols):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse each line (each line is a frame with comma-separated angles)
    matrix = []
    for line in lines[:rows]:  # Only read up to 'rows' lines
        values = [float(x) for x in line.strip().split(',')]
        matrix.append(values)
    
    return np.array(matrix)

