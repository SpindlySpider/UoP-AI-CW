import numpy as np

# Read the file
def read_matrix_from_file(filename, rows, cols):
    with open(filename, 'r') as f:
        data = f.read()

    # Split by commas and convert to float
    values = [float(x) for x in data.strip().split(',')]

    # Reshape into 300x24 matrix
    matrix = np.array(values).reshape(filename, rows, cols)
    return matrix

