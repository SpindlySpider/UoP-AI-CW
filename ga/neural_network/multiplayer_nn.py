import numpy as np
import os
from matrix_converter import read_matrix_from_file
from random import random

class Full_NN(object):

    def __init__(self, X=2, HL=[2,2], Y=2):
        
        self.X = X
        self.HL = HL
        self.Y = Y

        L = [X] + HL + [Y]

        W = []
        for i in range(len(L)-1):
            w = np.random.rand(L[i], L[i+1])

            W.append(w)

        self.W = W

        Der = []
        for i in range(len(L) - 1):
            d=np.zeros((L[i],L[i+1]))

            Der.append(d)
        self.Der = Der
        
        out = []
        for i in range(len(L)):
            o=np.zeros(L[i])
            out.append(o)
        self.out = out

    def FF(self,x):

        out = x

        self.out[0] = x

        for i, w in enumerate(self.W):
            Xnext = np.dot(out, w)

            # Use sigmoid for hidden layers, linear for output layer
            if i < len(self.W) - 1:
                out = self.sigmoid(Xnext)
            else:
                out = Xnext  # Linear output for regression

            self.out[i+1] = out

        return out
    
    def BP(self, Er):

        for i in reversed(range(len(self.Der))):
            out = self.out[i+1]

            # Use sigmoid derivative for hidden layers, 1.0 for linear output
            if i < len(self.Der) - 1:
                D = Er * self.sigmoid_Der(out)
            else:
                D = Er  # Linear output derivative is 1

            D_fixed = D.reshape(D.shape[0], -1).T

            this_out = self.out[i]

            this_out = this_out.reshape(this_out.shape[0], -1)

            self.Der[i] = np.dot(this_out,D_fixed)

            Er = np.dot(D, self.W[i].T)

    def train_nn(self, x, target, epochs, lr):
        for i in range(epochs):
            S_errors = 0

            for j, input in enumerate (x):

                t = target[j]

                output = self.FF(input)

                e=t-output
                self.BP(e)

                self.GD(lr)

                S_errors += self.msqe(t,output)
            
            avg_error = S_errors / len(x)
            if i % 50 == 0 or i < 10:
                print(f"Epoch {i}/{epochs} | Average Error: {avg_error:.3f}")

    def GD(self, lr=0.05):

        for i in range(len(self.W)):
            W = self.W[i]
            Der = self.Der[i]
            W += Der*lr

    def sigmoid(self, x):

        y = 1.0/ (1+np.exp(-x))
        return y
    
    def sigmoid_Der(self,x):

        sig_der = x*(1.0 - x)
        return sig_der
    
    def msqe (self, t, output):
        msq = np.average((t-output)**2)
        return msq
    
if __name__ == "__main__":
    # Read 300 frames, each with 24 angles
    # Use path relative to script location to find results.txt in project root
    script_dir = os.path.dirname(__file__)
    results_path = os.path.join(script_dir, '..', '..', 'results.txt')
    data = read_matrix_from_file(results_path, 300, 24)

    print(f"Loaded {len(data)} frames with {len(data[0])} angles each")

    # Normalize data to [-1, 1] range for better training
    data_min = data.min()
    data_max = data.max()
    data_normalized = 2 * (data - data_min) / (data_max - data_min) - 1
    
    print(f"Data range: [{data_min:.3f}, {data_max:.3f}] normalized to [-1, 1]")

    # Split into training inputs (frames 0-298) and targets (frames 1-299)
    # This trains the NN to predict the next pose given the current pose
    training_input = data_normalized[:-1]  # frames 0 to 298
    targets = data_normalized[1:]           # frames 1 to 299

    print(f"Training inputs: {len(training_input)} frames")
    print(f"Targets: {len(targets)} frames")

    # Create neural network: 24 inputs (current pose) -> hidden layers -> 24 outputs (next pose)
    # Smaller network to prevent overfitting with limited training data
    nn = Full_NN(24, [48, 24], 24)

    print("\n=============== Training the Neural Network ===============")
    nn.train_nn(training_input, targets, 1000, 0.05)
    print("Training complete!")

    # Test the network with a sample input
    test_input = training_input[0]
    expected_output = targets[0]

    NN_output = nn.FF(test_input)

    # Denormalize for display
    test_input_denorm = (test_input + 1) * (data_max - data_min) / 2 + data_min
    expected_output_denorm = (expected_output + 1) * (data_max - data_min) / 2 + data_min
    NN_output_denorm = (NN_output + 1) * (data_max - data_min) / 2 + data_min

    print("\n=============== Testing the Network ===============")
    print(f"Test input (24 angles): {test_input_denorm}")
    print(f"Expected output (24 angles): {expected_output_denorm}")
    print(f"NN output (24 angles): {NN_output_denorm}")
    error = np.average((expected_output - NN_output)**2)
    print(f"Mean Squared Error (normalized): {error:.3f}")
    error_denorm = np.average((expected_output_denorm - NN_output_denorm)**2)
    print(f"Mean Squared Error (degrees): {error_denorm:.3f}")
    
    # Test on multiple samples
    print("\n--- Testing on 5 random samples ---")
    test_indices = [0, 50, 100, 150, 200]
    total_error = 0
    for idx in test_indices:
        test_in = training_input[idx]
        test_out = targets[idx]
        pred_out = nn.FF(test_in)
        sample_error = np.average((test_out - pred_out)**2)
        total_error += sample_error
        print(f"Sample {idx}: MSE = {sample_error:.3f}")
    
    avg_test_error = total_error / len(test_indices)
    print(f"Average test MSE (normalized): {avg_test_error:.3f}")
    print("====================================================")
    
    # Save the trained weights
    weights_file = "nn_weights.npz"
    np.savez(weights_file, *nn.W)
    print(f"\n✓ Weights saved to {weights_file}")