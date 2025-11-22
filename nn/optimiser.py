from nn.neural_network import Neural_network
import numpy as np

def gradient_descent(nn:Neural_network) -> Neural_network:
    """
    Optimiser used to update weights based on derivatives calculated in back propagation.
    Parameters:
        nn (Neural_network): the neural network to perform gradient descent on.
    Returns:
        Neural network with updated weights and biases
    """
    for i in range(len(nn.weights)):
        nn.weights[i] = nn.weights[i] - nn.derivatives[i]*nn.learning_rate
        # np.average for averaging bias over batches
        nn.bias[i] = nn.bias[i] - np.average(nn.delta[i],axis=0)*nn.learning_rate
    return nn

def adam(nn:Neural_network, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8, gradient_clip:float=1.0) -> Neural_network:
    """
    Optimiser which uses momentum and RMSprop to adjust learning rate during training
    Parameters:
        nn (Neural_network): Neural network to optimize
        beta1 (float): Exponential decay rate for first moment estimates (momentum)
        beta2 (float): Exponential decay rate for second moment estimates (RMSprop)
        epsilon (float): Small constant to prevent division by zero
        gradient_clip (float): Maximum gradient norm to prevent exploding gradients
    Returns:
        Neural network with updated weights and biases
    """
    # Initialize moment vectors if not already present
    if not hasattr(nn, 'adam_m_weights'):
        nn.adam_m_weights = [np.zeros_like(w) for w in nn.weights]
        nn.adam_v_weights = [np.zeros_like(w) for w in nn.weights]
        nn.adam_m_bias = [np.zeros_like(b) for b in nn.bias]
        nn.adam_v_bias = [np.zeros_like(b) for b in nn.bias]
        nn.adam_t = 0
    
    # Increment time step
    nn.adam_t += 1
    
    # Update parameters for each layer
    for i in range(len(nn.weights)):
        # Clip gradients to prevent exploding gradients
        grad_norm = np.linalg.norm(nn.derivatives[i])
        if grad_norm > gradient_clip:
            nn.derivatives[i] = nn.derivatives[i] * (gradient_clip / grad_norm)
        
        # Update biased first moment estimate (momentum) for weights
        nn.adam_m_weights[i] = beta1 * nn.adam_m_weights[i] + (1 - beta1) * nn.derivatives[i]
        
        # Update biased second moment estimate (RMSprop) for weights
        nn.adam_v_weights[i] = beta2 * nn.adam_v_weights[i] + (1 - beta2) * (nn.derivatives[i] ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat_weights = nn.adam_m_weights[i] / (1 - beta1 ** nn.adam_t)
        
        # Compute bias-corrected second moment estimate
        v_hat_weights = nn.adam_v_weights[i] / (1 - beta2 ** nn.adam_t)
        
        # Update weights with safeguard against large updates
        update = nn.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + epsilon)
        nn.weights[i] = nn.weights[i] - update
        
        # Calculate gradient for bias (average over batch)
        bias_gradient = np.average(nn.delta[i], axis=0)
        
        # Clip bias gradients
        bias_grad_norm = np.linalg.norm(bias_gradient)
        if bias_grad_norm > gradient_clip:
            bias_gradient = bias_gradient * (gradient_clip / bias_grad_norm)
        
        # Update biased first moment estimate for bias
        nn.adam_m_bias[i] = beta1 * nn.adam_m_bias[i] + (1 - beta1) * bias_gradient
        
        # Update biased second moment estimate for bias
        nn.adam_v_bias[i] = beta2 * nn.adam_v_bias[i] + (1 - beta2) * (bias_gradient ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat_bias = nn.adam_m_bias[i] / (1 - beta1 ** nn.adam_t)
        
        # Compute bias-corrected second moment estimate
        v_hat_bias = nn.adam_v_bias[i] / (1 - beta2 ** nn.adam_t)
        
        # Update bias with safeguard
        bias_update = nn.learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + epsilon)
        nn.bias[i] = nn.bias[i] - bias_update
    
    return nn
