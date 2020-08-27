import numpy as np

def sigmoid(Z):
    # Activation Function - Sigmoid
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    # Activation Function - ReLU
    return np.maximum(0, Z)

def sigmoid_derivative(Z):
    # Activation Function - Sigmoid Derivative
    return Z * (1 - Z)

def relu_derivative(Z):
    # Activation Function - ReLU Derivative
    Z[Z <= 0] = 0
    Z[Z > 0] = 1
    return Z

def cost(A, Y):
    # Computes the cost
    m = Y.shape[1]
    return -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

def dAL(AL, Y):
    # Derivative of the Cost of the model
    return - np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)
