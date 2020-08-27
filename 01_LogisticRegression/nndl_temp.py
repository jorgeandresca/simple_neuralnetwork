import numpy as np
import sys

sys.path.append("Modules")
import nn_utils
import h5py

class NNDeepLearning:
    """
        This class contains the following functions:
            1) initialize_parameters_random
            2) forward_one_layer
            3) forward_propagation
            4) compute_cost (not sure if needed)
            5) backward_one_layer
            6) backward_propagation
            7) update_parameters
            8) model (this function architects all previous functions)
    """
    np.random.seed(1)

    @staticmethod
    def initialize_parameters_random(model_dimensions):
        """
            Initializes parameters of the model (W, b)
            model_dimensions -> architecture of the model, example: [5,4,4,1]
        """

        parameters_model = {}
        L = len(model_dimensions) - 1  # Number of layers, excluding Inputs layer

        for l in range(1, L + 1):
            layer = {}
            layer['W'] = np.random.randn(model_dimensions[l], model_dimensions[l - 1]) * 0.01  # Shape (l, l-1)
            layer['b'] = np.zeros((model_dimensions[l], 1)).astype(int)  # Shape (l, 1)

            parameters_model["l" + str(l)] = layer

        return parameters_model

    @staticmethod
    def forward_one_layer(A_previous, W, b, activation_function):
        """
            Forwarding only 1 layer ahead
            W.shape = (l, l-1)
            Z, A_prev, A.shape = (l, m)
            Returns A and [cache = A_prev, W, b, Z]
        """
        Z = np.dot(W, A_previous) + b  # W.shape: (l, l-1) / A_previous.shape: (l, m)

        A = None
        if activation_function == "sigmoid":
            A = nn_utils.sigmoid(Z)
        elif activation_function == "relu":
            A = nn_utils.relu(Z)

        cache = A_previous, W, b, Z

        return A, cache

    def forward_propagation(self, X, parameters_model):
        """
            Forwarding through all layers
                1) Compute activation for hidden layers - Store cache
                2) Computer activation for output layer - Store cache
            returns A (the last activation, output), cache_list
        """
        cache_list = {}
        A = X  # X is going to be treated as A, the result of an activation
        L = len(parameters_model)  # Num. of layers in the model / ex. Model: [2, 3, 1], so len(parameters_model) = 2

        # Loop through all layers, except for the output one
        for l in range(1, L + 1):
            A_previous = A  # In this loop, A = previous_A
            W = parameters_model["l" + str(l)]["W"]
            b = parameters_model["l" + str(l)]["b"]

            activation = 'sigmoid' if l == L else 'relu'  # Hidden layers = Relu, Output = Sigmoid

            A, cache = self.forward_one_layer(A_previous, W, b, activation)

            cache_list["c" + str(l)] = cache

        return A, cache_list

    @staticmethod
    def compute_cost(AL, Y):
        cost = nn_utils.cost(AL, Y)
        return cost

    @staticmethod
    def backward_one_layer(dA, cache, activation):
        """
            Backwarding only 1 layer back
            dA -> The derivative of the cost of 1 layer ahead (to the right)
            Returns dA_previous, dW, db (dA_previous -> Derivative of the previous layer cost, to the left)
        """
        A_previous, W, b, Z = cache

        if activation == "relu":
            dZ = dA * nn_utils.relu_derivative(Z)
        elif activation == "sigmoid":
            dZ = dA * nn_utils.sigmoid_derivative(Z)

        m = A_previous.shape[1]
        dW = 1 / m * np.dot(dZ, A_previous.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_previous = np.dot(W.T, dZ)

        return dA_previous, dW, db

    def backward_propagation(self, AL, Y, cache_list):
        """
            Backwarding through all layers
                1) Calculate the derivative of the activation of the last layer (dAL)
                2) Compute dAL, dA_previous, dW, db - Output Layer
                3) Compute dAL, dA_previous, dW, db - All Hidden Layers
            returns grads = Gradients for all layers
        """
        L = len(cache_list)  # Num. of layers

        grads = {}
        # Calculating dAL, dA_previous, dW, db - Output and Hidden Layers
        for l in reversed(range(1, L + 1)):
            current_cache = cache_list["c" + str(l)]
            if l == L:  # Output layer
                dAL = nn_utils.dAL(AL, Y)
                dA_previous, dW, db = self.backward_one_layer(dAL, current_cache, 'sigmoid')
            else:  # Hidden layers
                dA = grads["dA" + str(l)]
                dA_previous, dW, db = self.backward_one_layer(dA, current_cache, 'relu')

            grads["dA" + str(l - 1)] = dA_previous
            grads["dW" + str(l)] = dW
            grads["db" + str(l)] = db

        return grads

    @staticmethod
    def update_parameters(parameters_model, grads, learning_rate):
        """
            Update model parameters (W, b) for all Layers
        """
        L = len(parameters_model)  # Num. of layers in the model / ex. Model: [2, 3, 1], so len(parameters_model) = 2

        for l in range(1, L + 1):
            W = parameters_model["W" + str(l)]
            b = parameters_model["b" + str(l)]

            parameters_model["W" + str(l)] = W - learning_rate * grads["dW" + str(l)]
            parameters_model["b" + str(l)] = b - learning_rate * grads["db" + str(l)]

        return parameters_model

    def model(self, X, Y, layers_dimensions, learning_rate, num_iterations, print_cost):

        np.random.seed(1)
        costs = []

        parameters_model = initialize_parameters_random(layers_dimensions)

        for i in range(0, num_iterations):

            AL, cache_list = self.forward_propagation(X, parameters_model)

            cost = self.compute_cost(AL, Y)

            grads = self.backward_propagation(AL, Y, cache_list)

            parameters_model = self.update_parameters(parameters_model, grads, learning_rate)

            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)
        """
            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per hundreds)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
        """
        return parameters_model


DL = NNDeepLearning()

params = DL.initialize_parameters_random([3, 4, 1])
X = np.random.randn(3, 5)
Y = np.zeros((1, 5))

AL, cache_list = DL.forward_propagation(X, params)

cost = DL.compute_cost(AL, Y)
print(cost)