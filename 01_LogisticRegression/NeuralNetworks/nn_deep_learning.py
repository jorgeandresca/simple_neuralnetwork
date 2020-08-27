import numpy as np
import sys
sys.path.append("../Modules")
import nn_utils
import h5py

class NN_Deep_Learning:
    """
        This class contains the following functions:
            1) initialize_parameters_random
            2) forward_one_layer
            3) forward_propagation
            4) compute_cost (not sure if needed)
            5)
    """
    np.random.seed(1)

    def initialize_parameters_random(self, layers_dimensions):
        """
            layers_dimension -> arquitechture of the model, example: [5,4,3,1]
        """

        parameters_model = {}
        L = len(layers_dimensions) - 1  # Number of layers, doesn't include the input

        for l in range(0, L):
            layer = {}
            layer['W'] = np.random.randn(layers_dimensions[l + 1], layers_dimensions[l]) * 0.01
            layer['b'] = np.zeros((layers_dimensions[l + 1], 1)).astype(int)

            parameters_model["l" + str(l+1)] = layer

        return parameters_model

    def forward_one_layer(self, A_previous, W, b, activation_function):
        """
            Forwarding only 1 layer ahead
            W.shape = (nl, nl-1)
            Z, A_prev, A.shape = (nl, m)
            Returns A and [cache = A_prev, W, b, Z]
        """

        print("--------------------")
        print(A_previous.shape)
        print(W.shape)
        print("--------------------")

        Z = np.dot(W, A_previous) + b

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
        """
        #cache_list = []
        cache_list = {}
        A = X
        L = len(parameters_model) # Number of layers - [10,4,4,1] -> L = 3

        #--- Loop through all layer, except for the last one
        for i in range(0, L):
            A_previous = A #In this loop, A = previous_A
            W = parameters_model["l" + str(i+1)]["W"]
            b = parameters_model["l" + str(i+1)]["b"]
            A, cache = self.forward_one_layer(A_previous, W, b, 'relu')
            #cache_list.append(cache)
            cache_list["c" + str(i+1)] = cache

        #--- Compute the activation of the LAST layer of the model
        W = parameters_model["l" + str(L)]["W"]
        b = parameters_model["l" + str(L)]["b"]
        AL, cache = self.forward_one_layer(A, W, b, 'sigmoid')
        #cache_list.append(cache)
        cache_list["c" + str(L)] = cache

        return AL, cache_list

    def compute_cost(AL, Y):
        cost = nn_utils.cost(AL, Y)
        return cost

    def backward_one_layer(self, dA, cache, activation):
        """
            Backwarding only 1 layer back
            dA -> The derivative of the cost of 1 layer ahead
            Returns dA_previous, dW, db (dA_previous -> Derivative of the previous layer cost)
        """
        A_previous, W, b, Z = cache

        if activation == "relu":
            dZ = dA * nn_utils.relu_derivative(Z)
        elif activation == "sigmoid":
            dZ = dA * nn_utils.sigmoid_derivative(Z)

        m = A_previous.shape[1]
        dW = 1 / m * np.dot(dZ, A_previous.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=1)
        dA_previous = np.dot(W.T, dZ)

        return dA_previous, dW, db

    def backward_propagation(self, AL, Y, cache_list):

        L = len(cache_list)  #Num. of layers
        #Y = Y.reshape(AL.shape)

        # Calculating dAL, dA_previous, dW, db - Last Layer
        dAL = nn_utils.dAL(AL, Y)

        dA_previous, dw, db = self.backward_one_layer(dAL, cache_list["c" + str(L)], 'sigmoid')
        grads = {}
        grads["dA" + str(L-1)] = dA_previous
        grads["dW" + str(L)] = dw
        grads["db" + str(L)] = db

        # Calculating dAL, dA_previous, dW, db - Hidden Layers
        for l in reversed(range(L - 1)):
            current_cache = cache_list["c" + str(l)]
            dA = grads["dA" + str(l + 1)]
            dA_previous, dW, db = self.backward_one_layer(dA, current_cache, 'relu')
            grads["dA" + str(l)] = dA_previous
            grads["dW" + str(l + 1)] = dW
            grads["db" + str(l + 1)] = db

        return grads

    def update_parameters(self, parameters_model, grads, learning_rate):
        """
            Update model parameters (W, b)
        """
        L = len(parameters_model)  # Num. of layers in the model

        for l in range(0, L):
            W = parameters_model["W" + str(l + 1)]
            b = parameters_model["b" + str(l + 1)]

            parameters_model["W" + str(l + 1)] = W - learning_rate * grads["dW" + str(l + 1)]
            parameters_model["b" + str(l + 1)] = b - learning_rate * grads["db" + str(l + 1)]

        return parameters_model

    def model(self, X, Y, layers_dimensions, learning_rate, num_iterations, print_cost):

        np.random.seed(1)
        costs = []

        parameters_model = self.initialize_parameters_random(layers_dimensions)

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


DL = NN_Deep_Learning()
params = DL.initialize_parameters_random([10,4,4,1])

print(len(params))