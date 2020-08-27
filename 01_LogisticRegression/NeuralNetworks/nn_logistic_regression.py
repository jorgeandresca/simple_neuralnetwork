import numpy as np
import sys
sys.path.append("../Modules")
import nn_utils
import matplotlib.pyplot as plt

class NN_Logistic_Regression:
    """
        This class contains the following functions:
            1) initialize_parameters_zeros
            2) propagate
            3) optimization
            4) predict
            5) model
    """

    def initialize_parameters_zeros(self, num_features):
        """
            Initialize W and b with zeros
            w.shape = (num_features, 1)
        """
        w = np.zeros((num_features, 1))
        b = 0
        return w, b

    def propagate(self, w, b, X, Y):
        """
            Forward the dataset
            X.shape: (numFeatures, m)
            Y.shape: (1, num examples)
        """
        m = X.shape[1]

        #--- Forward Propagation
        Z = np.dot(w.T, X) + b
        A = nn_utils.sigmoid(Z) #computing the Activation

        #--- Cost calculation
        cost = nn_utils.cost(A, Y)

        #--- Backward Propagation
        dZ = A - Y
        dw = np.dot(X, (dZ).T) / m
        db = np.sum(dZ) / m

        grads = { "dw": dw,
                  "db": db }

        return grads, cost

    def optimization(self, w, b, X, Y, num_iterations, learning_rate, print_cost):
        """
            Optimization of the model
        """
        costs = [] #we will store the cost after some iterations to plot later
        for i in range(num_iterations):

            grads, cost = self.propagate(w, b, X, Y) #calculating gradients for this iteration
            dw = grads["dw"]
            db = grads["db"]

            #--- Gradient Descent
            w = w - learning_rate * dw
            b = b - learning_rate * db


            #--- Storing the cost after some iterations
            if i % 100 == 0:
                costs.append(cost)

            if print_cost and i % 100 == 0:
                print("Cost iteration " + str(i) + ": " + str(cost))

        parameters = {"w": w,
                       "b": b }
        return parameters, costs

    def predict(self, w, b, X, show_image):
        """
            Predicting a dataset using a model
            X.shape = (features, m)
        """

        #--- #The predictions - Computing the Activation of all X
        Z = np.dot(w.T, X) + b
        A = nn_utils.sigmoid(Z)

        #--- Store each prediction in the vector
        m = X.shape[1]
        Y_predictions = np.zeros((1, m))
        zero_count = 0
        one_count = 0
        for i in range(m):
            prediction = 1 if A[0, i] > 0.5 else 0
            Y_predictions[0, i] = prediction
            if(prediction == 1):
                one_count += 1
            elif(prediction == 0):
                zero_count += 1


            if(prediction == 1 and show_image): #Showing in the screen certain pictures
                plt.imshow(X[:, i].reshape((64, 64, 3)))
                #plt.show()


        print("1s: " + str(one_count))
        print("0s: " + str(zero_count))

        return Y_predictions

    def model(self, X_train, Y_train, X_test, Y_test, learning_rate, num_iterations, print_cost):
        """
            Building the model
                1) Initialize parameters
                2) Propagation
                3) Optimization
        """
        num_features = X_train.shape[0]

        #--- Initializing parameters
        w, b = self.initialize_parameters_zeros(num_features)

        #--- Optimization of parameters
        parameters, costs = self.optimization(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

        #--- Parameters of the model
        w = parameters["w"]
        b = parameters["b"]

        #--- Evaluate the model
        Y_prediction_train = self.predict(w, b, X_train, False)
        Y_prediction_test = self.predict(w, b, X_test, False)

        difference_train = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
        difference_test = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100

        print("")
        print("Iterations: " + str(num_iterations))
        print("Learning rate: " + str(learning_rate))
        print("Train examples: " + str(X_train.shape[1]))
        print("Test examples: " + str(X_test.shape[1]))
        print("Train accuracy: " + str(difference_train) + " %")
        print("Test accuracy: " + str(difference_test) + " %")

        return w, b