import sys
sys.path.append("Modules")
import nn_deep_learning
import nn_utils

# --------------------- Dataset ---------------------------

x_train_set, y_train_set, x_dev_set, y_dev_set, classes = nn_utils.load_data()

x_train_set_flatten = x_train_set.reshape(x_train_set.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
x_dev_set_flatten= x_dev_set.reshape(x_dev_set.shape[0], -1).T

x_train_set_norm = x_train_set_flatten/255
x_dev_set_norm = x_dev_set_flatten/255


# --------------------- Model ---------------------------

model_dimensions = [x_train_set_flatten.shape[0], 20, 7, 5, 1]
learning_rate = 0.0075
iterations = 300

parameters = nn_deep_learning.L_layer_model(x_train_set_norm, y_train_set, model_dimensions, learning_rate=learning_rate,num_iterations = iterations, print_cost = True)
