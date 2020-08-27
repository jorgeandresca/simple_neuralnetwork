import sys
sys.path.append("Modules")
sys.path.append("NeuralNetworks")
import nn_logistic_regression as nnLR
import pic2matrix


""" Coursera DeepLearning.ai dataset
        Dataset stored in HDF5 files
        Model results (same as in the course):
            w[0,0]: 0.00773216117723552
            b: -0.017480313732343933
            Train accuracy: 97.12918660287082 %
            Test accuracy: 70.0 %
"""

num_iterations = 2000
learning_rate = 0.003
print_cost = True

h5_trainset_dir = r"F:\Data_Science\Projects\fish_classifier\courseraDS\train_catvnoncat.h5"
h5_testset_dir = r"F:\Data_Science\Projects\fish_classifier\courseraDS\test_catvnoncat.h5"
train_set_x, train_set_y, test_set_x, test_set_y = pic2matrix.load_coursera_dataset(h5_trainset_dir, h5_testset_dir)
nnLR.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations, learning_rate, print_cost)
