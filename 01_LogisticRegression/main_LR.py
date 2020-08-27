import sys
sys.path.append("Modules")
sys.path.append("NeuralNetworks")
import nn_logistic_regression as nnLR
import pic2matrix

project_dir = r"Y:\Data_Science\Projects\fish_classifier\\"
file_model = project_dir + r"model_parameters"

#Dataset
pics_fish_dir = project_dir + r"dataset\Fish\\"
pics_noFish_dir = project_dir + r"dataset\NoFish\\"

file_type = "HDF5" # CSV or HDF5
dataset_amount = "min" #min = 10% of the folder | max = 100% of the folder

X_train_set, Y_train_set, X_test_set, Y_test_set = pic2matrix.generate_datasets_from_pic_folders_fish(False,
                                                                                                     project_dir,
                                                                                                     dataset_amount,
                                                                                                     file_type,
                                                                                                     pics_fish_dir,
                                                                                                     pics_noFish_dir)

#--- Building de model
LogRegression = nnLR.NN_Logistic_Regression()
num_iterations = 500
learning_rate = 0.003
print_cost = True
w, b = LogRegression.model(X_train_set, Y_train_set, X_test_set, Y_test_set, learning_rate, num_iterations, print_cost)

#--- Saving model parameters in a file
pic2matrix.model_parameters_to_file(w, b, file_model, file_type)


""" 
    Fish Classifier
        Iterations: 500
        Learning rate: 0.003
        Train examples: 80
        Test examples: 20
        Train accuracy: 98.75 %
        Test accuracy: 55.0 %
"""


"""
#--- Loading model parameters from a file
w, b = pic2matrix.file_to_model_parameters(file_model_parameters, file_type)

#--- Building dataset To Predict
dir_to_predict = pics_to_predict_fish_dir
pic2matrix.dataset_to_file(False, None, dir_to_predict, file_to_predict, file_type, "max")
X_to_predict = pic2matrix.file_to_matrix(file_to_predict, file_type)

#--- Predictions
Y_predictions = LogRegression.predict(w, b, X_to_predict, True)

"""