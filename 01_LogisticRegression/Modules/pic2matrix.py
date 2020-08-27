from PIL import Image #to manipulate the images
import numpy as np
import os #to open the directories
import h5py



def dataset_to_file(includeTag, tag, pics_directory, to_save_directory_filename, file_type, dataset_amount):
    """ This function receives a Directory and returns a CSV file of shape (width*height*3 + 1, m)
            width and height are being resized to 64px each.
            width*height*3 = features
            1 = output
            m = number of examples
    """

    m = len(os.listdir(pics_directory)) #m = num of samples
    if(dataset_amount.lower() == "min"):
        m = int(m / 10)
    picsMatrix = np.zeros((m, 64, 64, 3)).astype(int) #create main matrix (m, height, width, 3)

    #--- Resizing all pictures and building the first Matrix
    i = 0
    for filename in os.listdir(pics_directory):
        img = Image.open(pics_directory + filename).convert('RGB') #open a file (assuming its a picture) - convert(RGB) to only work with RGB and not RGB-1
        img = img.resize((64, 64)) #resize all pictures' (width, height)
        picsMatrix[i] = np.array(img) #np.array(img).shape = (64, 64, 3)
        i += 1
        if (i == m - 1):  # in case dataset_amount = min, so this loop will get only 10% of the files
            break
    #--- Flattening the Matrix to (w*h*3, m)
    flattenMatrix = picsMatrix.reshape(m, -1).T #The result is a matrix of shape (12288, m)

    #--- Normalizing data
    flattenMatrix = flattenMatrix / 255

    #--- Adding the tag to the final_matrix - first row
    if(includeTag):
        tagVector = np.zeros((1, flattenMatrix.shape[1])).astype(int)  # This vector will store the tags and will be appended to the main matrix
        tagVector[0] = tag
        flattenMatrix = np.concatenate((tagVector, flattenMatrix), axis=0) #first row contains the tag (the output)

    #--- Saving the dataset in a file
    if(file_type.lower() == "csv"):
        np.savetxt(to_save_directory_filename + ".csv", flattenMatrix, delimiter=",")
    elif(file_type.lower() == "hdf5"):
        hf = h5py.File(to_save_directory_filename + ".h5", 'w')
        hf.create_dataset('dataset_1', data=flattenMatrix)
        hf.close()

    return flattenMatrix

def model_parameters_to_file(w, b, to_save_directory_filename, file_type):
    b_vector = np.zeros((1,1))
    b_vector[0,0] = b

    final_matrix = np.concatenate((b_vector, w), axis=0)

    if(file_type.lower() == "csv"):
        np.savetxt(to_save_directory_filename + ".csv", final_matrix, delimiter=",")
    elif(file_type.lower() == "hdf5"):
        hf = h5py.File(to_save_directory_filename + ".h5", 'w')
        hf.create_dataset('dataset_1', data=final_matrix)
        hf.close()

def file_to_model_parameters(directory_filename, file_type):
    """ Bulks model parameters from a file into w and b variables """
    if(file_type.lower() == "csv"):
        matrix = np.loadtxt(open(directory_filename + ".csv") , delimiter=",")
    elif(file_type.lower() == "hdf5"):
        hf = h5py.File(directory_filename + ".h5", 'r')
        matrix = np.array(hf.get('dataset_1'))
        hf.close()

    matrix = matrix.reshape(matrix.shape[0], 1)

    b = matrix[0, 0]
    w = matrix[1:, 0]
    w = w.reshape(w.shape[0], 1)

    return w, b

def file_to_matrix(directory_filename, file_type):
    """ Bulks data from a file into a Matrix """
    if(file_type.lower() == "csv"):
        matrix = np.loadtxt(open(directory_filename + ".csv") , delimiter=",")
    elif(file_type.lower() == "hdf5"):
        hf = h5py.File(directory_filename + ".h5", 'r')
        matrix = np.array(hf.get('dataset_1'))
        hf.close()

    return matrix

def generate_datasets_from_matrices(matrices_array):
    """
        finalDataset
            1) Concatenating all matrices
            2) Shuffle all dataset
            3) Substract the Tag vector from the dataset
            3) Splitting between Training set and Test set (80/20)
    """
    #--- Concatenating all matrices (num of examples)
    dataset_XY = matrices_array[0]
    for i in range(1, len(matrices_array)):
        dataset_XY = np.concatenate((dataset_XY, matrices_array[i]), axis=1)


    #--- Shuffling the columns (examples) of the dataset
    np.random.shuffle(dataset_XY.T)

    #--- Substracting the Tag vector
    Y_dataset = dataset_XY[0]
    X_dataset = dataset_XY[1:]

    #--- Splitting training set and test set (80/20)
    m = dataset_XY.shape[1]
    eightyPercent = int(80 * m / 100)

    X_dataset = X_dataset.T
    Y_dataset = Y_dataset.T

    X_train_set = X_dataset[:eightyPercent].T
    Y_train_set = Y_dataset[:eightyPercent].T.reshape(1, eightyPercent)

    X_test_set = X_dataset[eightyPercent:].T
    Y_test_set = Y_dataset[eightyPercent:].T.reshape(1, m - eightyPercent)


    #--- Some validations
    assert(X_train_set.shape == (matrices_array[0].shape[0] - 1, eightyPercent))
    assert(Y_train_set.shape == (1, eightyPercent))
    assert(X_test_set.shape == (matrices_array[0].shape[0] - 1, m - eightyPercent))
    assert(Y_test_set.shape == (1, m - eightyPercent))

    return X_train_set, Y_train_set, X_test_set, Y_test_set

def generate_datasets_from_pic_folders_fish(create_dataset_files,
                                            project_dir,
                                            dataset_amount,
                                            file_type,
                                            pics_fish_dir,
                                            pics_noFish_dir):
    file_dataset_fish = project_dir + "dataset_fish"
    file_dataset_noFish = project_dir + "dataset_noFish"
    file_model_parameters = project_dir + "model_parameters"

    if (dataset_amount.lower() == "min"):
        file_dataset_fish += "_min"
        file_dataset_noFish += "_min"

    # --- Creating dataset files
    dataset_XY_fish = dataset_to_file(True, 1, pics_fish_dir, file_dataset_fish, file_type, dataset_amount)
    dataset_XY_noFish = dataset_to_file(True, 0, pics_noFish_dir, file_dataset_noFish, file_type, dataset_amount)


    # --- Generating the final dataset
    matrices_array = [dataset_XY_fish, dataset_XY_noFish]
    return generate_datasets_from_matrices(matrices_array)

def load_coursera_dataset(h5_trainset_dir, h5_testset_dir):
    """ This function works only with DeepLearning.ai week 2 dataset of Course 1 """
    train_dataset = h5py.File(h5_trainset_dir, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File(h5_testset_dir, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    return train_set_x, train_set_y, test_set_x, test_set_y