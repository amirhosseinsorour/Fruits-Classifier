import Loading_Dataset
import numpy as np
from matplotlib import pyplot as plt
import time


def train_data(m=1962, n=102, k=4):
    """ First m data of training set, n is num of features anf k is num of classes """
    X = np.zeros([m, n])
    Y = np.zeros([m, k])

    train_set = Loading_Dataset.get_train_set(k)
    # X, Y = shuffle(X, Y)

    for i in range(m):
        t = train_set[i]
        X[i] = t[0].reshape(n)
        Y[i] = t[1].reshape(k)

    return X, Y


def test_data(m=662, n=102, k=4):
    """ First m data of testing set, n is num of features anf k is num of classes """
    X = np.zeros([m, n])
    Y = np.zeros([m, k])

    test_set = Loading_Dataset.get_test_set(k)

    for i in range(m):
        t = test_set[i]
        X[i] = t[0].reshape(n)
        Y[i] = t[1].reshape(k)

    return X, Y


def shuffle(X, Y):
    """ Shuffling two numpy arrays simultaneously """
    seed = np.random.randint(0, 100000)

    np.random.seed(seed)
    np.random.shuffle(X)

    np.random.seed(seed)
    np.random.shuffle(Y)

    return X, Y


def initialize_parameters(L, layers_dim):
    """ Initialize weights of network randomly, and biases of network to zero """

    parameters = {}  # Dictionary containing weights and biases of the network
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dim[l], layers_dim[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dim[l], 1))

    return parameters


def initialize_gradients(L, layers_dim):
    """ Initialize gradients of weights and biases to zero """

    grads = {}  # Dictionary containing gradient of weights and biases of the network
    for l in range(1, L):
        grads['dW' + str(l)] = np.zeros((layers_dim[l], layers_dim[l - 1]))
        grads['db' + str(l)] = np.zeros((layers_dim[l], 1))

    return grads


class Activation:
    def __init__(self, type):
        self.type = type

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def d_sigmoid(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def tanh(self, z):
        return np.tanh(z)

    def d_tanh(self, z):
        return 1 - self.tanh(z) ** 2

    def ReLU(self, z):
        return np.max(0, z)

    def d_ReLU(self, z):
        return (z > 0).astype(float)

    def compute(self, Z):
        if self.type == "sigmoid":
            A = self.sigmoid(Z)
        elif self.type == "tanh":
            A = self.tanh(Z)
        elif self.type == "ReLU":
            A = self.ReLU(Z)
        else:
            A = Z

        return A

    def derivate(self, Z):
        if self.type == "sigmoid":
            dZ = self.d_sigmoid(Z)
        elif self.type == "tanh":
            dZ = self.d_tanh(Z)
        elif self.type == "ReLU":
            dZ = self.d_ReLU(Z)
        else:
            dZ = Z

        return dZ


def linear_activation_forward(A, W, b, activation_type="sigmoid"):
    """ Compute activation of next layer. Default activation is sigmoid """
    activation = Activation(activation_type)

    Z_next = A @ W.T + b.T
    A_next = activation.compute(Z_next)

    return A_next, Z_next


def feed_forward(X, parameters, L):
    """ Forward Propagation """

    cache = {}  # Dictionary containing A and Z in all layers for computing backward pass
    A = X  # The input layer
    cache['A0'] = X
    for l in range(1, L):
        # Retrieve parameters of l-th layer
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]

        # Compute next layer
        A_next, Z_next = linear_activation_forward(A, W, b)

        # Saving A and Z of l-th layer
        cache['A' + str(l)] = A_next
        cache['Z' + str(l)] = Z_next

        # Go to next layer
        A = A_next

    AL = A  # The output layer
    return AL, cache


def compute_accuracy(AL, Y):
    """ Accuracy of model: Ratio of the correct detected images to the total number of images """
    true_detections = 0  # Number of images the network correctly detected
    m = Y.shape[0]  # Number of all images
    for i in range(m):
        aL = AL[i]
        y = Y[i]
        if not aL.any():
            continue
        if np.max(aL) == np.dot(y, aL):
            true_detections += 1

    return true_detections / m


def soft_max(AL):
    """ Normalize the output of network using soft-max """
    sum = np.sum(np.exp(AL), axis=1).reshape(AL.shape[0], 1)
    return np.exp(AL) / sum


def compute_cost(AL, Y, m):
    return np.sum((AL - Y) ** 2) / m


def plot_sgd_costs(epochs_num, costs, title):
    epochs = np.linspace(0, epochs_num, num=epochs_num + 1)
    plt.plot(epochs, costs, 'g')
    plt.xticks(epochs)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    # plt.show()


def iterative_back_propagation(y, cache, grads, parameters, L):
    """ Backward Propagation, Each of Training data is processed separately """
    aL = cache['A' + str(L - 1)]
    grads['dA' + str(L - 1)] = 2 * (aL - y)

    for l in reversed(range(1, L)):
        dW = grads['dW' + str(l)].T
        W = parameters['W' + str(l)].T
        da_next = grads['dA' + str(l)][0]
        dz_next = Activation("sigmoid").derivate(cache['Z' + str(l)])[0]
        a_prev = cache['A' + str(l - 1)][0]

        J, K = dW.shape
        for j in range(J):
            for k in range(K):
                dW[j][k] = da_next[k] * dz_next[k] * a_prev[j]
        grads['dW' + str(l)] += dW.T

        db = grads['db' + str(l)]
        db = (da_next * dz_next).reshape(db.shape)
        grads['db' + str(l)] += db

        if l > 1:
            da_prev = np.zeros(cache['A' + str(l - 1)].shape)
            for j in range(J):
                da_prev[0][j] = np.sum(da_next * dz_next * W[j])
            grads['dA' + str(l - 1)] = da_prev

    return grads


def iterative_sgd(X, Y, parameters, m, L, layers_dim, learning_rate, epochs_num, batch_size):
    """ Stochastic Gradient Descent, using loops """
    costs_per_epoch = [compute_cost(feed_forward(X, parameters, L)[0], Y, m)]

    for epoch in range(epochs_num):
        # print("Epoch : ", epoch)

        X, Y = shuffle(X, Y)
        batch_index = 0

        while batch_index < m:
            batch = zip(X[batch_index:batch_index + batch_size],
                        Y[batch_index:batch_index + batch_size])

            grads = initialize_gradients(L, layers_dim)

            for x, y in batch:
                x = x.reshape(1, x.shape[0])
                y = y.reshape(1, y.shape[0])

                # Forward and Backward Propagation
                aL, cache = feed_forward(x, parameters, L)
                grads = iterative_back_propagation(y, cache, grads, parameters, L)

            # Updating parameters
            for l in range(1, L):
                parameters['W' + str(l)] -= (learning_rate / batch_size) * grads['dW' + str(l)]
                parameters['b' + str(l)] -= (learning_rate / batch_size) * grads['db' + str(l)]

            batch_index += batch_size

        costs_per_epoch.append(compute_cost(feed_forward(X, parameters, L)[0], Y, m))

    plot_sgd_costs(epochs_num, costs_per_epoch, title='Iterative SGD')
    return parameters


def vectorized_back_propagation(Y, cache, grads, parameters, L):
    """ Backward Propagation, All of training data is processed as one matrix """
    AL = cache['A' + str(L - 1)]
    grads['dA' + str(L - 1)] = 2 * (AL - Y)

    for l in reversed(range(1, L)):
        dW = grads['dW' + str(l)].T
        W = parameters['W' + str(l)]
        dA_next = grads['dA' + str(l)]
        dZ_next = Activation("sigmoid").derivate(cache['Z' + str(l)])
        A_prev = cache['A' + str(l - 1)]

        dW = A_prev.T @ (dA_next * dZ_next)
        grads['dW' + str(l)] = dW.T

        db = grads['db' + str(l)]
        db = np.sum(dA_next * dZ_next, axis=0).reshape(db.shape)
        grads['db' + str(l)] = db

        if l > 1:
            dA_prev = (dA_next * dZ_next) @ W
            grads['dA' + str(l - 1)] = dA_prev

    return grads


def vectorized_sgd(X, Y, parameters, m, L, layers_dim, learning_rate, epochs_num, batch_size):
    """ Stochastic Gradient Descent, vectorized version(without using loops) """
    costs_per_epoch = [compute_cost(feed_forward(X, parameters, L)[0], Y, m)]

    for epoch in range(epochs_num):
        # print("Epoch : ", epoch)

        X, Y = shuffle(X, Y)
        batch_index = 0

        while batch_index < m:
            batch_X = X[batch_index:batch_index + batch_size]
            batch_Y = Y[batch_index:batch_index + batch_size]

            grads = initialize_gradients(L, layers_dim)

            # Forward and Backward Propagation
            AL, cache = feed_forward(batch_X, parameters, L)
            grads = vectorized_back_propagation(batch_Y, cache, grads, parameters, L)

            # Updating parameters
            for l in range(1, L):
                parameters['W' + str(l)] -= (learning_rate / batch_size) * grads['dW' + str(l)]
                parameters['b' + str(l)] -= (learning_rate / batch_size) * grads['db' + str(l)]

            batch_index += batch_size

        costs_per_epoch.append(compute_cost(feed_forward(X, parameters, L)[0], Y, m))
        # print(costs_per_epoch[-1])

    # plot_sgd_costs(epochs_num, costs_per_epoch, title='Costs without using softmax')
    return parameters


if __name__ == '__main__':
    """ Please un-comment each part you want to execute """
    # ====================================================== #

    """ Part 1 : Define Number of training data, features, classes and hidden layers """
    m = 200  # Number of training data
    n = 102  # Number of features
    k = 4  # Number of classes
    layers_dim = [n, 150, 60, k]  # Layers dimensions
    L = len(layers_dim)  # Number of layers in the network
    # ============================================================================================================ #

    """ Part 2 : Initialize training data matrix and parameters of network """
    X, Y = train_data(m, n, k)
    parameters = initialize_parameters(L, layers_dim)
    # ============================================================================================================ #

    """ Part 3 :Forward Propagation and Report accuracy (Don't expect much accuracy in this part!) """
    AL = feed_forward(X, parameters, L)[0]
    accuracy = compute_accuracy(AL, Y)
    print('========================================================')
    print(">> Accuracy of model after random initialization: ", end='')
    print(accuracy * 100, '%')
    print('--------------------------------------------------------')
    # ============================================================================================================ #

    """ Part 4 : Iterative sgd, This part may be slow """
    start_time = time.time()
    parameters = iterative_sgd(X, Y, parameters, m, L, layers_dim, learning_rate=1, epochs_num=5, batch_size=10)

    AL = feed_forward(X, parameters, L)[0]
    accuracy = compute_accuracy(AL, Y)

    print('===============================================================')
    print(">> Accuracy of model with iterative sgd for epochs=5 : ", end='')
    print(accuracy * 100, '%')
    print(">> Learning Time : %s seconds" % (time.time() - start_time))
    print('---------------------------------------------------------------')
    # ============================================================================================================ #

    """ Part 5 : Vectorized sgd """
    start_time = time.time()
    parameters = vectorized_sgd(X, Y, parameters, m, L, layers_dim, learning_rate=1, epochs_num=20, batch_size=10)

    AL = feed_forward(X, parameters, L)[0]
    accuracy = compute_accuracy(AL, Y)

    print('===============================================================')
    print(">> Accuracy of model with vectorized sgd for epochs=20 : ", end='')
    print(accuracy * 100, '%')
    print(">> Learning Time : %s seconds" % (time.time() - start_time))
    print('---------------------------------------------------------------')
    # ============================================================================================================ #

    """ Part 6 : Run the code for 10 times and report the average result """
    average_AL = np.zeros(Y.shape)
    for r in range(10):
        parameters = vectorized_sgd(X, Y, parameters, m, L, layers_dim, learning_rate=1, epochs_num=20, batch_size=10)
        X, Y = train_data(m, n, k)
        AL = feed_forward(X, parameters, L)[0]
        average_AL += AL
        parameters = initialize_parameters(L, layers_dim)

    accuracy = compute_accuracy(average_AL, Y)
    print('===============================================================')
    print(">> Accuracy of model after 10 iterations of vectorized sgd : ", end='')
    print(accuracy * 100, '%')
    print('---------------------------------------------------------------')
    # ============================================================================================================ #

    """ 
        Part 7 : 
        Train Model with all of training data and report accuracy on train and test data
        and take the average of 10 runs as the result
    """
    X_train, Y_train = train_data()
    X_test, Y_test = test_data()

    average_AL_train = np.zeros(Y_train.shape)
    average_AL_test = np.zeros(Y_test.shape)

    for r in range(10):
        # Learning
        parameters = vectorized_sgd(X_train, Y_train, parameters, m, L, layers_dim,
                                    learning_rate=1, epochs_num=10, batch_size=10)

        # Shuffle data back to normal order
        X_train, Y_train = train_data()

        # Compute output of train data
        AL_train = feed_forward(X_train, parameters, L)[0]
        average_AL_train += AL_train

        # Compute output of test data
        AL_test = feed_forward(X_test, parameters, L)[0]
        average_AL_test += AL_test

        # Resetting parameters
        parameters = initialize_parameters(L, layers_dim)

    train_accuracy = compute_accuracy(average_AL_train, Y_train)
    test_accuracy = compute_accuracy(average_AL_test, Y_test)

    print('===============================================================')
    print(">> Accuracy of model after 10 iterations of code:")
    print(">> For train data : ", train_accuracy * 100, '%')
    print(">> For test data : ", test_accuracy * 100, '%')
    print('---------------------------------------------------------------')
    # ============================================================================================================ #

    """
        Part 8 (Bonus) : Build Model with more classes 
        Don't forget to change Loading_Dataset parameters first and comment out Part 2
    """
    m_train = 2918  # Number of train data
    m_test = 984  # Number of test data
    n = 102  # Number of features
    k = 6  # Number of classes
    layers_dim = [n, 150, 200, 80, 20, k]  # Layers dimensions
    L = len(layers_dim)  # Number of layers in the network

    X_train, Y_train = train_data(m_train, n, k)
    X_test, Y_test = test_data(m_test, n, k)

    parameters = initialize_parameters(L, layers_dim)
    parameters = vectorized_sgd(X_train, Y_train, parameters, m_train, L, layers_dim,
                                learning_rate=0.5, epochs_num=15, batch_size=20)

    train_accuracy = compute_accuracy(feed_forward(X_train, parameters, L)[0], Y_train)
    test_accuracy = compute_accuracy(feed_forward(X_test, parameters, L)[0], Y_test)

    print('===============================================================')
    print(">> Accuracy of model with 6 classes : ")
    print(">> For train data : ", train_accuracy * 100, '%')
    print(">> For test data : ", test_accuracy * 100, '%')
    print('---------------------------------------------------------------')
    # ============================================================================================================ #

    """ Part 9 (Bonus) : Use Softmax in the last layer of network """
    parameters = vectorized_sgd(X, Y, parameters, m, L, layers_dim, learning_rate=1, epochs_num=10, batch_size=10)

    AL = feed_forward(X, parameters, L)[0]
    soft_Al = soft_max(AL)

    accuracy = compute_accuracy(AL, Y)
    soft_accuracy = compute_accuracy(soft_Al, Y)

    cost = compute_cost(AL, Y, m)
    soft_cost = compute_cost(soft_Al, Y, m)

    print('===============================================================')
    print(">> Accuracy of model, without using softmax for the output layer : ", accuracy * 100, '%')
    print(">> Accuracy of model, after using softmax for the output layer : ", soft_accuracy * 100, '%')
    print('---------------------------------------------------------------')
    # ============================================================================================================ #
