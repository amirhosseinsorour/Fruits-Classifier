import numpy as np
import pickle


# k = Number of classes
# seed = np.random.randint(0, 100000)


def get_train_set(k):
    # loading training set features
    f = open("Datasets/Data/" + str(k) + "class/train_set_features.pkl", "rb")
    train_set_features2 = pickle.load(f)
    f.close()

    # reducing feature vector length

    features_STDs = np.std(a=train_set_features2, axis=0)
    train_set_features = train_set_features2[:, features_STDs > 52.3]  # 52.3 for k=4, 50.8 for k=6

    # changing the range of data between 0 and 1
    train_set_features = np.divide(train_set_features, train_set_features.max())

    # loading training set labels
    f = open("Datasets/Data/" + str(k) + "class/train_set_labels.pkl", "rb")
    train_set_labels = pickle.load(f)
    f.close()

    # preparing our training and test sets - joining datasets and lables
    train_set = []

    for i in range(len(train_set_features)):
        label = np.zeros(k)
        label[int(train_set_labels[i])] = 1
        label = label.reshape(k, 1)
        train_set.append((train_set_features[i].reshape(102, 1), label))

    # shuffle
    # np.random.seed(seed)
    # np.random.shuffle(train_set)

    return train_set


def get_test_set(k):
    # loading test set features
    f = open("Datasets/Data/" + str(k) + "class/test_set_features.pkl", "rb")
    test_set_features2 = pickle.load(f)
    f.close()

    # reducing feature vector length
    features_STDs = np.std(a=test_set_features2, axis=0)
    test_set_features = test_set_features2[:, features_STDs > 48]  # 48 for k=4, 46 for k = 6

    # changing the range of data between 0 and 1
    test_set_features = np.divide(test_set_features, test_set_features.max())

    # loading test set labels
    f = open("Datasets/Data/" + str(k) + "class/test_set_labels.pkl", "rb")
    test_set_labels = pickle.load(f)
    f.close()

    # preparing our training and test sets - joining datasets and lables
    test_set = []

    for i in range(len(test_set_features)):
        label = np.zeros(k)
        label[int(test_set_labels[i])] = 1
        label = label.reshape(k, 1)
        test_set.append((test_set_features[i].reshape(102, 1), label))

    # shuffle
    # np.random.seed(seed)
    # np.random.shuffle(test_set)

    return test_set

# print size
# print(len(train_set))  # 1962 for k = 4
# print(len(train_set))  # 2918 for k = 6
# print(len(test_set))   # 662 for k = 4
# print(len(test_set))   # 984 for k = 6
