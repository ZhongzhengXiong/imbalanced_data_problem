import numpy as np
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import CondensedNearestNeighbour
from load_data import load_data
import logistic_regression
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # load data
    X_train, y_train = load_data('./dataset/car/car-vgood-5-fold/car-vgood-5-1tra.dat')
    X_test, y_test = load_data('./dataset/car/car-vgood-5-fold/car-vgood-5-1tst.dat')
    X_train, y_train = map(np.array, [X_train, y_train])
    X_test, y_test = map(np.array, [X_test, y_test])




    test_loader = logistic_regression.data_loader(X_test, y_test)

    # baseline
    train_loader = logistic_regression.data_loader(X_train, y_train)
    model = logistic_regression.Module(X_train.shape[1], 2)
    logistic_regression.train_and_test(model, train_loader, test_loader)

    # generation
    model = logistic_regression.Module(X_train.shape[1], 2)
    cc = ClusterCentroids(random_state=0)
    X_res, y_res = cc.fit_sample(X_train, y_train)
    print(X_train.shape)
    print(X_res.shape, y_res.shape)
    print(np.sum(y_res))
    train_loader = logistic_regression.data_loader(X_res, y_res)
    logistic_regression.train_and_test(model, train_loader, test_loader)

    # selection
    # random under-sample
    random_samplers = {
        'random under-sample': RandomUnderSampler(random_state=0),
        'random under-sample, set replacement': RandomUnderSampler(random_state=0, replacement=True)
    }
    for key in random_samplers:
        print("######################## %s ########################" % (key))
        rus = random_samplers.get(key)
        model = logistic_regression.Module(X_train.shape[1], 2)
        X_res, y_res = rus.fit_sample(X_train, y_train)
        print(X_train.shape)
        print(X_res.shape, y_res.shape)
        print(np.sum(y_res))
        train_loader = logistic_regression.data_loader(X_res, y_res)
        logistic_regression.train_and_test(model, train_loader, test_loader)

    # near miss
    near_miss_models = {
        'near miss1': NearMiss(random_state=0, version=1),
        'near miss2': NearMiss(random_state=0, version=2),
        'near miss3': NearMiss(random_state=0, version=3)
    }
    for key in near_miss_models:
        print("######################## %s ########################" % (key))
        nm = near_miss_models.get(key)
        model = logistic_regression.Module(X_train.shape[1], 2)
        X_res, y_res = rus.fit_sample(X_train, y_train)
        print(X_train.shape)
        print(X_res.shape, y_res.shape)
        print(np.sum(y_res))
        train_loader = logistic_regression.data_loader(X_res, y_res)
        logistic_regression.train_and_test(model, train_loader, test_loader)

    # Tomek's links

    # Edited data set using nearest neighbours
    print("######################## ENN ########################")
    enn = EditedNearestNeighbours(random_state=0)
    X_res, y_res = enn.fit_sample(X_train, y_train)
    print(X_train.shape)
    print(X_res.shape)
    print(np.sum(y_res))
    model = logistic_regression.Module(X_train.shape[1], 2)
    train_loader = logistic_regression.data_loader(X_res, y_res)
    logistic_regression.train_and_test(model, train_loader, test_loader)


    # Condensed nearest neighbors and derived algorithms
    print("######################## CNN ########################")
    cnn = CondensedNearestNeighbour(random_state=0)
    X_res, y_res = cnn.fit_sample(X_train, y_train)
    print(X_train.shape)
    print(X_res.shape)
    print(np.sum(y_res))
    model = logistic_regression.Module(X_train.shape[1], 2)
    train_loader = logistic_regression.data_loader(X_res, y_res)
    logistic_regression.train_and_test(model, train_loader, test_loader)


    pass
