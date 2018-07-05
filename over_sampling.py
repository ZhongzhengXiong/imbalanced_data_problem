import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.svm import SVC

from load_data import load_data
import logistic_regression
from roc import calculate_roc, evaluate


def train_and_test(module, train_loader, test_loader):
    logistic_regression.train(module, train_loader)
    scores = logistic_regression.test(module, test_loader)
    calculate_roc(y_test, scores)


if __name__ == '__main__':
    X_train, y_train = load_data('./dataset/car/car-vgood-5-fold/car-vgood-5-2tra.dat')
    X_test, y_test = load_data('./dataset/car/car-vgood-5-fold/car-vgood-5-2tst.dat')
    X_train, y_train = map(np.array, [X_train, y_train])
    X_test, y_test = map(np.array, [X_test, y_test])

    clf = SVC(probability=True)
    clf.fit(X_train, y_train)
    # module = logistic_regression.Module(X_train.shape[1], 2)
    # train_loader = logistic_regression.data_loader(X_train, y_train)
    # test_loader = logistic_regression.data_loader(X_test, y_test, shuffle=False)

    # Baseline
    print("################## Baseline ######################")
    #train_and_test(module, train_loader, test_loader)
    score = clf.predict_proba(X_test)
    evaluate(y_test, score)

    # Random oversampling
    print("################## Random Sampling ######################")
    X_res, y_res = RandomOverSampler(random_state=42).fit_sample(X_train, y_train)
    # module = logistic_regression.Module(X_train.shape[1], 2)
    # train_loader = logistic_regression.data_loader(X_res, y_res)

    # train_and_test(module, train_loader, test_loader)
    clf = SVC(probability=True)
    clf.fit(X_res, y_res)
    score = clf.predict_proba(X_test)
    evaluate(y_test, score)

    # SMOTE oversampling
    print("################## SMOTE Sampling ######################")
    X_res, y_res = SMOTE().fit_sample(X_train, y_train)
    # module = logistic_regression.Module(X_train.shape[1], 2)
    # train_loader = logistic_regression.data_loader(X_res, y_res)
    # train_and_test(module, train_loader, test_loader)
    clf = SVC(probability=True)
    clf.fit(X_res, y_res)
    score = clf.predict_proba(X_test)
    evaluate(y_test, score)
    # ADASYN oversampling
    print("################## ADASYN Sampling ######################")
    X_res, y_res = ADASYN().fit_sample(X_train, y_train)
    # module = logistic_regression.Module(X_train.shape[1], 2)
    # train_loader = logistic_regression.data_loader(X_res, y_res)
    # train_and_test(module, train_loader, test_loader)
    clf = SVC(probability=True)
    clf.fit(X_res, y_res)
    score = clf.predict_proba(X_test)
    evaluate(y_test, score)