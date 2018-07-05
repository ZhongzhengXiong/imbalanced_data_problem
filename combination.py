from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from sklearn.svm import SVC

from load_data import load_data
import logistic_regression
import numpy as np

from roc import evaluate

if __name__ == '__main__':
    # load data
    X_train, y_train = load_data('./dataset/car/car-vgood-5-fold/car-vgood-5-1tra.dat')
    X_test, y_test = load_data('./dataset/car/car-vgood-5-fold/car-vgood-5-1tst.dat')
    X_train, y_train = map(np.array, [X_train, y_train])
    X_test, y_test = map(np.array, [X_test, y_test])

    test_loader = logistic_regression.data_loader(X_test, y_test)

    # baseline
    print("######################## baseline ########################")

    clf = SVC(probability=True)
    clf.fit(X_train, y_train)
    score = clf.predict_proba(X_test)
    evaluate(y_test, score)

    # smote enn
    print("######################## smote-enn ########################")

    smote_enn = SMOTEENN(random_state=0)
    X_res, y_res = smote_enn.fit_sample(X_train, y_train)
    clf = SVC(probability=True)
    clf.fit(X_res, y_res)
    score = clf.predict_proba(X_test)
    evaluate(y_test, score)
    # smote tomek
    print("######################## smote-tomek ########################")
    smote_tomek = SMOTETomek(random_state=0)
    clf = SVC(probability=True)
    clf.fit(X_res, y_res)
    score = clf.predict_proba(X_test)
    evaluate(y_test, score)