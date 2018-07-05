from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from load_data import load_data
import logistic_regression
import numpy as np

if __name__ == '__main__':
    # load data
    X_train, y_train = load_data('./dataset/car/car-vgood-5-fold/car-vgood-5-1tra.dat')
    X_test, y_test = load_data('./dataset/car/car-vgood-5-fold/car-vgood-5-1tst.dat')
    X_train, y_train = map(np.array, [X_train, y_train])
    X_test, y_test = map(np.array, [X_test, y_test])

    test_loader = logistic_regression.data_loader(X_test, y_test)

    # baseline
    print("######################## baseline ########################")

    train_loader = logistic_regression.data_loader(X_train, y_train)
    model = logistic_regression.Module(X_train.shape[1], 2)
    logistic_regression.train_and_test(model, train_loader, test_loader)

    # smote enn
    print("######################## smote-enn ########################")

    smote_enn = SMOTEENN(random_state=0)
    X_res, y_res = smote_enn.fit_sample(X_train, y_train)
    train_loader = logistic_regression.data_loader(X_res, y_res)
    logistic_regression.train_and_test(model, train_loader, test_loader)

    # smote tomek
    print("######################## smote-tomek ########################")
    smote_tomek = SMOTETomek(random_state=0)
    train_loader = logistic_regression.data_loader(X_res, y_res)
    logistic_regression.train_and_test(model, train_loader, test_loader)