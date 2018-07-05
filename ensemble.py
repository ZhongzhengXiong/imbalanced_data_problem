from imblearn.ensemble import EasyEnsemble
from sklearn.metrics import recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier

from load_data import load_data
import logistic_regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from imblearn.ensemble import BalancedBaggingClassifier

from roc import calculate_roc, evaluate

if __name__ == '__main__':
    X_train, y_train = load_data('./dataset/car/car-vgood-5-fold/car-vgood-5-2tra.dat')
    X_test, y_test = load_data('./dataset/car/car-vgood-5-fold/car-vgood-5-2tst.dat')
    X_train, y_train = map(np.array, [X_train, y_train])
    X_test, y_test = map(np.array, [X_test, y_test])

    bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                    ratio='auto',
                                    replacement=False,
                                    random_state=0)

    bbc.fit(X_train, y_train)
    score = bbc.predict_proba(X_test)
    evaluate(y_test, score)

