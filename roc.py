from sklearn.metrics import roc_curve, recall_score, precision_score, accuracy_score, f1_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def evaluate(y_test, score):
    index = np.argmax(score, axis=1)
    recall = recall_score(y_test, index)
    precision = precision_score(y_test, index)
    accuracy = accuracy_score(y_test, index)
    f_mean = f1_score(y_test, index)

    print("recall:%.4f, precision:%.4f, accuracy:%.4f, F_mean:%.4f" %
          (recall, precision, accuracy, f_mean))
    calculate_roc(y_test, score[:, 1])
    pass


def calculate_roc(targets, scores):
    fpr, tpr, thresholds = roc_curve(targets, scores)
    plt.plot(fpr, tpr)
    plt.show()
    print('auc: %.4f' % (auc(fpr, tpr)))
