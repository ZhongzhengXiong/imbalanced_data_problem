from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt


def calculate_roc(targets, scores):
    fpr, tpr, thresholds = roc_curve(targets, scores)
    plt.plot(fpr, tpr)
    plt.show()
    print('auc: %.4f' % (auc(fpr, tpr)))
