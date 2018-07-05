from imblearn.ensemble import EasyEnsemble
from load_data import load_data
import logistic_regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

if __name__ == '__main__':
    pca = PCA(n_components=2)

    # load data

    X_train, y_train = load_data('./dataset/car/car-vgood-5-fold/car-vgood-5-1tra.dat')
    X_test, y_test = load_data('./dataset/car/car-vgood-5-fold/car-vgood-5-1tst.dat')
    X_train, y_train = map(np.array, [X_train, y_train])
    X_test, y_test = map(np.array, [X_test, y_test])

    X_vis = pca.fit_transform(X_train)

    ee = EasyEnsemble(random_state=42)

    X_res, y_res = ee.fit_sample(X_train, y_train)
    X_res_vis = []
    for X_r in X_res:
        X_res_vis.append(pca.transform(X_r))
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(X_vis[y_train == 0, 0], X_vis[y_train == 0, 1], label="Class #0", alpha=0.5)
    ax1.scatter(X_vis[y_train == 1, 0], X_vis[y_train == 1, 1], label="Class #1", alpha=0.5)
    ax1.set_title('Original set')
    ax2.scatter(X_vis[y_train == 0, 0], X_vis[y_train == 0, 1], label="Class #0", alpha=0.5)
    for iy, e in enumerate(X_res_vis):
        ax2.scatter(e[y_res[iy] == 1, 0], e[y_res[iy] == 1, 1],
                    label="Class #1 - set #{}".format(iy), alpha=0.5)
    ax2.set_title('Easy ensemble')

    # make nice plotting
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.set_xlim([-6, 8])
        ax.set_ylim([-6, 6])
        ax.legend()

    plt.tight_layout()
    plt.show()
