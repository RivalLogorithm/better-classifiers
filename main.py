from classifier.classifiers import ParallelClassifier
from sklearn.datasets import make_classification
import pandas as pd


def get_data():
    X, y = make_classification(n_samples=2000,
                               n_features=20,
                               n_informative=10,
                               random_state=100,
                               weights=[0.8])
    dataset = pd.DataFrame(X)
    dataset['y'] = y
    normal_data = dataset[dataset.y == 0]
    normal_data.iloc[:, -1] = 1
    outlier_data = dataset[dataset.y == 1]
    outlier_data.iloc[:, -1] = -1
    train = normal_data.iloc[:1000, :]
    test = normal_data.iloc[1001:, :]
    outliers = outlier_data.iloc[:, :]

    return train, pd.concat([test, outliers])


if __name__ == "__main__":
    train, test = get_data()
    clf = ParallelClassifier(["oc-svm", "if", "lof"])
    varbounds = []

    # Varbounds for SVM (kernel, nu, degree, gamma, coef0, shrinking)
    varbound_svm = [['linear', 'rbf', 'sigmoid'],
                             [0, 1],
                             [0, 3],
                             [0, 100],
                             [0, 1],
                             [True, False]]
    varbounds.append(varbound_svm)

    # Varbounds for IF (n_estimators, max_samples, contamination, max_features, bootstrap)
    varbound_if = [[1, 100],
                   [1, 100],
                   [0, 0.5],
                   [0, 1],
                   [True, False]]
    varbounds.append(varbound_if)

    # Varbounds for LOF (n_neighbors, algorithm, leaf_size, metric, p, contamination
    varbound_lof = [[10, 50],
                    ['ball_tree', 'kd_tree', 'brute', 'auto'],
                    [10, 100],
                    ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan'],
                    [1, 5],
                    [0, 0.5]]

    varbounds.append(varbound_lof)

    clf.fit(train.iloc[:,:-1], test, varbounds)
