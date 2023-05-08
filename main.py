from classifier.classifiers import ParallelClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    df = pd.DataFrame(load_breast_cancer().data, columns=load_breast_cancer().feature_names)
    df["target"] = load_breast_cancer().target
    normal_data = df[df["target"] == 0]
    anomaly_data = df[df["target"] == 1]

    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1],
                                                        df.iloc[:, -1],
                                                        test_size=0.2,
                                                        random_state=128)
    clf = ParallelClassifier(["oc-svm", "if", "lof"])
    varbounds = []
    # Varbounds for SVM (kernel, nu, degree, gamma, coef0, shrinking)
    varbound_svm = [['linear', 'poly', 'rbf', 'sigmoid'],
                             [0, 1],
                             [0, 5],
                             [0, 1],
                             [0, 1],
                             [True, False]]
    varbounds.append(varbound_svm)

    # Varbounds for IF (n_estimators, max_samples, contamination, max_features, bootstrap
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

    clf.fit(X_train, X_test, y_train, y_test, varbounds)

    pred = clf.predict(X_test)
    print(accuracy_score(y_test, pred))
