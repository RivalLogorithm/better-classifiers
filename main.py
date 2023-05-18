from classifier.classifiers import ParallelClassifier
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

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

    # preprocessing
    # df = pd.DataFrame(load_breast_cancer().data, columns=load_breast_cancer().feature_names)
    # df["target"] = load_breast_cancer().target
    # normal_data = df[df["target"] == 1]
    # anomaly_data = df[df["target"] == 0]
    # print(len(normal_data['target']), len(anomaly_data['target']))
    # df = pd.read_csv('phishing.csv')
    # normal_data = df[df['class'] == 1]
    # anomaly_data = df[df['class'] == -1]
    # df = pd.read_csv('creditcard.csv')
    # normal_data = df[df['Class'] == 0].iloc[:10000,:]
    # anomaly_data = df[df['Class'] == 1]
    # scaler = MinMaxScaler()
    # normal_data_normalized = scaler.fit_transform(normal_data.iloc[:, :-1])
    # anomaly_data_normalized = scaler.fit_transform(anomaly_data.iloc[:, :-1])
    # anomaly_data_normalized = pd.DataFrame(anomaly_data_normalized, columns=anomaly_data.columns[:-1])
    # anomaly_data_normalized["target"] = anomaly_data["target"]


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
    # n_test = df.iloc[10001:, :][df['Class'] == 0]
    # a_test = df[df['Class'] == 1]
    # a_test['Class'] = a_test['Class'].replace(1, -1)
    # anomaly_data['target'] = anomaly_data['target'].replace(0, -1)
    # print(anomaly_data['target'].max())
    # test_data = pd.concat([normal_data.sample(len(anomaly_data['target'])), anomaly_data])

    # fit Classificator
    # clf.fit(normal_data_normalized, test_data, varbounds)
    # d = pd.concat([anomaly_data, normal_data.iloc[20000:20200, :]], ignore_index=True)
    # # d = anomaly_data.append(normal_data.iloc[20000:20001, :])
    # anomaly_data_normalized = scaler.fit_transform(d.iloc[:, :-1])

    # pred = clf.predict(test_data.iloc[:,:-1])
    # print("Accuracy {}".format(accuracy_score(test_data.iloc[:,-1], pred)))
    # print("Precision {}".format(precision_score(test_data.iloc[:,-1], pred, average='weighted')))
    # print("Recall {}".format(recall_score(test_data.iloc[:,-1], pred, average='weighted')))
    # print("F1-score {}".format(f1_score(test_data.iloc[:,-1], pred, average='weighted')))

    # cm = confusion_matrix([-1]*len(pred), pred)
    # sns.heatmap(cm, cmap='Blues')
    # plt.show()
    # print(clf.predict(anomaly_data_normalized))
