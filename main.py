from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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

def get_credit_data():
    credit_data = pd.read_csv("Credit.csv")
    credit_data = credit_data.dropna()
    le = LabelEncoder()
    credit_data['approval'] = le.fit_transform(credit_data.approval.values)
    credit_data['job'] = le.fit_transform(credit_data.job.values)
    credit_data['marital'] = le.fit_transform(credit_data.marital.values)
    credit_data['education'] = le.fit_transform(credit_data.education.values)
    credit_data['housing'] = le.fit_transform(credit_data.housing.values)
    x = credit_data.values
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    credit_df = pd.DataFrame(x_scaled)
    credit_normal = credit_df[credit_df.iloc[:, -1] == 0]
    credit_normal.iloc[:, -1] = 1
    credit_outlier = credit_df[credit_df.iloc[:, -1] == 1]
    credit_outlier.iloc[:, -1] = -1
    train = credit_normal.iloc[:3479, :]
    test = credit_normal.iloc[3479:, :]
    return train, pd.concat([test, credit_outlier])

def get_celebral_data():
    celebral_data = pd.read_csv("celebral.csv")
    celebral_data = celebral_data.dropna()
    celebral_data = celebral_data.iloc[:, 1:]
    le = LabelEncoder()
    celebral_data['gender'] = le.fit_transform(celebral_data.gender.values)
    celebral_data['ever_married'] = le.fit_transform(celebral_data.ever_married.values)
    celebral_data['work_type'] = le.fit_transform(celebral_data.work_type.values)
    celebral_data['Residence_type'] = le.fit_transform(celebral_data.Residence_type.values)
    celebral_data['smoking_status'] = le.fit_transform(celebral_data.smoking_status.values)
    x = celebral_data.values
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    celebral_df = pd.DataFrame(x_scaled)
    celebral_outlier = celebral_df[celebral_df.iloc[:, -1] == 1]
    celebral_outlier.iloc[:, -1] = -1
    celebral_normal = celebral_df[celebral_df.iloc[:, -1] == 0]
    celebral_normal.iloc[:, -1] = 1
    train = celebral_normal.iloc[:27976, :]
    test = celebral_normal.iloc[27976:, :]
    return train, pd.concat([test, celebral_outlier])

if __name__ == "__main__":
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

    # train, test = get_credit_data()
    # clf.fit(train.iloc[:, :-1], test, varbounds)

    train, test = get_celebral_data()
    clf.fit(train.iloc[:, :-1], test, varbounds)

