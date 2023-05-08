from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
from multiprocess import pool

import time
import evolutional.optimization as opt


class ParallelClassifier:
    def __init__(self, classifiers):
        self.y_test = None
        self.X_test = None
        self.y_train = None
        self.X_train = None
        self.classifiers = classifiers
        self.best_results = {}
        self.best_classifier = None
        print("ParallelClassifier({})".format(self.classifiers))

    def fit(self, X_train, X_test, y_train, y_test, varbound):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        models = []
        if 'oc-svm' in self.classifiers:
            models.append(opt.Optimization('genetic-algorithm',
                                     function=self.one_class_svm_func,
                                     population_size=100,
                                     generations=10,
                                     varbound=varbound[self.classifiers.index('oc-svm')],
                                     vartype=['str', 'real', 'int', 'real', 'real', 'bool']))
        if 'if' in self.classifiers:
            models.append(opt.Optimization('genetic-algorithm',
                                           function=self.if_func,
                                           population_size=100,
                                           generations=10,
                                           varbound=varbound[self.classifiers.index('if')],
                                           vartype=['int', 'int', 'real', 'real', 'bool']))
        if 'lof' in self.classifiers:
            models.append(opt.Optimization('genetic-algorithm',
                                           function=self.lof_func,
                                           population_size=100,
                                           generations=10,
                                           varbound=varbound[self.classifiers.index('lof')],
                                           vartype=['int', 'str', 'int', 'str', 'int', 'real']))

        if len(models) == 0:
            raise ValueError("Classifiers {} not supported".format(self.classifiers))

        self.__run(models)
        temp_res = 0
        best_clf = None
        for clf, res in self.best_results.items():
            if res[1] > temp_res:
                temp_res = res[1]
                best_clf = res[0]
        self.best_classifier = best_clf
        print("Best classifier: {}".format(best_clf))


    def predict(self, X_test):
        return self.best_classifier.predict(X_test)

    def __run(self, models):
        start_time = time.time()
        p = pool.Pool(processes=20)

        results = []
        for model in models:
            results.append(p.apply_async(model.optimize))


        for i in range(len(results)):
            self.best_results.update({self.classifiers[i]: results[i].get()})

        print("--- %s seconds ---" % (time.time() - start_time))
        print("Best results: {}".format(self.best_results))

    def one_class_svm_func(self, X):
        clf = OneClassSVM(kernel=X[0],
                          nu=X[1],
                          degree=X[2],
                          gamma=X[3],
                          coef0=X[4],
                          shrinking=X[5]
                          )
        clf.fit(self.X_train, self.y_train)
        pred = clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, pred)
        return clf, accuracy

    def if_func(self, X):
        clf = IsolationForest(n_estimators=X[0],
                              max_samples=X[1],
                              contamination=X[2],
                              max_features=X[3],
                              bootstrap=X[4],
                              verbose=False,
                              n_jobs=-1)
        clf.fit(self.X_train, self.y_train)
        pred = clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, pred)

        return clf, accuracy

    def lof_func(self, X):
        clf = LocalOutlierFactor(n_neighbors=X[0],
                                 algorithm=X[1],
                                 leaf_size=X[2],
                                 metric=X[3],
                                 p=X[4],
                                 contamination=X[5],
                                 novelty=True,
                                 n_jobs=-1)
        clf.fit(self.X_train, self.y_train)
        pred = clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, pred)

        return clf, accuracy
