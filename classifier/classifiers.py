from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score
import time
import evolutional.optimization as opt


class ParallelClassifier:
    def __init__(self, classifiers):
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.classifiers = classifiers
        self.best_results = {}
        self.best_classifier = None
        print("ParallelClassifier({})".format(self.classifiers))

    def fit(self, X_train, test_data, varbound):
        self.X_train = X_train
        self.X_test = test_data.iloc[:, :-1]
        self.y_test = test_data.iloc[:, -1]
        models = []
        # if 'oc-svm' in self.classifiers:
        #     models.append(opt.Optimization('genetic-algorithm',
        #                              function=self.one_class_svm_func,
        #                              population_size=100,
        #                              generations=1000,
        #                              varbound=varbound[self.classifiers.index('oc-svm')],
        #                              vartype=['str', 'real', 'int', 'real', 'real', 'bool']))
        # if 'if' in self.classifiers:
        #     models.append(opt.Optimization('genetic-algorithm',
        #                                    function=self.if_func,
        #                                    population_size=100,
        #                                    generations=1000,
        #                                    varbound=varbound[self.classifiers.index('if')],
        #                                    vartype=['int', 'int', 'real', 'real', 'bool']))
        # if 'lof' in self.classifiers:
        #     models.append(opt.Optimization('genetic-algorithm',
        #                                    function=self.lof_func,
        #                                    population_size=100,
        #                                    generations=1000,
        #                                    varbound=varbound[self.classifiers.index('lof')],
        #                                    vartype=['int', 'str', 'int', 'str', 'int', 'real']))

        models.append(opt.Optimization('differential-evolution',
                                       function=self.one_class_svm_func,
                                       population_size=100,
                                       crossover_rate=0.3,
                                       F=1,
                                       generations=100,
                                       division_count=5,
                                       varbound=varbound[self.classifiers.index('oc-svm')],
                                       vartype=['str', 'real', 'int', 'real', 'real', 'bool']))

        models.append(opt.Optimization('differential-evolution',
                                       function=self.if_func,
                                       population_size=100,
                                       crossover_rate=0.3,
                                       F=1,
                                       generations=100,
                                       division_count=5,
                                       varbound=varbound[self.classifiers.index('if')],
                                       vartype=['int', 'int', 'real', 'real', 'bool']))

        models.append(opt.Optimization('differential-evolution',
                                       function=self.lof_func,
                                       population_size=100,
                                       crossover_rate=0.3,
                                       F=1,
                                       generations=100,
                                       division_count=5,
                                       varbound=varbound[self.classifiers.index('lof')],
                                       vartype=['int', 'str', 'int', 'str', 'int', 'real']))

        if len(models) == 0:
            raise ValueError("Classifiers {} not supported".format(self.classifiers))

        self.__run(models)
        temp_res = 0
        best_clf = None
        for clf, res in self.best_results.items():
            if res[0] > temp_res:
                temp_res = res[0]
                best_clf = res[1]
        self.best_classifier = best_clf
        print("Best classifier: {}".format(best_clf))

    def predict(self, data):
        return self.best_classifier.predict(data)

    def __run(self, models):
        start_time = time.time()
        p = ThreadPoolExecutor(len(models))

        results = []
        for model in models:
            # results.append(model.optimize())
            results.append(p.submit(model.optimize))
        for i in range(len(results)):
            self.best_results.update({self.classifiers[i]: results[i].result()})
            # self.best_results.update({self.classifiers[i]: results[i]})
        p.shutdown()
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
        clf.fit(self.X_train)
        pred = clf.predict(self.X_test)
        f1 = f1_score(self.y_test, pred, average='weighted')
        return -f1, clf, X

    def if_func(self, X):
        clf = IsolationForest(n_estimators=X[0],
                              max_samples=X[1],
                              contamination=X[2],
                              max_features=X[3],
                              bootstrap=X[4],
                              verbose=False,
                              n_jobs=-1)
        clf.fit(self.X_train)
        pred = clf.predict(self.X_test)
        f1 = f1_score(self.y_test, pred, average='weighted')
        return -f1, clf, X

    def lof_func(self, X):
        clf = LocalOutlierFactor(n_neighbors=X[0],
                                 algorithm=X[1],
                                 leaf_size=X[2],
                                 metric=X[3],
                                 p=X[4],
                                 contamination=X[5],
                                 novelty=True,
                                 n_jobs=-1)
        clf.fit(self.X_train)
        pred = clf.predict(self.X_test)
        f1 = f1_score(self.y_test, pred, average='weighted')
        return -f1, clf, X
