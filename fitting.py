import numpy as np
from sklearn.linear_model import SGDClassifier
from data import DataReader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics


diabetes_reader = DataReader("diabetes.csv")


def fit(subset1, subset2, classifier="knn"):

    data = diabetes_reader.data_train[[subset1, subset2]]
    data_test = diabetes_reader.data_test[[subset1, subset2]]
    target = diabetes_reader.target_train
    target_test = diabetes_reader.target_test

    if classifier == "knn":
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(data, np.ravel(target))
        target_pred = knn.predict(data_test)

    if classifier == "SVC":
        logreg = SVC(gamma='scale')
        logreg.fit(data, np.ravel(target))
        print(np.ravel(target))
        target_pred = logreg.predict(data_test)

    results = metrics.accuracy_score(target_test, target_pred)
    print(results)

if __name__ == "__main__":
    fit("pressure", "glucose", "SVC")

