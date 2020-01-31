from sklearn import neighbors, datasets, linear_model
import pylab as pl
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from data import DataReader



 
def plot_visualizer(classifier_name, subset1, subset2):
    """Plotter based on plot_irirs from lecture-gitHub
        Args:
            Classifier(string): knn or svc
            subset1(string): Feature1 from csv
            subset2(string): Feature1 from csv
    """

    
    if(classifier_name == "knn"):
        classifier = KNeighborsClassifier(n_neighbors=3)
    if(classifier_name == "svc"):
        classifier = SVC(gamma='scale')
    
    dataset = "diabetes.csv"
    df = DataReader(dataset)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    X = df.data_train[[subset1, subset2]]
    y = df.target_train.values
    #Setting pos/neg to 1/0
    y = [0 if e == "neg" else 1 for e in y]

    classifier.fit(X, y)
    x_min, x_max = X[subset1].min() - 1, X[subset1].max() + 1
    y_min, y_max = X[subset2].min() - 1, X[subset2].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    pl.figure()
    pl.pcolormesh(xx, yy, Z, cmap=cmap_light)

    pl.scatter(X[subset1], X[subset2], c=y, cmap=cmap_bold)
    pl.xlabel(subset1)
    pl.ylabel(subset2)
    pl.axis('tight')
    savename = classifier_name + subset1 + subset2
    pl.savefig("static/" + savename  +".png")
    #pl.show()

    data_test = df.data_test[[subset1, subset2]]
    target_test = df.target_test.values
    #Setting pos/neg to 1/0
    target_test = [0 if e == "neg" else 1 for e in target_test]
    target_pred = classifier.predict(data_test)
    results = metrics.accuracy_score(target_test, target_pred)

    return results


if __name__ == "__main__":
    plot_visualizer("svc", "glucose", "pressure")
