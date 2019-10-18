"""
Designed and developed by: Nika Abedi - contact email: nikka.abedi@gmail.com
*****************************************************************************
Logistic Regression Class
*****************************************************************************
ds : m by n array, m samples, n features
data : m samples by n features
target : 1 by m class label or targets

"""
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

class logisticRegression:

    # -------------------------------------------------------------
    # Class constructor
    # -------------------------------------------------------------
    model = []

    def __init__(self):
        # The first gaussian samples
        center1 = [0, 0]
        cov1 = [[3, 0], [0, 3]]
        c1 = np.random.multivariate_normal(center1, cov1, 300)
        c1 = np.append(c1, np.zeros([c1.shape[0], 1]), 1)

        # The second gaussian samples
        center2 = [8, 8]
        cov2 = [[4, 0], [0, 4]]
        c2 = np.random.multivariate_normal(center2, cov2, 150)
        c2 = np.append(c2, np.ones([c2.shape[0], 1]), 1)

        # Scatter the gaussian samples
        plt.scatter(c1[:, 0], c1[:, 1])
        plt.scatter(c2[:, 0], c2[:, 1])

        self.ds = c1
        self.ds = np.append(self.ds, c2, 0)
        np.random.shuffle(self.ds)
        self.data = self.ds[:, 0:1]
        self.target = self.ds[:, 2]

        # Set training and test data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.ds[:, 0:2], self.ds[:, -1],
                                                                              test_size=0.3)
        # Train the model
        self.train()

        # Load Data set
        # self.ds = datasets.load_iris()
        # Load samples
        # self.data = self.ds.data
        # Load class labels
        # self.target = self.ds.target

    # -------------------------------------------------------------
    # Run logistic regression
    # -------------------------------------------------------------
    def train(self):
        self.model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial',max_iter=1000).\
            fit(self.x_train, self.y_train)

    # -------------------------------------------------------------
    # predict targets of all samples targets
    # -------------------------------------------------------------
    def predict_all(self):
        return self.model.predict(self.x_test)

    # -------------------------------------------------------------
    # Probability estimates to predict single sample target
    # -------------------------------------------------------------
    def predict_single(self, n):
        return self.model.predict(n)

    # -------------------------------------------------------------
    # predict data interval sample targets
    # -------------------------------------------------------------
    def predict_interval(self, start, end):

        return self.model.predict(self.x_test[start:end])

    # -------------------------------------------------------------
    # Returns the mean accuracy on the given test data and labels.
    # -------------------------------------------------------------
    def accuracy(self):
        accr = self.model.score(self.x_test, self.y_test)
        return accr

    # -------------------------------------------------------------
    # defines which condition to be run
    # -------------------------------------------------------------
    def main(self, flag):
        if flag == 0:
            result = self.predict_all()
        elif flag == 1:
            result = self.predict_single(80)
        elif flag == 2:
            result = self.predict_interval(50, 100)
        else:
            result = self.predict_all()
        return result
