import numpy
import sklearn.linear_model


def linear_regresion(data, X_name, Y_name):
    # split data on X and Y
    X = numpy.c_[data[X_name]]
    Y = numpy.c_[data[Y_name]]
    # linear regresion
    model = sklearn.linear_model.LinearRegression()
    # training model
    return model.fit(X, Y)