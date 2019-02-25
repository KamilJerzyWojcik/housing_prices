import matplotlib.pyplot as pyplot
from pandas.plotting import scatter_matrix
import numpy as np

def DisplayData(data, categories=[]):
    print("Dane......................")
    print(data.head(n=1))
    print("")
    print("kategorie.................")
    for cat in categories:
        print("...", cat, "................")
        print(data[cat].value_counts())
        print("")
    print("Info.....................")
    print(data.info())


def ShowHistogramsData(dataFramePandas):
    dataFramePandas.hist(bins=50, figsize=(5, 2))
    pyplot.show()


def DisplayCorellation(data, params):
    corr_matrix = data.corr()
    for p in params:
        print("..ABSOLUTE.", p, "...")
        print(np.abs(corr_matrix[p]).sort_values(ascending=False))
        print(np.sum(corr_matrix[p]))
        print(".........................................")


def ShowCompareMatrix(data, attributes=["median_house_value", "median_income", "total_rooms", "housing_median_age"]):
    scatter_matrix(data[attributes], figsize=(8, 6))
    pyplot.show()
