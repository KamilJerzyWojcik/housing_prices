import matplotlib.pyplot as pyplot
import pandas as pd
from pandas.plotting import scatter_matrix

def ShowNumericData(dataFramePandas):
    pd.set_option('display.expand_frame_repr', False)
    print("---dane-------------------------------------")
    print(dataFramePandas.head(n=10))
    print("")
    print("---info-------------------------------------")
    print(dataFramePandas.info())
    print("")
    print("---kategorie ocean_proximity----------------")
    print(dataFramePandas["ocean_proximity"].value_counts())
    print("")
    print("---atrybuty numeryczne------------------------------")
    print(dataFramePandas.describe())

def ShowHistogramsData(dataFramePandas):
    dataFramePandas.hist(bins=50, figsize=(15, 10))
    pyplot.show()

def ShowCompareData(data, trainData, testData, byColumn):
    print("All data:")
    print(data[byColumn].value_counts() / len(data))
    print("Train data:")
    print(trainData[byColumn].value_counts() / len(trainData))
    print("Test data:")
    print(testData[byColumn].value_counts() / len(testData))

def ShowVisualizationDataByGeo(housingExplorer, circleDimension, color, normalizationNumber = 100):
    housingExplorer.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                         s=housingExplorer[circleDimension] / normalizationNumber, label=circleDimension,
                         figsize=(10, 7), c=color,
                         cmap=pyplot.get_cmap("jet"), colorbar=True)
    pyplot.show()
    pyplot.legend()

def ShowCompareMatrix(data, attributes=["median_house_value", "median_income", "total_rooms", "housing_median_age"]):
    corrMatrix = data.corr()
    print(corrMatrix["median_house_value"].sort_values(ascending=False))
    scatter_matrix(data[attributes], figsize=(8, 6))
    data.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
