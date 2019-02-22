import matplotlib.pyplot as pyplot
import pandas as pd

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
    dataFramePandas.hist(bins=50, figsize=(20, 15))
    pyplot.show()