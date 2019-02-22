from Infrastructure import ImportData, ShowData

# import numpy
import pandas as pd
# import sklearn
# import matplotlib.pyplot as pyplot
# import scipy

print("-------------------start---------------------")

ImportData.GetHousingDataFromUrl()
housing = ImportData.LoadHousingDataFromPath()
#ShowData.ShowNumericData(housing)
ShowData.ShowHistogramsData(housing)

print("-------------------end---------------------")
