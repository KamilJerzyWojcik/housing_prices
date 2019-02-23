from Infrastructure import ImportData, ShowData, EditData
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split as TrainTestSplit, StratifiedShuffleSplit
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import sklearn.impute as impute
from sklearn import preprocessing

# import numpy
import pandas as pd
# import sklearn
# import matplotlib.pyplot as pyplot
# import scipy

print("-------------------start---------------------")
pd.set_option('display.expand_frame_repr', False)

housingWithComputedId = EditData.GetData()
stratTrainSet, stratTestSet = EditData.GetTrainAndTestDataSKL(housingWithComputedId)

housingExplorer = EditData.AddNewColums(stratTrainSet)
housingDataToAssert, housingExpected = EditData.GetCopyAssertAndExpected(housingExplorer, "median_house_value")
housingAfterTransform = EditData.SetMedianWhereEmpty(housingDataToAssert)

housingCatEncoded = EditData.EncodeColumn(housingExplorer["ocean_proximity"])

hotEncoder = preprocessing.OneHotEncoder(handle_unknown='ignore')

housingCatEncodedHotOne = hotEncoder.fit_transform(housingCatEncoded.reshape(-1, 1))

print(housingCatEncodedHotOne)


print("--------------------end----------------------")
