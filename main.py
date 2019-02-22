from Infrastructure import ImportData, ShowData, EditData

# import numpy
import pandas as pd
# import sklearn
# import matplotlib.pyplot as pyplot
# import scipy

print("-------------------start---------------------")
pd.set_option('display.expand_frame_repr', False)

ImportData.GetHousingDataFromUrl()
housing = ImportData.LoadHousingDataFromPath()
#ShowData.ShowNumericData(housing)
#ShowData.ShowHistogramsData(housing)

trainSetRandom, testSetRandom = EditData.SplitRandomTrainTestData(housing, 0.2)

housingWithId = housing.reset_index()
housingWithComputedId = housing.reset_index()
housingWithComputedId["id"] = EditData.GetComputedId(housing, "longitude", "latitude")
trainSetFixed, testSetFixed = EditData.SplitTrainTestDataById(housingWithComputedId, 0.2, "index")


print( trainSetFixed)



print("--------------------end----------------------")
