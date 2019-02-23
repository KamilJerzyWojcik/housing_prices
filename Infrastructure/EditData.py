import numpy as np
import hashlib
from Infrastructure import ImportData
from sklearn.model_selection import train_test_split as TrainTestSplit, StratifiedShuffleSplit
import pandas as pd
import sklearn.impute as impute
from sklearn import preprocessing

def SplitRandomTrainTestData(data, testRatio):
    shuffledIndicates = np.random.permutation(len(data))
    testSizeSet = int(len(data) * testRatio)
    testIndices = shuffledIndicates[:testSizeSet]
    trainIndices = shuffledIndicates[testSizeSet:]
    return data.iloc[trainIndices], data.iloc[testIndices]

def _testSetCheck(identifier, testRatio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * testRatio

def SplitTrainTestDataById(data, testRatio, idColumn, hash=hashlib.md5):
    ids = data[idColumn]
    inTestSet = ids.apply(lambda id_: _testSetCheck(id_, testRatio, hash))
    return data.loc[~inTestSet], data.loc[inTestSet]

def GetComputedId(data, col1, col2):
    return data[col1] * 1000 + data[col2]

def SplitTestAndTrainData(dataSplit, data, column):
    for train_index, test_index in dataSplit.split(data, data[column]):
        stratTrainSet = data.loc[train_index]
        stratTestSet = data.loc[test_index]
    return stratTrainSet, stratTestSet

def DropColumn(stratTrainSet, stratTestSet, column):
    for set_ in (stratTrainSet, stratTestSet):
        set_.drop(column, axis=1, inplace=True)
    return stratTrainSet, stratTestSet

def AddNewColums(housingExplorer):
    housingExplorer["Rooms_per_households"] = housingExplorer["total_rooms"] / housingExplorer["households"]
    housingExplorer["Bedrooms_per_rooms"] = housingExplorer["total_bedrooms"] / housingExplorer["total_rooms"]
    housingExplorer["Population_per_households"] = housingExplorer["population"] / housingExplorer["households"]
    return housingExplorer

def AddIdAndModifyIncome(housingWithComputedId):
    housingWithComputedId["id"] = GetComputedId(housingWithComputedId, "longitude", "latitude")
    housingWithComputedId["income_cat"] = np.ceil(housingWithComputedId["median_income"] / 1.5)
    # jeżeli >5 to nie przypisuj, inczej zamien na 5
    housingWithComputedId["income_cat"].where(housingWithComputedId["income_cat"] < 5, 5.0, inplace=True)
    return housingWithComputedId

def GetTrainAndTestData():
    ImportData.GetHousingDataFromUrl()
    housing = ImportData.LoadHousingDataFromPath()
    housingWithComputedId = AddIdAndModifyIncome(housing.copy())
    return TrainTestSplit(housingWithComputedId, test_size=0.2, random_state=42)

def GetData():
    ImportData.GetHousingDataFromUrl()
    housing = ImportData.LoadHousingDataFromPath()
    return AddIdAndModifyIncome(housing.copy())

def GetTrainAndTestDataSKL(housingWithComputedId):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    stratTrainSet, stratTestSet = SplitTestAndTrainData(split, housingWithComputedId, "income_cat")
    return DropColumn(stratTrainSet, stratTestSet, "income_cat")

def GetCopyAssertAndExpected(housingExplorer, label):
    housingDataToAssert = housingExplorer.drop("median_house_value", axis=1)
    housingExpected = housingExplorer["median_house_value"].copy()
    return housingDataToAssert, housingExpected

def SetMedianWhereEmpty(housingDataToAssert):
    imputer = impute.SimpleImputer(strategy="median")
    housingNum = housingDataToAssert.drop("ocean_proximity", axis=1)
    imputer.fit(housingNum)
    X = imputer.transform(housingNum)  # uzupełnanie pustych danych medianą
    return pd.DataFrame(X, columns=housingNum.columns)

def EncodeColumn(column):
    # zamiana tekstu na wartości liczbowe
    encoder = preprocessing.LabelEncoder()
    return encoder.fit_transform(column)
