import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from os import path
from pandas import read_csv
from sklearn.model_selection import StratifiedShuffleSplit



class LoadData:
    def __init__(self, HOUSING_PATH=path.join("data", "house")):
        self.housingPath = HOUSING_PATH

    def LoadFromPath(self):
        csvPath = path.join(self.housingPath, "housing.csv")
        return read_csv(csvPath)


class TrainTestSeparator:
    def __init__(self):
        self.TestSet = []
        self.TrainSet = []
        self.TrainSetPredictors = []
        self.TrainSetExpected = []

    def GetDataByMedianIncome(self, data):
        data["income_cat"] = np.ceil(data["median_income"] / 1.5)
        data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace=True)
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_index, test_index in split.split(data, data["income_cat"]):
            self.TrainSet = data.loc[train_index]
            self.TestSet = data.loc[test_index]

        for set_ in (self.TestSet, self.TrainSet):
            set_.drop("income_cat", axis=1, inplace=True)

        self.TrainSetPredictors = self.TrainSet.drop("median_house_value", axis=1)
        self.TrainSetExpected = self.TrainSet["median_house_value"]

        return self


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attributes].values


class DataAtributeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, typeAttributes):
        self.typeAttributes = typeAttributes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.typeAttributes]


class CustomBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.encoder = LabelBinarizer()
        self.attributes = attributes

    def fit(self, X, y=None,**fit_params):
        return self

    def transform(self, X):
        self.encoder_t = self.encoder.fit(X).transform(X)
        self.attributes.categories = self.encoder.classes_
        return self.encoder_t


class Attributes:
    def __init__(self, numeric, categories):
        self.numeric = np.array(numeric)
        self.categories = categories

    def GetAllAtributes(self):
        return np.append(self.numeric, self.categories)


class ClearData(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        new_data = X[X["median_house_value"] < 500000].copy()
        new_data = new_data[new_data["housing_median_age"] < 51].copy()
        new_data = new_data.reset_index(drop=True)
        return new_data


class AttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        new_data = pd.DataFrame(X, columns=self.attributes.numeric)

        new_data["Rooms_per_family"] = new_data["total_rooms"] / new_data["households"]
        new_data["Bedrooms_per_room"] = new_data["total_bedrooms"] / new_data["total_rooms"]
        new_data["Population_per_family"] = new_data["population"] / new_data["households"]
        new_data["Income_in_population"] = np.power(new_data["population"] * new_data["median_income"], 1 / 3)
        new_data["Population_per_rooms"] = np.power(new_data["population"] / new_data["total_rooms"], 1 / 5)
        new_data["total_rooms"] = np.power(new_data["total_rooms"], 1 / 2)
        new_data["households"] = np.power(new_data["households"], 1 / 3)
        self.attributes.numeric = list(new_data)
        return new_data.values
