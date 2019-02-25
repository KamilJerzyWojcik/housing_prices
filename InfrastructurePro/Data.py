from pandas import read_csv
import pandas as pd
from os import path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

HOUSING_PATH = path.join("data", "house")

def Load_Housing_Data_From_Path(housingPath = HOUSING_PATH):
    csvPath = path.join(housingPath, "housing.csv")
    return read_csv(csvPath)


def AddMedianToGap(housing):
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)
    housing_tr["ocean_proximity"] = housing["ocean_proximity"]
    return housing_tr


def Get_Train_And_Test_Data(data):
    data["income_cat"] = np.ceil(data["median_income"] / 1.5)
    data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    for set_ in (strat_test_set, strat_train_set):
        set_.drop("income_cat", axis=1, inplace=True)
    return strat_train_set, strat_test_set


def Modify_Data(data):
    new_data1 = data[data["median_house_value"] < 500000].copy()
    new_data = new_data1[new_data1["housing_median_age"] < 51].copy()
    new_data = new_data.reset_index(drop=True)
    new_data["Rooms_per_family"] = new_data["total_rooms"] / new_data["households"]
    new_data["Bedrooms_per_room"] = new_data["total_bedrooms"] / new_data["total_rooms"]
    new_data["Population_per_family"] = new_data["population"] / new_data["households"]
    new_data["Income_in_population"] = np.power(new_data["population"] * new_data["median_income"], 1/3)
    new_data["Population_per_rooms"] = np.power(new_data["population"] / new_data["total_rooms"], 1/5)

    new_data["total_rooms"] = np.power(new_data["total_rooms"], 1 / 2)
    new_data["households"] = np.power(new_data["households"], 1 / 3)

    #negatywny wplyw:
    # new_data["total_bedrooms"] = np.power(new_data["total_bedrooms"], 1/3)
    # new_data["population"] = np.power(new_data["population"], 1/3)
    #new_data["median_income"] = np.sqrt(new_data["median_income"])
    # new_data["Bedroom_per_population"] = new_data["population"] / new_data["total_bedrooms"]
    return new_data


def BinarizingCategory(data, category):
    encoder = LabelBinarizer()
    housing_Cat_One_Hot = encoder.fit_transform(data[category].astype(str))
    housing_cat = pd.DataFrame(housing_Cat_One_Hot, columns=encoder.classes_)
    for cl in encoder.classes_:
        data[cl] = housing_cat[cl]
    data.drop(category, axis=1, inplace=True)
    return data


def ScalerData(housing, category):
    scaler = StandardScaler()
    housing_num = housing.drop(category, axis=1)
    housing_num = scaler.fit_transform(housing_num)
    housing_num = pd.DataFrame(housing_num, columns=housing.drop(category, axis=1).columns)
    housing_num[category[0]] = housing[category[0]]
    housing_num[category[1]] = housing[category[1]]

    return housing_num


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
        new_data = new_data.drop("median_house_value", axis=1)
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
