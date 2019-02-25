from InfrastructurePro import Data, DataDisplayer, MachineLearning, PipelinesModels as pip
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import LabelBinarizer



def NoPipeline():
    pd.set_option('display.expand_frame_repr', False)
    housing = Data.Load_Housing_Data_From_Path()
    housing = Data.AddMedianToGap(housing)
    print(housing.head())
    housing = Data.Modify_Data(housing)
    housing = Data.ScalerData(housing, ["ocean_proximity", "median_house_value"])
    housing = Data.BinarizingCategory(housing, "ocean_proximity")

    train, test = Data.Get_Train_And_Test_Data(housing)

    # console
    # DataDisplayer.DisplayData(train)
    print(train.head(n=1))
    DataDisplayer.DisplayCorellation(train, ["median_house_value"])
    # DataDisplayer.ShowHistogramsData(housing["Population_per_rooms"])

    # MachineLearning.ModelGet(train, LinearRegression(), "Linear Regresion")
    # MachineLearning.ModelGet(train, DecisionTreeRegressor(), "Decision Tree Regresion")
    # MachineLearning.ModelGet(train, RandomForestRegressor(n_estimators=10), "Random Forest Regresion")
    # plot
    # DataDisplayer.ShowCompareMatrix(train, ["median_house_value", "median_income"])


def PipelineDefault():
    pd.set_option('display.expand_frame_repr', False)
    housing = pip.LoadData().LoadFromPath()
    data = pip.TrainTestSeparator().GetDataByMedianIncome(housing.copy())

    attributes = pip.Attributes(
        list(housing.drop("median_house_value", axis=1).drop("ocean_proximity", axis=1)),
        ["ocean_proximity"])

    num_pipeline = Pipeline([
        ('clear_data', pip.ClearData()),
        ('attribute_selector', pip.DataFrameSelector(attributes.numeric)),
        ('imputer', SimpleImputer(strategy="median")),
        ('new_attributes', pip.AttributeAdder(attributes)),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('clear_data', pip.ClearData()),
        ('selector', pip.DataFrameSelector(attributes.categories)),
        ('encoder', pip.CustomBinarizer(attributes))
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline)
    ])
    housing_prepared = full_pipeline.fit_transform(data.TrainSet)

    print(housing_prepared[0])
