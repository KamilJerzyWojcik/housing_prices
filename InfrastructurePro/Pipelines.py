from InfrastructurePro import Data, DataDisplayer, MachineLearning, PipelinesModels as pip
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.svm import SVR


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
    attributesTrain = pip.Attributes(
        list(housing.drop("median_house_value", axis=1).drop("ocean_proximity", axis=1)),
        ["ocean_proximity"])
    attributesTest = pip.Attributes(
        list(housing.drop("median_house_value", axis=1).drop("ocean_proximity", axis=1)),
        ["ocean_proximity"])

    housing_prepared = prepare_data(getPipeline(attributesTrain), attributesTrain, data.TrainSet)

    if False:
        param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
    ]
        best_predictor = MachineLearning.ModelGet(housing_prepared,
                                              RandomForestRegressor(n_estimators=10),
                                              "Random Forest Regresion",
                                              paramGrid=param_grid)

    if False:
        param_grid_randomize = {
            'n_estimators': [30, 40, 50],
            'max_features': [8, 10, 12],
            "bootstrap": [True, False]
        }
        best_predictor = MachineLearning.ModelGet(housing_prepared,
                              RandomForestRegressor(n_estimators=10),
                              "Random Forest Regresion",
                              paramGrid=param_grid_randomize,
                              iterationRandom=20)

    if True:
        best_predictor = MachineLearning.ModelGet(housing_prepared,
                                                  SVR(gamma='scale', C=100.0, epsilon=0.1, kernel="rbf"),
                                                  "SVR")

    data_test_prepare = prepare_data(getPipeline(attributesTest), attributesTest, data.TestSet)
    MachineLearning.PredictFinalData(data_test_prepare, best_predictor, "Random Forest Regresion")


def getPipeline(attributes):
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
    return FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline)
    ])


def prepare_data(pipeline, attributes, data_set):
    data_prepare = pipeline.fit_transform(data_set)
    data_frame = pd.DataFrame(data_prepare, columns=attributes.GetAllAtributes())
    data_frame["median_house_value"] = pip.ClearData().transform(data_set)["median_house_value"]
    return data_frame
