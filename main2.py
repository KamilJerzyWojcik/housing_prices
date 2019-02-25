import pandas as pd
from InfrastructurePro import Data, DataDisplayer, MachineLearning
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

print("-------------------start---------------------")
pd.set_option('display.expand_frame_repr', False)
housing = Data.Load_Housing_Data_From_Path()
housing = Data.AddMedianToGap(housing)
print(housing.head())
housing = Data.Modify_Data(housing)
housind_display = housing.copy()
housing = Data.ScalerData(housing, ["ocean_proximity", "median_house_value"])

housing = Data.BinarizingCategory(housing, "ocean_proximity")


train, test = Data.Get_Train_And_Test_Data(housing)

#console
#DataDisplayer.DisplayData(train)
print(train.head(n=1))
DataDisplayer.DisplayCorellation(train, ["median_house_value"])
DataDisplayer.ShowHistogramsData(housing["Income_in_population"])

#MachineLearning.ModelGet(train, LinearRegression(), "Linear Regresion")
#MachineLearning.ModelGet(train, DecisionTreeRegressor(), "Decision Tree Regresion")
#MachineLearning.ModelGet(train, RandomForestRegressor(n_estimators=10), "Random Forest Regresion")


# plot

#DataDisplayer.ShowCompareMatrix(train, ["median_house_value", "median_income"])
print("--------------------end----------------------")
