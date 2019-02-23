from Infrastructure import EditData, CombinedAttributesAdder as CAA, DataFrameSelector as DFS
from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd
import sklearn.impute as impute
from sklearn.preprocessing import StandardScaler, LabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression

print("-------------------start---------------------")
pd.set_option('display.expand_frame_repr', False)

housingWithComputedId = EditData.GetData()
stratTrainSet, stratTestSet = EditData.GetTrainAndTestDataSKL(housingWithComputedId)

housingExplorer = EditData.AddNewColums(stratTrainSet)
housingDataToAssert, housingExpected = EditData.GetCopyAssertAndExpected(housingExplorer, "median_house_value")

housingNum = housingDataToAssert.drop("ocean_proximity", axis=1)

numAttributs = list(housingNum)
catAttributs = ["ocean_proximity"]

numPipeline = Pipeline([
        ('imputer', impute.SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])

fullPipeline = ColumnTransformer([
        ("num", numPipeline, numAttributs),
        ("cat", OneHotEncoder(), catAttributs),
    ])

housingPrepared = fullPipeline.fit_transform(stratTrainSet)
housingLabels = stratTrainSet["median_house_value"].copy()

lin_reg = LinearRegression()

lin_reg.fit(housingPrepared, housingLabels)
print(housingPrepared.shape)

someDataPrepared = fullPipeline.transform(stratTrainSet.iloc[:3])

someLabels = housingLabels.iloc[:3]

print("prognoza: ", lin_reg.predict(someDataPrepared))
print("Etykiety: ", list(someLabels))
print("--------------------end----------------------")
