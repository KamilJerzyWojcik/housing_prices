from Infrastructure import EditData, MachineLearning
from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd
import sklearn.impute as impute
from sklearn.preprocessing import StandardScaler, LabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer



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



#MachineLearning.LinearRegressionModelGet(stratTrainSet, fullPipeline)
#MachineLearning.TreeRegressionModelGet(stratTrainSet, fullPipeline)
#MachineLearning.RandomForestRegressionModelGet(stratTrainSet, fullPipeline)
#final_model = MachineLearning.RandomForestRegressionBestParamGet(stratTrainSet, fullPipeline)
#MachineLearning.TestFinalModel(final_model, fullPipeline, stratTestSet)


print("--------------------end----------------------")
