from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

paramGridDefault = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

def LinearRegressionModelGet(stratTrainSet, fullPipeline):
    housingPrepared = fullPipeline.fit_transform(stratTrainSet)
    housingLabels = stratTrainSet["median_house_value"].copy()
    lin_reg = LinearRegression()
    lin_reg.fit(housingPrepared, housingLabels)
    someDataPrepared = fullPipeline.transform(stratTrainSet.iloc[:3])
    someLabels = housingLabels.iloc[:3]
    housingPrediction = lin_reg.predict(housingPrepared)
    linMSE = mean_squared_error(housingLabels, housingPrediction)
    linRMSE = np.sqrt(linMSE)
    scores = cross_val_score(lin_reg, housingPrepared, housingLabels, scoring="neg_mean_squared_error", cv=10)
    linRMSEscores = np.sqrt(-scores)
    print("")
    print("...............Linear Regresion:...................")
    print("prognoza: ", lin_reg.predict(someDataPrepared))
    print("Etykiety: ", list(someLabels))
    print("RMSE: ", linRMSE)
    print(".................................................")
    print("score RMSE średnia z 10: ", linRMSEscores.mean())
    print("score RMSE ochylenie standardowe: ", linRMSEscores.std())
    print("..................................................")
    return lin_reg

def TreeRegressionModelGet(stratTrainSet, fullPipeline):
    housingPrepared = fullPipeline.fit_transform(stratTrainSet)
    housingLabels = stratTrainSet["median_house_value"].copy()
    treeReg = DecisionTreeRegressor()
    treeReg.fit(housingPrepared, housingLabels)
    someDataPrepared = fullPipeline.transform(stratTrainSet.iloc[:3])
    someLabels = housingLabels.iloc[:3]
    housingPrediction = treeReg.predict(housingPrepared)
    linMSE = mean_squared_error(housingLabels, housingPrediction)
    linRMSE = np.sqrt(linMSE)
    scores = cross_val_score(treeReg, housingPrepared, housingLabels, scoring="neg_mean_squared_error", cv=10)
    treeRMSEscores = np.sqrt(-scores)
    print("")
    print("...............Tree Regresion:...................")
    print("prognoza: ", treeReg.predict(someDataPrepared))
    print("Etykiety: ", list(someLabels))
    print("RMSE: ", linRMSE)
    print(".................................................")
    print("score RMSE średnia z 10: ", treeRMSEscores.mean())
    print("score RMSE ochylenie standardowe: ", treeRMSEscores.std())
    print("..................................................")
    return treeReg

def RandomForestRegressionModelGet(stratTrainSet, fullPipeline):
    housingPrepared = fullPipeline.fit_transform(stratTrainSet)
    housingLabels = stratTrainSet["median_house_value"].copy()
    randomForestReg = RandomForestRegressor(n_estimators=10)
    randomForestReg.fit(housingPrepared, housingLabels)
    someDataPrepared = fullPipeline.transform(stratTrainSet.iloc[:3])
    someLabels = housingLabels.iloc[:3]
    housingPrediction = randomForestReg.predict(housingPrepared)
    linMSE = mean_squared_error(housingLabels, housingPrediction)
    linRMSE = np.sqrt(linMSE)
    scores = cross_val_score(randomForestReg, housingPrepared, housingLabels, scoring="neg_mean_squared_error", cv=10)#cv - ilosc wyuczen
    treeRMSEscores = np.sqrt(-scores)
    print("")
    print("..........Random forest Regresion:...................")
    print("prognoza: ", randomForestReg.predict(someDataPrepared))
    print("Etykiety: ", list(someLabels))
    print("RMSE: ", linRMSE)
    print(".................................................")
    print("score RMSE średnia z 10: ", treeRMSEscores.mean())
    print("score RMSE ochylenie standardowe: ", treeRMSEscores.std())
    print("..................................................")
    return randomForestReg

def RandomForestRegressionBestParamGet(stratTrainSet, fullPipeline, paramGrid=paramGridDefault):
    housingPrepared = fullPipeline.fit_transform(stratTrainSet.drop("median_house_value", axis=1))
    housingLabels = stratTrainSet["median_house_value"].copy()
    forestReg = RandomForestRegressor()
    gridSearch = GridSearchCV(forestReg, paramGrid, cv=5, scoring="neg_mean_squared_error")
    gridSearch.fit(housingPrepared, housingLabels)
    #print(gridSearch.best_params_)
    #print(gridSearch.best_estimator_)
    featureImportances = gridSearch.best_estimator_.feature_importances_

    print("nazwy kategorii: ", fullPipeline["cat"].classes)
    columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
               'total_bedrooms', 'population', 'households', 'median_income',
               'median_house_value', 'ocean_proximity', 'Rooms_per_households',
               'Bedrooms_per_rooms', 'Population_per_households'] + fullPipeline["cat"].classes

    for x, y in zip(featureImportances, columns):
        print(x, y)



    # print("istotnosc parametrow: ", featureImportances)
    # print(stratTrainSet.columns)
    cvres = gridSearch.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    return gridSearch.best_estimator_

def TestFinalModel(final_model, fullPipeline, stratTestSet):
    Y_test = stratTestSet["median_house_value"].copy()
    X_test_prepared = fullPipeline.transform(stratTestSet.drop("median_house_value", axis=1))
    final_prediction = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(Y_test, final_prediction)
    final_rmse = np.sqrt(final_mse)

    print("......wynik......")
    print("RMSE: ", final_rmse)