from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def ModelGet(TrainSet, predictor_model, name, paramGrid = {}, iterationRandom = 0):
    housingPrepared = TrainSet.drop("median_house_value", axis=1)
    housingLabels = TrainSet["median_house_value"].copy()
    grid_search = []

    #optymalizacja modelu
    if len(paramGrid) > 0 & iterationRandom == 0:
        grid_search = GridSearchCV(predictor_model,
                                   param_grid=paramGrid,
                                   cv=5,
                                   scoring="neg_mean_squared_error")
        grid_search.fit(housingPrepared.values, housingLabels)
        predictor_model = grid_search.best_estimator_
    elif len(paramGrid) > 0 & iterationRandom != 0:
        grid_search = RandomizedSearchCV(predictor_model,
                                         param_distributions=paramGrid,
                                         n_iter=iterationRandom,
                                         cv=5,
                                         scoring="neg_mean_squared_error")
        grid_search.fit(housingPrepared.values, housingLabels)
        predictor_model = grid_search.best_estimator_

    predictor_model.fit(housingPrepared.values, housingLabels)

    #sprawdzenie kilku danch modelu
    someDataPrepared = TrainSet.drop("median_house_value", axis=1).iloc[:5]
    someLabels = TrainSet["median_house_value"].iloc[:5]

    #sprawdzanie wydajnosci modelu
    housingPrediction = predictor_model.predict(housingPrepared)
    linMSE = mean_squared_error(housingLabels, housingPrediction)
    linRMSE = np.sqrt(linMSE)
    scores = cross_val_score(predictor_model, housingPrepared, housingLabels, scoring="neg_mean_squared_error", cv=10)
    linRMSEscores = np.sqrt(-scores)
    print("")
    print("...............", name, ":.TRAIN...............")
    if len(paramGrid) > 0:
        print("Najlepsze.parametry: ", grid_search.best_params_)
        print("........................................................")
    for pr, et in zip(predictor_model.predict(someDataPrepared), list(someLabels)):
        print("prognoza: ", round(pr, 2), ", Etykieta: ", round(et, 2), ", różnica %: ", round(100*np.abs((et-pr)/et), 2), "%")
    print("RMSE: ", linRMSE)
    print(".................................................")
    print("score RMSE średnia z 10: ", linRMSEscores.mean())
    print("score RMSE ochylenie standardowe: ", linRMSEscores.std())
    print("..................................................")
    return predictor_model


def PredictFinalData(TestSet, predictor, name):
    housingPrepared = TestSet.drop("median_house_value", axis=1)
    housingLabels = TestSet["median_house_value"].copy()

    #sprawdzenie kilku danch modelu
    someDataPrepared = TestSet.drop("median_house_value", axis=1).iloc[:5]
    someLabels = TestSet["median_house_value"].iloc[:5]

    #sprawdzanie wydajnosci modelu
    housingPrediction = predictor.predict(housingPrepared)
    linMSE = mean_squared_error(housingLabels, housingPrediction)
    linRMSE = np.sqrt(linMSE)
    scores = cross_val_score(predictor, housingPrepared, housingLabels, scoring="neg_mean_squared_error", cv=10)
    linRMSEscores = np.sqrt(-scores)
    print("")
    print("...............", name, ":.TEST................")
    for pr, et in zip(predictor.predict(someDataPrepared), list(someLabels)):
        print("prognoza: ", round(pr, 2), ", Etykieta: ", round(et, 2), ", różnica %: ", round(100*np.abs((et-pr)/et), 2), "%")
    print("RMSE: ", linRMSE)
    print(".................................................")
    print("score RMSE średnia z 10: ", linRMSEscores.mean())
    print("score RMSE ochylenie standardowe: ", linRMSEscores.std())
    print("..................................................")
    return predictor