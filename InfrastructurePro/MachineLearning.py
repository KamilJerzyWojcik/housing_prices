from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def ModelGet(TrainSet, predictor, name):
    housingPrepared = TrainSet.drop("median_house_value", axis=1)
    housingLabels = TrainSet["median_house_value"].copy()

    lin_reg = predictor
    lin_reg.fit(housingPrepared.values, housingLabels)

    #sprawdzenie kilku danch modelu
    someDataPrepared = TrainSet.drop("median_house_value", axis=1).iloc[:5]
    someLabels = TrainSet["median_house_value"].iloc[:5]

    #sprawdzanie wydajnosci modelu
    housingPrediction = lin_reg.predict(housingPrepared)
    linMSE = mean_squared_error(housingLabels, housingPrediction)
    linRMSE = np.sqrt(linMSE)
    scores = cross_val_score(lin_reg, housingPrepared, housingLabels, scoring="neg_mean_squared_error", cv=10)
    linRMSEscores = np.sqrt(-scores)
    print("")
    print("...............", name, ":...................")
    for pr, et in zip(lin_reg.predict(someDataPrepared), list(someLabels)):
        print("prognoza: ", pr, ", Etykieta: ", et, ", różnica %: ", round(100*np.abs((et-pr)/et), 3), "%")

    print("RMSE: ", linRMSE)
    print(".................................................")
    print("score RMSE średnia z 10: ", linRMSEscores.mean())
    print("score RMSE ochylenie standardowe: ", linRMSEscores.std())
    print("..................................................")
    return lin_reg
