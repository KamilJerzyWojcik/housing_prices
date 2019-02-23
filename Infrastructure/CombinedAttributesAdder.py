from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, addBedroomsPerRoom = True):
        self.addBedroomsPerRoom = addBedroomsPerRoom

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        roomsPerFamily = X[:, rooms_ix] / X[:, household_ix]
        populationPerFamily = X[:, population_ix] / X[:, household_ix]
        if self.addBedroomsPerRoom:
            bedroomsPerRooms = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, roomsPerFamily, populationPerFamily, bedroomsPerRooms]
        else:
            return np.c_[X, roomsPerFamily, populationPerFamily]
