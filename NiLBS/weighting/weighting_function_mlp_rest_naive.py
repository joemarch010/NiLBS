
from NiLBS.weighting.weighting_function import WeightingFunction

import numpy as np
import tensorflow.keras as keras


class WeightingFunctionMLPRestNaive(WeightingFunction):
    """


    Weighting function backed by a Keras regression network.
    Queries are performed using the naive pose encoding.


    """
    def __init__(self, model=None, model_path=None):

        if model_path is not None:

            self.model = keras.models.load_model(model_path)
        else:

            self.model = model

    def generate_query(self, x, pose):

        return x

    def generate_query_set(self, X, pose):

        return X

    def evaluate(self, x, pose):

        query = self.generate_query(x, pose)

        return self.model.predict(np.array([query]))[0]

    def evaluate_set(self, X, pose):

        query_set = self.generate_query_set(X, pose)

        return self.model.predict(query_set)
