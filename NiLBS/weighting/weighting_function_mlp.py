

from NiLBS.weighting.weighting_function import WeightingFunction


class WeightingFunctionMLP(WeightingFunction):
    """


    Weighting function backed by an MLP


    """
    def __init__(self, mlp):

        self.mlp = mlp


    def generate_query(self, x, pose):

        return None

    def generate_query_set(self, X, pose):

        return None

    def evaluate(self, x, pose):

        return None

    def evaluate_set(self, X, pose):

        return None