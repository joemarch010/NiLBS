

from NiLBS.weighting.weighting_function import WeightingFunction


class WeightingFunctionMLP(WeightingFunction):
    """


    Weighting function backed by an MLP


    """
    def __init__(self, mlp):

        self.mlp = mlp
