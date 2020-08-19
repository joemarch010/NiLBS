

from weighting.weight_function import WeightFunction


class WeightingFunctionMLP(WeightFunction):
    """


    Weighting function backed by an MLP


    """
    def __init__(self, mlp):

        self.mlp = mlp
