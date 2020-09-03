

from NiLBS.weighting.weighting_function_mlp import WeightingFunctionMLP


class WeightingFunctionNiLBS(WeightingFunctionMLP):
    """


    Weighting function similar in form to that described in the NiLBS technical report.


    """
    def __init__(self, mlp):

        self.mlp = mlp
