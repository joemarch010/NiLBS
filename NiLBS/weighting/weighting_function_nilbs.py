

from NiLBS.weighting.weighting_function import WeightingFunction


class WeightingFunctionNiLBS(WeightingFunction):
    """


    Weighting function similar in form to that described in the NiLBS technical report.


    """
    def __init__(self, mlp):

        self.mlp = mlp
