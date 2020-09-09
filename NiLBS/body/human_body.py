
import numpy as np

from NiLBS.skinning.util import redistribute_weights


class HumanBody:

    def __init__(self, body_dict=None, body_dict_path=None, active_bones=None):

        if body_dict_path is not None:

            body_dict = np.load(body_dict_path)

        self.vertex_template = body_dict['v_template']
        self.weights = body_dict['weights']
        self.faces = body_dict['f']
        self.joints = body_dict['J']
        self.bone_hierarchy = body_dict['kintree_table']

        if active_bones is not None:
            self.weights = redistribute_weights(self.weights, self.bone_hierarchy, active_bones)