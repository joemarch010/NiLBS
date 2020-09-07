
class HumanBody:

    def __init__(self, body_dict):

        self.vertex_template = body_dict['v_template']
        self.weights = body_dict['weights']
        self.faces = body_dict['f']
        self.joints = body_dict['J']
        self.bone_hierarchy = body_dict['kintree_table']