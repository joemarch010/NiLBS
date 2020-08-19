
import numpy as np

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from pose.util import pose_from_smplh

class PoseSamplerAMASS:
    """


    Class for sampling pose data from an AMASS .npz file.


    """

    def __init__(self, body_model, bdata):
        """

        :param body_model: human_body_prior.BodyModel
        :param bdata: .npz file, must contain the following attributes: 'poses'
        """
        self.v_template = c2c(body_model.v_template)[0]
        self.bone_hierachy = c2c(body_model.kintree_table)
        self.j_regressor = body_model.J_regressor
        self.bdata = bdata

    def sample_frames(self, n_frames=-1, step=1, offset=0):
        """

        :param n_frame: total number of frames to extract, -1 for all.
        :param step: step between each sampled frame.
        :param offset: frame to begin sampling at.
        :return: Numpy array-like, F, contains one pose for each frame.
        """
        poses = []
        total_frames = self.bdata['poses'].shape[0]

        for i in range(offset, total_frames, step):

            if i > n_frames != -1:
                break

            root_orient = self.bdata['poses'][i:i + 1, :3]
            pose_body = self.bdata['poses'][i:i + 1, 3:66]
            pose_hand = self.bdata['poses'][i:i + 1, 66:]

            full_pose = np.concatenate((root_orient, pose_body, pose_hand), axis=1).transpose()
            pose = pose_from_smplh(self.v_template, full_pose, self.bone_hierachy, self.j_regressor)

            poses.append(pose)

        result = np.array(poses)

        return result
