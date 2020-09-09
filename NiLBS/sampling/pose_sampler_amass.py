
import numpy as np

from NiLBS.pose.util import pose_from_smplh

class PoseSamplerAMASS:
    """


    Class for sampling pose data from an AMASS .npz file.


    """

    def __init__(self, body_model, bdata):
        """

        :param body_model: HumanBody
        :param bdata: .npz file, must contain the following attributes: 'poses'
        """
        self.v_template = body_model.vertex_template
        self.bone_hierachy = body_model.bone_hierarchy
        #self.j_regressor = body_model.J_regressor
        self.joints = body_model.joints
        self.bdata = bdata

    def sample_frames(self, n_frames=-1, step=1, offset=0):
        """

        :param n_frames: total number of frames to extract, -1 for all.
        :param step: step between each sampled frame.
        :param offset: frame to begin sampling at.
        :return: Numpy array-like, F, contains one pose for each frame.
        """
        poses = []
        total_frames = self.bdata['poses'].shape[0]
        count = 0

        for i in range(offset, total_frames, step):

            if count >= n_frames != -1:
                break

            root_orient = np.array([[0, 0, 0]])
            pose_body = self.bdata['poses'][i:i + 1, 3:66]
            pose_hand = self.bdata['poses'][i:i + 1, 66:]

            full_pose = np.concatenate((root_orient, pose_body), axis=1).transpose()
            pose = pose_from_smplh(self.v_template, full_pose, self.bone_hierachy, self.joints)

            poses.append(pose)

            count += 1

        result = np.array(poses)

        return result
