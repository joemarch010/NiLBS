
import numpy as np

from human_body_prior.body_model.body_model import BodyModel

from NiLBS.sampling.pose_sampler_amass import PoseSamplerAMASS


if __name__ == '__main__':

    bm_path = '../data/AMASS/body_models/smplh/female/model.npz'
    bd_path = '../data/AMASS/MPILimits/MPI_Limits/03099/lar1_poses.npz'
    output_path = '../../data/weight/weight_small_train.npz'
    n_frames = 100
    frame_step = 1
    frame_offset = 0

    bm = BodyModel(bm_path=bm_path)
    bdata = np.load(bd_path)

    psa = PoseSamplerAMASS(bm, bdata)




