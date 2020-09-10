import pyrender

from NiLBS.demo.scene import make_default_model_pose_scene
from NiLBS.demo.scene import make_default_joint_view_scene


imw, imh=1000, 1000

scene = make_default_model_pose_scene()

viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)
viewer.run()
