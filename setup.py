from setuptools import setup

setup(
   name='NiLBS',
   version='0.5.4',
   description='NiLBS 3D implementation.',
   author='Joe March',
   author_email='jgm45@cam.ac.uk',
   packages=['NiLBS',
             'NiLBS.body',
             'NiLBS.mesh',
             'NiLBS.occupancy',
             'NiLBS.occupancy.voxel',
             'NiLBS.pose',
             'NiLBS.sampling',
             'NiLBS.weighting'],
   install_requires=['trimesh', 'pyrender', 'tensorflow'], #external packages as dependencies
)
