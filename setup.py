from setuptools import setup

setup(
   name='NiLBS',
   version='0.5.9',
   description='NiLBS 3D implementation.',
   author='Joe March',
   author_email='jgm45@cam.ac.uk',
   packages=['NiLBS',
             'NiLBS.body',
             'NiLBS.demo',
             'NiLBS.mesh',
             'NiLBS.occupancy',
             'NiLBS.occupancy.voxel',
             'NiLBS.pose',
             'NiLBS.sampling',
             'NiLBS.skinning',
             'NiLBS.weighting'],
   package_data={'NiLBS': ['data/*.npz']},
   install_requires=['rtree', 'trimesh', 'pyrender', 'tensorflow']
)
