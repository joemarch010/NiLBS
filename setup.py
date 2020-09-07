from setuptools import setup

setup(
   name='NiLBS',
   version='0.5',
   description='NiLBS 3D implementation.',
   author='Joe March',
   author_email='jgm45@cam.ac.uk',
   packages=['NiLBS'],  #same as name
   install_requires=['trimesh', 'pyrender', 'tensorflow'], #external packages as dependencies
)
