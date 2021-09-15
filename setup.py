"""
Author: Viswambhar Yasa

Setup file to download the Automatic Differentiation package
"""
#Download required packages
from setuptools import setup

setup(name='autodiff',
      version='1.0',
      description='Automatic differentiation framework',

      packages=['autodiff', 'autodiff.core'],
      license='None',
      install_requires=['numpy'],
      )
