from distutils.core import setup
import versioneer
from setuptools import find_packages

setup(name='booster',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Gradient boosting with arbitrary scikit-learn regressors.',
      author='Jason Rudy',
      author_email='jcrudy@gmail.com',
      packages=find_packages(),
      install_requires = ['scikit-learn']
     )