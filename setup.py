from setuptools import setup

setup(name='antspynet',
      version='0.0',
      description='A collection of deep learning architectures ported to the python language and tools for basic medical image processing.',
      url='http://github.com/ntustison/antspynet',
      author='Nicholas J. Tustison and Brian B. Avants',
      author_email='ntustison@gmail.com',
      packages=['antspynet'],
      install_requires=['antspyx'],
      zip_safe=False)