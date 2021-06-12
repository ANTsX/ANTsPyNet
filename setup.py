from setuptools import setup

setup(name='antspynet',
      version='0.1.1',
      description='A collection of deep learning architectures ported to the python language and tools for basic medical image processing.',
      url='http://github.com/ntustison/antspynet',
      author='Nicholas J. Tustison and Brian B. Avants and Nick Cullen',
      author_email='ntustison@gmail.com',
      packages=['antspynet','antspynet/architectures','antspynet/utilities'],
      install_requires=['antspyx','keras','scikit-learn','tensorflow','tensorflow-probability','numpy','requests','statsmodels','matplotlib'],
      zip_safe=False)
