from setuptools import setup
import os

long_description = open("README.md").read()

def read(rel_path: str) -> str:
      here = os.path.abspath(os.path.dirname(__file__))
      with open(os.path.join(here, rel_path)) as fp:
            return fp.read()


def get_version(rel_path: str) -> str:
      for line in read(rel_path).splitlines():
            if line.startswith("__version__"):
                  delim = '"' if '"' in line else "'"
                  return line.split(delim)[1]
      raise RuntimeError("Unable to find version string.")


setup(name='antspynet',
      version=get_version("antspynet/__init__.py"),
      description='A collection of deep learning architectures ported to the python language and tools for basic medical image processing.',
      long_description=long_description,
      long_description_content_type="text/markdown; charset=UTF-8; variant=GFM",
      url='https://github.com/ANTsX/ANTsPyNet',
      author='Nicholas J. Tustison and Brian B. Avants and Nick Cullen',
      author_email='ntustison@gmail.com',
      packages=['antspynet','antspynet/architectures','antspynet/utilities'],
      install_requires=['antspyx','keras','scikit-learn','tensorflow','tensorflow-probability','numpy','requests','statsmodels','matplotlib'],
      zip_safe=False)
